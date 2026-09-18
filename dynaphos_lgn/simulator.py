"""End-to-end differentiable phosphene simulator for LGN stimulation.

Same shape as Dynaphos's V1 simulator -- stimulation parameters in, a
rendered percept out, with a leaky-integrator state between frames -- so
an encoder trained against one can be pointed at the other. The
temporal-dynamics, thresholding and brightness machinery is imported
from `dynaphos.simulator` unchanged; what is new here is spatial:
recruitment over real atlas voxels, grouped by layer, with phosphene
geometry from the atlas's local Jacobian.

Gradients reach amplitude, pulse width and frequency through the image,
the recruitment weights and the activation state -- never through voxel
positions or layer labels, which are fixed buffers.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from dynaphos.image_processing import scale_image
from dynaphos.simulator import (Activation, ActivationThreshold, Brightness,
                                Trace)
from dynaphos.utils import (get_data_kwargs, get_deg2pix_coeff, print_stats,
                            set_deterministic, to_numpy, to_tensor)
from dynaphos_lgn.current_spread import RecruitmentKernel
from dynaphos_lgn.electrodes import LGNElectrodeArray
from dynaphos_lgn.params import optional, require
from dynaphos_lgn.receptive_fields import center_radius_deg
from dynaphos_lgn.rendering import EllipseGaussianRenderer, ScatterRenderer


class LGNSize:
    """Phosphene sigmas in degrees, from current via the local Jacobian.

    Its own class so the size model stays swappable: anything exposing
    `update` and `get` can replace it.
    """

    def __init__(self, shape: Tuple[int, ...],
                 kernel: RecruitmentKernel,
                 major_deg_per_mm: np.ndarray, minor_deg_per_mm: np.ndarray,
                 data_kwargs: dict, verbose: bool = False):
        self.kernel = kernel
        self.shape = shape
        self.verbose = verbose
        self.data_kwargs = data_kwargs

        major = np.asarray(major_deg_per_mm, dtype=float).ravel()
        minor = np.asarray(minor_deg_per_mm, dtype=float).ravel()

        # A non-finite magnification means the atlas has no usable
        # Jacobian at that electrode's voxel. Zeroing it is the only
        # fallback that does not invent a magnification the atlas never
        # measured -- but a zero sigma renders an invisible phosphene,
        # so the electrode drops out of the percept entirely. Say so
        # rather than letting it vanish, and keep the mask queryable.
        self.no_magnification = ~(np.isfinite(major) & np.isfinite(minor))
        n_bad = int(self.no_magnification.sum())
        if n_bad:
            logging.warning(
                "%d of %d electrodes have no finite local magnification "
                "and will render as zero-size (invisible) phosphenes; "
                "their recruitment is still computed. Electrode indices: "
                "%s. See `.no_magnification` for the full mask.",
                n_bad, self.no_magnification.size,
                np.array2string(np.flatnonzero(self.no_magnification),
                                threshold=20))

        self.major = to_tensor(
            np.reshape(np.where(self.no_magnification, 0.0, major),
                       shape[-3:]), **data_kwargs)
        self.minor = to_tensor(
            np.reshape(np.where(self.no_magnification, 0.0, minor),
                       shape[-3:]), **data_kwargs)
        self.sigma_major = None
        self.sigma_minor = None
        self.reset()

    def reset(self):
        self.sigma_major = torch.zeros(self.shape, **self.data_kwargs)
        self.sigma_minor = torch.zeros(self.shape, **self.data_kwargs)

    def update(self, amplitude_ua: torch.Tensor):
        sigma_mm = self.kernel.sigma_mm(amplitude_ua)
        self.sigma_major = sigma_mm * self.major
        self.sigma_minor = sigma_mm * self.minor
        print_stats('sigma major (deg)', self.sigma_major, self.verbose)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sigma_major, self.sigma_minor


class LGNPhospheneSimulator:
    """Differentiable LGN phosphene simulator.

    :param params: Dynaphos-style nested parameter dict. See
        `config/params_lgn.yaml` for the LGN defaults and what each
        block is grounded in.
    :param electrode_array: A built `LGNElectrodeArray`.
    :param rng: Numpy generator, used for threshold initialisation.

    `rendering.adaptive_scatter_bandwidth` decides whether the scatter
    splat bandwidth is recomputed from the live recruitment weights each
    frame, instead of using the precomputed floor. It costs little and
    matters at low amplitudes, where only a tight cluster of voxels is
    active.
    """

    def __init__(self, params: dict, electrode_array: LGNElectrodeArray,
                 rng: Optional[np.random.Generator] = None):
        self.params = params
        self.array = electrode_array
        self.data_kwargs = get_data_kwargs(params)
        self.verbose = require(params, 'run.print_stats')
        self.adaptive_scatter_bandwidth = require(
            params, 'rendering.adaptive_scatter_bandwidth')

        rng = np.random.default_rng() if rng is None else rng
        set_deterministic(require(params, 'run.seed'))
        self.deg2pix_coeff = get_deg2pix_coeff(params['run'])

        self.kernel = electrode_array.kernel
        self.num_phosphenes = electrode_array.n_electrodes

        batch_size = require(params, 'run.batch_size')
        if batch_size:
            self.shape = (batch_size, self.num_phosphenes, 1, 1)
            self._electrode_dimension = 1
        else:
            self.shape = (self.num_phosphenes, 1, 1)
            self._electrode_dimension = 0
        self.batch_size = batch_size

        self._warn_about_out_of_view_electrodes()
        self._build_renderers()
        self._build_recruitment_buffers()

        self.activation = Activation(params, self.shape, verbose=self.verbose)
        self.trace = Trace(params, self.shape)
        self.brightness = Brightness(params, self.shape)
        self.threshold = ActivationThreshold(params, self.shape, rng)
        self.size = LGNSize(
            self.shape, self.kernel,
            electrode_array.local_magnification.major_deg_per_mm,
            electrode_array.local_magnification.minor_deg_per_mm,
            self.data_kwargs, self.verbose)
        # Alongside `out_of_view`: the other way an electrode can be
        # present in the model yet absent from the rendered image.
        self.no_magnification = self.size.no_magnification

        self.effective_charge_per_second = None
        self.layer_activation = None
        self.recruited_cells = None

        self._sampling_mask = None
        self._phosphene_centers = None
        self._sampling_method = require(params, 'sampling.sampling_method')
        self._pulse_width = (require(params, 'default_stim.pw_default')
                             * torch.ones(self.shape, **self.data_kwargs))
        self._frequency = (require(params, 'default_stim.freq_default')
                           * torch.ones(self.shape, **self.data_kwargs))
        self._zero = to_tensor(0, **self.data_kwargs)

        self.soft_threshold = bool(
            require(params, 'thresholding.soft_threshold'))
        # Match the spread already in the model rather than adding a new
        # free parameter.
        self._soft_threshold_slope = optional(
            params, 'thresholding.soft_threshold_slope',
            1.0 / max(require(params, 'thresholding.activation_threshold_sd'),
                      1e-30))

        self.soft_threshold_baseline = optional(
            params, 'thresholding.soft_threshold_baseline', 'zero_activation')
        self.reset()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    @property
    def electrode_dimension(self) -> int:
        """Which tensor dimension numbers the electrodes.

        0 without batching, 1 with. Public because
        `dynaphos_lgn.bilateral` has to split and concatenate along it.
        """
        return self._electrode_dimension

    @property
    def hemisphere(self) -> str:
        """Which nucleus this simulator stimulates."""
        return self.array.hemisphere

    def _warn_about_out_of_view_electrodes(self):
        """
        Say so when electrodes sit outside the rendered field of view.
        """
        half = require(self.params, 'run.view_angle') / 2
        origin = np.asarray(require(self.params, 'run.origin'), dtype=float)
        offset = np.abs(self.array.vf_xy - origin)
        outside = np.any(offset > half, axis=1) | ~np.isfinite(
            self.array.vf_xy).all(axis=1)
        self.out_of_view = outside
        if outside.any():
            logging.warning(
                "%d of %d electrodes lie outside the rendered %.1f deg "
                "field of view and will contribute nothing to the image "
                "(their recruitment is still computed). Widen "
                "run.view_angle or lower electrodes.max_eccentricity_deg "
                "to see them.", int(outside.sum()), self.num_phosphenes,
                2 * half)

    def _build_renderers(self):
        array = self.array
        mode = array.render_mode
        self.ellipse_index = np.flatnonzero(mode == 'ellipse')
        self.scatter_index = np.flatnonzero(mode == 'scatter')

        self.ellipse_renderer = None
        if len(self.ellipse_index):
            self.ellipse_renderer = EllipseGaussianRenderer(
                array.vf_xy[self.ellipse_index],
                array.local_magnification.orientation_rad[self.ellipse_index],
                self.params, self.data_kwargs)

        self.scatter_renderer = None
        if len(self.scatter_index):
            self.scatter_renderer = ScatterRenderer(
                array.scatter_points_xy[self.scatter_index],
                array.vf_xy[self.scatter_index],
                self.params,
                array.scatter_patch_radius_deg()[self.scatter_index],
                array.scatter_bandwidth_floor_deg[self.scatter_index],
                # Ceiling: a fraction of the patch radius, so the splat
                # can never approach the point cloud's own extent.
                (require(self.params,
                         'rendering.scatter_bandwidth_ceiling_fraction')
                 * array.scatter_patch_radius_deg()[self.scatter_index]),
                self.data_kwargs)

    def _build_recruitment_buffers(self):
        array = self.array
        hist = array.distance_histograms()
        self.layer_names = hist['layer_names']
        self._bin_centers = to_tensor(hist['bin_centers_mm'],
                                      **self.data_kwargs)
        self._layer_counts = to_tensor(hist['counts'], **self.data_kwargs)
        self._layer_cells = to_tensor(hist['cells'], **self.data_kwargs)

        if len(self.scatter_index):
            self._scatter_distance = to_tensor(
                array.scatter_distance_mm, **self.data_kwargs)
            self._scatter_mask = to_tensor(
                array.scatter_valid[self.scatter_index].astype(float),
                **self.data_kwargs)
            self._scatter_points = to_tensor(
                array.scatter_points_xy[self.scatter_index],
                **self.data_kwargs)

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    def reset(self):
        self.activation.reset()
        self.trace.reset()
        self.size.reset()

    def to_tensor(self, x) -> torch.Tensor:
        return to_tensor(x, **self.data_kwargs)

    def get_current(self, amplitude: torch.Tensor, frequency: torch.Tensor,
                    pulse_width: torch.Tensor) -> torch.Tensor:
        """Effective charge per second from a square-wave pulse train.

        Identical in form to Dynaphos's V1 version, leak current
        included. Rheobase and pulse-width dependence are transplanted
        from human V1.
        """
        leak = self.trace.get() + require(self.params,
                                          'thresholding.rheobase')
        charge_per_s = torch.relu((amplitude - leak) * pulse_width * frequency)
        self.effective_charge_per_second = charge_per_s
        print_stats('charge per second', charge_per_s, self.verbose)
        return charge_per_s

    def recruitment(self, amplitude_ua: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-layer recruitment totals for the current amplitude.

        :param amplitude_ua: (..., n_electrodes, 1, 1) amplitudes, uA.
        :return: (per-layer summed weight, per-layer recruited cells),
            each (..., n_electrodes, n_layers).
        """
        amp = amplitude_ua.reshape(*amplitude_ua.shape[:-1])   # (..., n, 1)
        weights = self.kernel.weights(self._bin_centers, amp)  # (..., n, bins)
        counts = self._layer_counts                            # (n, L, bins)
        cells = self._layer_cells
        per_layer_weight = torch.einsum('...nb,nlb->...nl', weights, counts)
        per_layer_cells = torch.einsum('...nb,nlb->...nl', weights, cells)
        return per_layer_weight, per_layer_cells

    def update(self, amplitude: torch.Tensor,
               pulse_width: Optional[torch.Tensor] = None,
               frequency: Optional[torch.Tensor] = None):
        """Advance the tissue state by one frame.

        :param amplitude: Stimulation amplitude per electrode, in the
            config's units (amperes). The conversion to the
            microamperes `current_spread` works in happens here, once,
            via ``run.current_unit_to_ua``.
        """
        if pulse_width is None:
            pulse_width = self._pulse_width
        if frequency is None:
            frequency = self._frequency

        amplitude = amplitude.reshape(self.shape)
        charge_per_s = self.get_current(amplitude,
                                        frequency.reshape(self.shape),
                                        pulse_width.reshape(self.shape))
        self.activation.update(charge_per_s)
        self.trace.update(charge_per_s)

        amplitude_ua = amplitude * require(self.params,
                                           'run.current_unit_to_ua')
        self.size.update(amplitude_ua)
        self.layer_activation, self.recruited_cells = \
            self.recruitment(amplitude_ua)
        self.brightness.update(self.activation.get())
        self._last_amplitude_ua = amplitude_ua

    def spatial_activation(self) -> torch.Tensor:
        """Per-electrode activation maps over the visual field, peak 1.

        Ellipse-rendered and scatter-rendered electrodes are computed
        separately and reassembled in the original electrode order, so
        downstream code never has to know which is which.
        """
        res_x, res_y = require(self.params, 'run.resolution')
        lead = self.shape[:-3]
        out = torch.zeros((*lead, self.num_phosphenes, res_y, res_x),
                          **self.data_kwargs)

        if self.ellipse_renderer is not None:
            idx = torch.as_tensor(self.ellipse_index,
                                  device=out.device, dtype=torch.long)
            sa = self.size.sigma_major.index_select(
                self._electrode_dimension, idx)
            sb = self.size.sigma_minor.index_select(
                self._electrode_dimension, idx)
            rendered = self.ellipse_renderer.render(sa, sb)
            out = out.index_copy(self._electrode_dimension, idx, rendered)

        if self.scatter_renderer is not None:
            idx = torch.as_tensor(self.scatter_index,
                                  device=out.device, dtype=torch.long)
            amp = self._last_amplitude_ua.index_select(
                self._electrode_dimension, idx).reshape(
                *lead, len(self.scatter_index), 1)
            w = self.kernel.weights(self._scatter_distance, amp)
            w = w * self._scatter_mask
            h = self.scatter_renderer.bandwidth_deg(
                w, self._scatter_points,
                adaptive=self.adaptive_scatter_bandwidth)
            rendered = self.scatter_renderer.render(w, h)
            out = out.index_copy(self._electrode_dimension, idx, rendered)

        return out

    def __call__(self, amplitude: torch.Tensor,
                 pulse_width: Optional[torch.Tensor] = None,
                 frequency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Render one frame of simulated percept."""
        self.update(amplitude, pulse_width, frequency)
        activation_map = self.spatial_activation()
        intensity = self.brightness.get() * self.detection_probability()
        summed = torch.sum(intensity * activation_map,
                           dim=self._electrode_dimension)
        return summed.clamp(0, 1)

    def detection_probability(self) -> torch.Tensor:
        """Gate on tissue activation versus the per-electrode threshold.

        A logistic psychometric curve by default
        (``thresholding.soft_threshold``), or Dynaphos's hard 0/1 mask
        when that is off.

        The logistic is baseline-corrected unless
        ``thresholding.soft_threshold_baseline`` says otherwise, because
        the raw logistic does **not** vanish at zero activation. The
        threshold sits only ``mu/sd = 1.36`` slope-widths above zero, so
        a silent electrode gets ``p(detect) = 0.204`` and an
        unstimulated array renders a visible glow. Subtracting the
        zero-activation value and renormalising fixes that without
        flattening the curve: ``dp/da`` stays strictly positive
        everywhere, including at ``a = 0``, which is the whole reason
        the soft threshold exists.
        """
        activation = self.activation.get()
        threshold = self.threshold.get()
        margin = activation - threshold
        if not self.soft_threshold:
            return torch.greater(margin, self._zero).to(margin.dtype)
        probability = torch.sigmoid(self._soft_threshold_slope * margin)
        if self.soft_threshold_baseline == 'none':
            return probability
        # p(0), i.e. the curve evaluated at zero activation. A tensor,
        # not a constant: `threshold` is sampled per electrode.
        floor = torch.sigmoid(self._soft_threshold_slope * -threshold)
        return ((probability - floor)
                / torch.clamp(1.0 - floor, min=1e-12)).clamp(min=0.0)

    # ------------------------------------------------------------------
    # Stimulus sampling (unchanged in spirit from Dynaphos)
    # ------------------------------------------------------------------
    @property
    def phosphene_maps(self) -> torch.Tensor:
        """Distance from every pixel to each phosphene centre, degrees.

        Used only for stimulus sampling. Built from the electrodes'
        visual-field positions, independently of which renderer each
        electrode uses.
        """
        if getattr(self, '_phosphene_maps', None) is None:
            from dynaphos_lgn.rendering import visual_field_grid
            gx, gy = visual_field_grid(self.params, **self.data_kwargs)
            centers = to_tensor(self.array.vf_xy, **self.data_kwargs)
            dx = gx[None] - centers[:, 0, None, None]
            dy = gy[None] - centers[:, 1, None, None]
            self._phosphene_maps = torch.sqrt(dx ** 2 + dy ** 2)
        return self._phosphene_maps

    @property
    def phosphene_centers(self) -> torch.Tensor:
        if self._phosphene_centers is None:
            self._phosphene_centers = self.phosphene_maps.flatten(
                start_dim=1).argmin(dim=-1)
        return self._phosphene_centers

    @property
    def sampling_mask(self) -> torch.Tensor:
        """Which pixels fall inside each electrode's sampling region.

        In ``receptive_fields`` mode the radius is the LGN receptive-field
        centre radius at that electrode's eccentricity -- degrees
        directly, not millimetres of tissue as in the V1 model.
        """
        if self._sampling_mask is None:
            if self._sampling_method == 'receptive_fields':
                radius = center_radius_deg(
                    self.array.eccentricity_deg, self.params,
                    require(self.params, 'sampling.rf_cell_class'),
                    require(self.params, 'sampling.rf_form'))
                radius = radius * require(self.params,
                                          'sampling.rf_size_scale')
                radius = to_tensor(
                    np.reshape(radius, (-1, 1, 1)), **self.data_kwargs)
                self._sampling_mask = torch.less(self.phosphene_maps, radius)
            elif self._sampling_method == 'center':
                maps = self.phosphene_maps
                flat = (torch.arange(maps.shape[0], device=maps.device)
                        * maps.shape[-2] * maps.shape[-1])
                mask = torch.zeros_like(maps)
                mask.flatten()[self.phosphene_centers + flat] = 1
                self._sampling_mask = mask
            else:
                raise NotImplementedError(
                    f"Unknown sampling method {self._sampling_method!r}.")
        return self._sampling_mask

    def sample_stimulus(self, activation_mask, rescale: bool = False
                        ) -> torch.Tensor:
        """Turn an image of desired regional intensity into per-electrode
        stimulation amplitudes."""
        if isinstance(activation_mask, np.ndarray):
            dtype = activation_mask.dtype
            activation_mask = self.to_tensor(activation_mask)
            if dtype == np.dtype('uint8') or activation_mask.max() > 1:
                activation_mask = scale_image(activation_mask, 1 / 255)
        if self._sampling_method == 'receptive_fields':
            out = torch.amax(self.sampling_mask * activation_mask,
                             dim=(-2, -1))
        else:
            out = activation_mask.flatten(-2)[..., self.phosphene_centers]
        if rescale:
            out = out * require(self.params, 'sampling.stimulus_scale')
        return out

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, torch.Tensor]:
        return {
            'brightness': self.brightness.get(),
            'sigma_major': self.size.sigma_major,
            'sigma_minor': self.size.sigma_minor,
            'activation': self.activation.get(),
            'trace': self.trace.get(),
            'threshold': self.threshold.get(),
            'detection_probability': self.detection_probability(),
            'effective_charge_per_second': self.effective_charge_per_second,
            'layer_activation': self.layer_activation,
            'recruited_cells': self.recruited_cells,
        }

    def layer_activation_table(self) -> Dict[str, np.ndarray]:
        """Last frame's per-layer recruitment, keyed by layer name."""
        if self.layer_activation is None:
            raise RuntimeError("Call the simulator at least once first.")
        values = to_numpy(self.layer_activation.detach())
        return {name: values[..., i]
                for i, name in enumerate(self.layer_names)}

    def describe(self) -> str:
        """The electrode array's configuration and per-electrode flags."""
        return self.array.describe()
