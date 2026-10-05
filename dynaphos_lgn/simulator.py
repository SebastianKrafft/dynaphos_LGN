"""Differentiable phosphene simulator for LGN stimulation.

Same interface as Dynaphos's V1 simulator, whose temporal dynamics,
thresholding and brightness classes it reuses. What is LGN-specific is
spatial: recruitment over atlas voxels, grouped by layer, and phosphene
geometry from the local magnification. Gradients reach amplitude, pulse
width and frequency, not electrode positions.
"""
from __future__ import annotations

import logging
import os
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
from dynaphos_lgn.params import (ParameterError, class_membership, optional,
                                 require)
from dynaphos_lgn.receptive_fields import center_radius_deg
from dynaphos_lgn.rendering import (EllipseGaussianRenderer, ScatterRenderer,
                                    visual_field_grid)


class LGNSize:
    """Phosphene sigmas in degrees, from current via the local
    magnification. Anything with `update` and `get` can replace it."""

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

        # No usable magnification: render at zero size, and say so.
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

    :param params: Nested parameter dict, see `config/params_lgn.yaml`.
    :param electrode_array: A built `LGNElectrodeArray`.
    :param rng: Numpy generator, used for threshold initialisation.

    `safety_limit_ua` (from `default_stim.safety_limit_ua`) is for
    whatever chooses the amplitudes; the forward pass does not clamp.
    """

    def __init__(self, params: dict, electrode_array: LGNElectrodeArray,
                 rng: Optional[np.random.Generator] = None):
        self.params = params
        self.array = electrode_array
        self.safety_limit_ua = float(
            require(params, 'default_stim.safety_limit_ua'))
        if not 0 < self.safety_limit_ua <= electrode_array.max_current_ua:
            raise ParameterError(
                f"default_stim.safety_limit_ua is {self.safety_limit_ua} "
                f"uA; it must be positive and at most "
                f"electrodes.max_current_ua ({electrode_array.max_current_ua}"
                f" uA), the current the array's neighbourhoods are "
                f"precomputed for.")
        self.data_kwargs = get_data_kwargs(params)
        self.verbose = require(params, 'run.print_stats')
        self.adaptive_scatter_bandwidth = require(
            params, 'rendering.adaptive_scatter_bandwidth')

        rng = np.random.default_rng() if rng is None else rng
        # Deterministic mode refuses cuBLAS matmuls on CUDA without this
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
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
        self._build_class_mixture()
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
        self.no_magnification = self.size.no_magnification

        self.effective_charge_per_second = None
        self.layer_activation = None
        self.recruited_cells = None
        self.class_fraction = None

        self._sampling_mask = None
        self._phosphene_maps = None
        self._phosphene_centers = None
        self._sampling_method = require(params, 'sampling.sampling_method')
        self._pulse_width = (require(params, 'default_stim.pw_default')
                             * torch.ones(self.shape, **self.data_kwargs))
        self._frequency = (require(params, 'default_stim.freq_default')
                           * torch.ones(self.shape, **self.data_kwargs))
        self._zero = to_tensor(0, **self.data_kwargs)

        self.soft_threshold = bool(
            require(params, 'thresholding.soft_threshold'))
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
        """Tensor dimension of the electrodes: 0 unbatched, 1 batched."""
        return self._electrode_dimension

    @property
    def hemisphere(self) -> str:
        """Which nucleus this simulator stimulates."""
        return self.array.hemisphere

    # The same per-electrode views `BilateralLGNSimulator` offers, so
    # callers need not care how many nuclei they hold.
    @property
    def simulators(self) -> Dict[str, 'LGNPhospheneSimulator']:
        return {self.hemisphere: self}

    @property
    def electrode_slices(self) -> Dict[str, slice]:
        return {self.hemisphere: slice(0, self.num_phosphenes)}

    @property
    def hemisphere_of_electrode(self) -> np.ndarray:
        return np.full(self.num_phosphenes, self.hemisphere)

    @property
    def vf_xy(self) -> np.ndarray:
        """(n, 2) visual-field position of every electrode, degrees."""
        return self.array.vf_xy

    def phosphene_sigma_deg(self, current_ua):
        """(major, minor) phosphene sigmas in degrees."""
        return self.array.phosphene_sigma_deg(current_ua)

    def _warn_about_out_of_view_electrodes(self):
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

    def _build_class_mixture(self):
        """Layer-to-class membership and per-class receptive-field widths
        for the 'class_mixture' percept; no classes for 'single'."""
        model = require(self.params, 'percept.model')
        if model not in ('single', 'class_mixture'):
            raise ParameterError(f"Unknown percept.model {model!r}; expected "
                                 f"'single' or 'class_mixture'.")
        self.classes = ([] if model == 'single' else list(
            optional(self.params, 'percept.classes', None)
            or require(self.params, 'cell_classes')))
        if not self.classes:
            return
        array = self.array
        membership = class_membership(self.params, array.layer_ids,
                                      self.classes)
        self._membership = to_tensor(membership, **self.data_kwargs)

        scale = require(self.params, 'percept.rf_blur_scale')
        form = require(self.params, 'percept.rf_form')
        ecc = np.nan_to_num(array.eccentricity_deg)
        self.rf_sigma_deg = np.stack(
            [scale * center_radius_deg(ecc, self.params, c, form)
             for c in self.classes], axis=-1) / np.sqrt(2.0)
        self._rf_sigma = to_tensor(self.rf_sigma_deg, **self.data_kwargs)

        point_layer = array.candidate_layer[:, array.scatter_index]
        onehot = (point_layer[..., None] == array.layer_ids).astype(
            np.float32)
        self._point_class = onehot @ membership   # (n, points, classes)

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
            idx = self.scatter_index
            patch_radius = array.scatter_patch_radius_deg()[idx]
            ceiling_fraction = require(
                self.params, 'rendering.scatter_bandwidth_ceiling_fraction')
            # Room for the receptive-field widening
            rf_sigma = (self.rf_sigma_deg[idx].max(axis=-1) if self.classes
                        else np.zeros(len(idx)))
            self.scatter_renderer = ScatterRenderer(
                array.scatter_points_xy[idx],
                array.vf_xy[idx],
                self.params,
                patch_radius + 3 * rf_sigma,
                array.scatter_bandwidth_floor_deg[idx],
                ceiling_fraction * patch_radius,
                self.data_kwargs,
                extra_blur_deg=float(rf_sigma.max()))

    def _build_recruitment_buffers(self):
        array = self.array
        hist = array.distance_histograms()
        self.layer_names = hist['layer_names']
        self._bin_centers = to_tensor(hist['bin_centers_mm'],
                                      **self.data_kwargs)
        self._layer_counts = to_tensor(hist['counts'], **self.data_kwargs)
        self._layer_cells = to_tensor(hist['cells'], **self.data_kwargs)

        if len(self.scatter_index):
            if self.classes:
                self._scatter_point_class = to_tensor(
                    self._point_class[self.scatter_index],
                    **self.data_kwargs)
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
        """Effective charge per second from a square-wave pulse train,
        as in Dynaphos's V1 model."""
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

    def _as_input(self, x) -> torch.Tensor:
        """A per-electrode input, or a scalar for all electrodes, on the
        simulator's device and dtype in its state shape; tensors keep
        their gradient."""
        x = (x.to(**self.data_kwargs) if isinstance(x, torch.Tensor)
             else torch.as_tensor(x, **self.data_kwargs))
        if x.numel() == 1:
            return x.reshape(()).expand(self.shape)
        return x.reshape(self.shape)

    def update(self, amplitude: torch.Tensor,
               pulse_width: Optional[torch.Tensor] = None,
               frequency: Optional[torch.Tensor] = None):
        """Advance the tissue state by one frame.

        :param amplitude: Stimulation amplitude per electrode, in amperes
            (converted to uA via ``run.current_unit_to_ua``).
        """
        if pulse_width is None:
            pulse_width = self._pulse_width
        if frequency is None:
            frequency = self._frequency

        amplitude = self._as_input(amplitude)
        charge_per_s = self.get_current(amplitude,
                                        self._as_input(frequency),
                                        self._as_input(pulse_width))
        self.activation.update(charge_per_s)
        self.trace.update(charge_per_s)

        amplitude_ua = amplitude * require(self.params,
                                           'run.current_unit_to_ua')
        self.size.update(amplitude_ua)
        self.layer_activation, self.recruited_cells = \
            self.recruitment(amplitude_ua)
        if self.classes:
            class_cells = self.recruited_cells @ self._membership
            total = class_cells.sum(dim=-1, keepdim=True)
            self.class_fraction = class_cells / torch.clamp(
                total, min=torch.finfo(total.dtype).tiny)
        self.brightness.update(self.activation.get())
        self._last_amplitude_ua = amplitude_ua

    def spatial_activation(self) -> torch.Tensor:
        """Per-electrode activation maps over the visual field, peak 1,
        in the original electrode order."""
        res_x, res_y = require(self.params, 'run.resolution')
        lead = self.shape[:-3]
        out = torch.zeros((*lead, self.num_phosphenes, res_y, res_x),
                          **self.data_kwargs)
        dim = self._electrode_dimension

        if self.ellipse_renderer is not None:
            idx = torch.as_tensor(self.ellipse_index,
                                  device=out.device, dtype=torch.long)
            sa = self.size.sigma_major.index_select(dim, idx)
            sb = self.size.sigma_minor.index_select(dim, idx)

            def render_ellipse(rf, c):
                rf = None if rf is None else rf[:, None, None]
                return self.ellipse_renderer.render(_widen(sa, rf),
                                                    _widen(sb, rf))
            out = out.index_copy(dim, idx, self._mix(idx, render_ellipse))

        if self.scatter_renderer is not None:
            idx = torch.as_tensor(self.scatter_index,
                                  device=out.device, dtype=torch.long)
            amp = self._last_amplitude_ua.index_select(dim, idx).reshape(
                *lead, len(self.scatter_index), 1)
            w = self.kernel.weights(self._scatter_distance, amp)
            w = w * self._scatter_mask

            def render_scatter(rf, c):
                weights = w if c is None else w * self._scatter_point_class[
                    ..., c]
                h = self.scatter_renderer.bandwidth_deg(
                    weights, self._scatter_points,
                    adaptive=self.adaptive_scatter_bandwidth)
                return self.scatter_renderer.render(weights, _widen(h, rf))
            out = out.index_copy(dim, idx, self._mix(idx, render_scatter))

        return out

    def _mix(self, idx: torch.Tensor, render) -> torch.Tensor:
        """`render(rf_sigma, class_index)` for the electrodes `idx`: once,
        class-blind, for the 'single' percept; otherwise summed over the
        classes, weighted by each one's share of the recruited cells."""
        if not self.classes:
            return render(None, None)
        fraction = self.class_fraction.index_select(
            self._electrode_dimension, idx)
        rf_sigma = self._rf_sigma.index_select(0, idx)
        return sum(fraction[..., c, None, None] * render(rf_sigma[:, c], c)
                   for c in range(len(self.classes)))

    def __call__(self, amplitude: torch.Tensor,
                 pulse_width: Optional[torch.Tensor] = None,
                 frequency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Render one frame of simulated percept."""
        self.update(amplitude, pulse_width, frequency)
        return self.unclamped_percept().clamp(0, 1)

    def unclamped_percept(self) -> torch.Tensor:
        """The current percept, summed over electrodes but not clamped,
        so several nuclei can be added before the clamp."""
        intensity = self.brightness.get() * self.detection_probability()
        return torch.sum(intensity * self.spatial_activation(),
                         dim=self._electrode_dimension)

    def detection_probability(self) -> torch.Tensor:
        """Probability that each phosphene is seen, from activation versus
        the per-electrode threshold.

        A logistic curve when ``thresholding.soft_threshold`` is on,
        Dynaphos's hard step otherwise. With the 'zero_activation'
        baseline the logistic is shifted and rescaled so that p(0) = 0.
        """
        activation = self.activation.get()
        threshold = self.threshold.get()
        margin = activation - threshold
        if not self.soft_threshold:
            return torch.greater(margin, self._zero).to(margin.dtype)
        probability = torch.sigmoid(self._soft_threshold_slope * margin)
        if self.soft_threshold_baseline == 'none':
            return probability
        # p(0), per electrode
        floor = torch.sigmoid(self._soft_threshold_slope * -threshold)
        return ((probability - floor)
                / torch.clamp(1.0 - floor, min=1e-12)).clamp(min=0.0)

    # ------------------------------------------------------------------
    # Stimulus sampling
    # ------------------------------------------------------------------
    @property
    def phosphene_maps(self) -> torch.Tensor:
        """Distance from every pixel to each phosphene centre, degrees."""
        if self._phosphene_maps is None:
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
        """Which pixels fall inside each electrode's sampling region: the
        receptive-field centre radius, or the single nearest pixel."""
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
            'class_fraction': self.class_fraction,
        }

    def layer_activation_table(self) -> Dict[str, np.ndarray]:
        """Last frame's per-layer recruitment, keyed by layer name."""
        if self.layer_activation is None:
            raise RuntimeError("Call the simulator at least once first.")
        values = to_numpy(self.layer_activation.detach())
        return {name: values[..., i]
                for i, name in enumerate(self.layer_names)}

    def class_fraction_table(self) -> Dict[str, np.ndarray]:
        """Last frame's share of recruited cells, keyed by percept class."""
        if self.class_fraction is None:
            raise RuntimeError("Needs percept.model 'class_mixture' and at "
                               "least one call of the simulator.")
        values = to_numpy(self.class_fraction.detach())
        return {name: values[..., i] for i, name in enumerate(self.classes)}

    def describe(self) -> str:
        """The electrode array's configuration and per-electrode flags."""
        return self.array.describe()


def _widen(sigma: torch.Tensor, extra: Optional[torch.Tensor]
           ) -> torch.Tensor:
    """The width of a Gaussian of width `sigma` blurred by one of width
    `extra`; `sigma` itself when `extra` is None."""
    if extra is None:
        return sigma
    # `tiny` keeps the gradient finite where both widths are zero
    return torch.sqrt(sigma ** 2 + extra ** 2 + torch.finfo(sigma.dtype).tiny)
