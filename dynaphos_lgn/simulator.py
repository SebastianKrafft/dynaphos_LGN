"""Differentiable phosphene simulator for LGN stimulation.

Same interface as Dynaphos's V1 simulator, whose temporal dynamics,
thresholding and brightness classes it reuses. What is LGN-specific is
spatial: recruitment over atlas voxels, grouped by layer, and phosphene
geometry from the local magnification. Gradients reach amplitude, pulse
width and frequency, not electrode positions.
"""
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from dynaphos.image_processing import scale_image
from dynaphos.simulator import (Activation, ActivationThreshold, Brightness,
                                State, Trace)
from dynaphos.utils import (get_data_kwargs, get_deg2pix_coeff, print_stats,
                            set_deterministic, to_numpy, to_tensor)
from dynaphos_lgn.aggregation import (Independent, build_aggregation,
                                      effective_mode)
from dynaphos_lgn.current_spread import RecruitmentKernel
from dynaphos_lgn.electrodes import LGNElectrodeArray
from dynaphos_lgn.merging import build_merging
from dynaphos_lgn.receptive_fields import center_radius_by_class
from dynaphos_lgn.rendering import (EllipseGaussianRenderer, ScatterRenderer,
                                    visual_field_grid)


class LGNSize(State):
    """Phosphene sigmas in degrees, from current via the local
    magnification. The LGN counterpart of Dynaphos's `Sigma`."""

    def __init__(self, params: dict, shape: Tuple[int, ...],
                 kernel: RecruitmentKernel,
                 major_deg_per_mm: np.ndarray, minor_deg_per_mm: np.ndarray,
                 verbose: bool = False):
        self.kernel = kernel
        major = np.asarray(major_deg_per_mm, dtype=float).ravel()
        minor = np.asarray(minor_deg_per_mm, dtype=float).ravel()

        # No usable magnification: render at zero size, and say so.
        self.no_magnification = ~(np.isfinite(major) & np.isfinite(minor))
        if self.no_magnification.any():
            logging.warning(
                "%d of %d electrodes have no finite local magnification "
                "and render as zero-size phosphenes: %s.",
                int(self.no_magnification.sum()), self.no_magnification.size,
                np.array2string(np.flatnonzero(self.no_magnification),
                                threshold=20))
        super().__init__(params, shape, verbose)
        self.major = self.to_tensor(np.reshape(
            np.where(self.no_magnification, 0.0, major), shape[-3:]))
        self.minor = self.to_tensor(np.reshape(
            np.where(self.no_magnification, 0.0, minor), shape[-3:]))

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

    :param params: Nested parameter dict, see `dynaphos_lgn/config/params_lgn.yaml`.
    :param electrode_array: A built `LGNElectrodeArray`.
    :param rng: Numpy generator, used for threshold initialization.

    `safety_limit_ua` (from `default_stim.safety_limit_ua`) is for
    whatever chooses the amplitudes; the forward pass does not clamp.
    """

    def __init__(self, params: dict, electrode_array: LGNElectrodeArray,
                 rng: Optional[np.random.Generator] = None):
        self.params = params
        self.array = electrode_array
        self.safety_limit_ua = float(
            params['default_stim']['safety_limit_ua'])
        if not 0 < self.safety_limit_ua <= electrode_array.max_current_ua:
            raise ValueError(
                f"default_stim.safety_limit_ua ({self.safety_limit_ua} uA) "
                f"must be positive and at most electrodes.max_current_ua "
                f"({electrode_array.max_current_ua} uA).")
        self.data_kwargs = get_data_kwargs(params)
        self.verbose = params['run']['print_stats']
        self.adaptive_scatter_bandwidth = params['rendering'][
            'adaptive_scatter_bandwidth']

        rng = np.random.default_rng() if rng is None else rng
        # Deterministic mode refuses cuBLAS matmuls on CUDA without this
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        set_deterministic(params['run']['seed'])
        self.deg2pix_coeff = get_deg2pix_coeff(params['run'])

        self.kernel = electrode_array.kernel
        self.num_phosphenes = electrode_array.n_electrodes

        batch_size = params['run']['batch_size']
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
            params, self.shape, self.kernel,
            electrode_array.local_magnification.major_deg_per_mm,
            electrode_array.local_magnification.minor_deg_per_mm,
            self.verbose)
        self.no_magnification = self.size.no_magnification

        self.effective_charge_per_second = None
        self.layer_activation = None
        self.recruited_cells = None
        self._volumes = None
        self._last_amplitude_ua = None

        self._sampling_mask = None
        self._phosphene_maps = None
        self._phosphene_centers = None
        self._sampling_method = params['sampling']['sampling_method']
        self._pulse_width = (params['default_stim']['pw_default']
                             * torch.ones(self.shape, **self.data_kwargs))
        self._frequency = (params['default_stim']['freq_default']
                           * torch.ones(self.shape, **self.data_kwargs))
        self._zero = to_tensor(0, **self.data_kwargs)

        thresholding = params['thresholding']
        self.soft_threshold = bool(thresholding['soft_threshold'])
        # Thresholds already vary across electrodes with the calibrated
        # sd, so the logistic stays narrow (10 / sd) to keep the
        # population curve at the calibrated spread.
        self._soft_threshold_slope = thresholding['soft_threshold_slope']
        if self._soft_threshold_slope is None:
            self._soft_threshold_slope = 10.0 / max(
                thresholding['activation_threshold_sd'], 1e-30)
        self.soft_threshold_baseline = thresholding['soft_threshold_baseline']
        self.reset()
    # Setup
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
        half = self.params['run']['view_angle'] / 2
        origin = np.asarray(self.params['run']['origin'], dtype=float)
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
            idx = self.scatter_index
            patch_radius = array.scatter_patch_radius_deg()[idx]
            ceiling_fraction = self.params['rendering'][
                'scatter_bandwidth_ceiling_fraction']
            self.scatter_renderer = ScatterRenderer(
                array.scatter_points_xy[idx], array.vf_xy[idx], self.params,
                patch_radius, array.scatter_bandwidth_floor_deg[idx],
                ceiling_fraction * patch_radius, self.data_kwargs)

    def _build_recruitment_buffers(self):
        array = self.array
        hist = array.distance_histograms()
        self.layer_names = hist['layer_names']
        self._bin_centers = to_tensor(hist['bin_centers_mm'],
                                      **self.data_kwargs)
        self._layer_counts = to_tensor(hist['counts'], **self.data_kwargs)
        self._layer_cells = to_tensor(hist['cells'], **self.data_kwargs)

        self.aggregation_configured = self.params['aggregation']['mode']
        self.aggregation_effective = effective_mode(self.params)
        self.aggregation = build_aggregation(array, self.params,
                                             self.data_kwargs, hist)
        if self.aggregation_effective != self.aggregation_configured:
            logging.warning(
                "aggregation.mode is %r but aggregation.simultaneous_pulses "
                "is false, so electrodes aggregate independently.",
                self.aggregation_configured)
        logging.info("Aggregation: configured %s, effective %s.",
                     self.aggregation_configured,
                     self.aggregation.describe())
        self.merging = build_merging(array, self.params, self.data_kwargs)

        if len(self.scatter_index):
            self._scatter_distance = to_tensor(
                array.scatter_distance_mm, **self.data_kwargs)
            self._scatter_mask = to_tensor(
                array.scatter_valid[self.scatter_index].astype(float),
                **self.data_kwargs)
            self._scatter_points = to_tensor(
                array.scatter_points_xy[self.scatter_index],
                **self.data_kwargs)
    # Runtime
    def reset(self):
        self.activation.reset()
        self.trace.reset()
        self.size.reset()

    def to_tensor(self, x) -> torch.Tensor:
        return to_tensor(x, **self.data_kwargs)

    def get_current(self, amplitude: torch.Tensor, frequency: torch.Tensor,
                    pulse_width: torch.Tensor) -> torch.Tensor:
        """Effective charge per second from a square-wave pulse train,
        as in Dynaphos's V1 model, but with a Lapicque strength-duration
        threshold: shorter pulses need more current. A chronaxie of 0
        gives the V1 model's constant rheobase.

        :param amplitude: Stimulation amplitudes for each electrode.
        :param pulse_width: Stimulation pulse widths for each electrode.
        :param frequency: Stimulation frequencies for each electrode.
        """
        rheobase = self.params['thresholding']['rheobase']
        chronaxie = self.params['default_stim']['chronaxie_s']
        # 1 ns floor keeps the threshold finite at zero pulse width
        safe_pulse_width = torch.clamp(pulse_width, min=1e-9)
        leak = self.trace.get() + rheobase * (1 + chronaxie / safe_pulse_width)
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
        unit_to_ua = self.params['run']['current_unit_to_ua']
        aggregate = None
        stimulated_ua = (amplitude * unit_to_ua).reshape(self.shape[:-2])
        self._volumes = (None if self.merging is None
                         else self.merging.volumes(stimulated_ua))
        if not isinstance(self.aggregation, Independent):
            # Everything downstream sees the equivalent current.
            aggregate = self.aggregation(stimulated_ua)
            amplitude = (aggregate.amplitude_ua / unit_to_ua).reshape(
                self.shape)

        charge_per_s = self.get_current(amplitude,
                                        self._as_input(frequency),
                                        self._as_input(pulse_width))
        self.activation.update(charge_per_s)
        self.trace.update(charge_per_s)

        amplitude_ua = amplitude * unit_to_ua
        self.size.update(amplitude_ua)
        if aggregate is not None and aggregate.per_layer_weight is not None:
            self.layer_activation = aggregate.per_layer_weight
            self.recruited_cells = aggregate.per_layer_cells
        else:
            self.layer_activation, self.recruited_cells = \
                self.recruitment(amplitude_ua)
        self.brightness.update(self.activation.get())
        self._last_amplitude_ua = amplitude_ua

    def spatial_activation(self) -> torch.Tensor:
        """Per-electrode activation maps over the visual field, peak 1,
        in the original electrode order."""
        res_x, res_y = self.params['run']['resolution']
        lead = self.shape[:-3]
        out = torch.zeros((*lead, self.num_phosphenes, res_y, res_x),
                          **self.data_kwargs)
        dim = self._electrode_dimension

        if self.ellipse_renderer is not None:
            idx = torch.as_tensor(self.ellipse_index,
                                  device=out.device, dtype=torch.long)
            sa = self.size.sigma_major.index_select(dim, idx)
            sb = self.size.sigma_minor.index_select(dim, idx)
            out = out.index_copy(dim, idx,
                                 self.ellipse_renderer.render(sa, sb))

        if self.scatter_renderer is not None:
            idx = torch.as_tensor(self.scatter_index,
                                  device=out.device, dtype=torch.long)
            amp = self._last_amplitude_ua.index_select(dim, idx).reshape(
                *lead, len(self.scatter_index), 1)
            w = self.kernel.weights(self._scatter_distance, amp)
            w = w * self._scatter_mask
            h = self.scatter_renderer.bandwidth_deg(
                w, self._scatter_points,
                adaptive=self.adaptive_scatter_bandwidth)
            out = out.index_copy(dim, idx,
                                 self.scatter_renderer.render(w, h))

        return out

    def __call__(self, amplitude: torch.Tensor,
                 pulse_width: Optional[torch.Tensor] = None,
                 frequency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Render one frame of simulated percept, (..., H, W)."""
        self.update(amplitude, pulse_width, frequency)
        return self.finish(self.unclamped_percept())

    def unclamped_percept(self) -> torch.Tensor:
        """The current percept, summed over electrodes but not clamped,
        so several nuclei can be added before `finish`."""
        intensity = self.brightness.get() * self.detection_probability()
        maps = self.spatial_activation()
        lead, n, image = maps.shape[:-3], maps.shape[-3], maps.shape[-2:]
        intensity = intensity.reshape(*lead, n)
        if self._volumes is None:
            return torch.sum(intensity[..., None, None] * maps,
                             dim=self._electrode_dimension)
        # One phosphene per recruited volume.
        frames = self.merging.percept(
            intensity.reshape(-1, n), maps.reshape(-1, n, *image),
            self._volumes)
        return frames.reshape(*lead, *frames.shape[1:])

    def finish(self, unclamped: torch.Tensor) -> torch.Tensor:
        """The displayed percept from an unclamped one."""
        return unclamped.clamp(0, 1)

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
    # Stimulus sampling
    @property
    def phosphene_maps(self) -> torch.Tensor:
        """Distance from every pixel to each phosphene center, degrees."""
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
    def rf_cell_class(self) -> np.ndarray:
        """Cell class whose receptive field each electrode samples: the
        class of the layer at its tip, or the one `sampling.rf_cell_class`
        names."""
        sampling = self.params['sampling']
        if sampling['rf_cell_class'] != 'tip':
            return np.full(self.array.n_electrodes, sampling['rf_cell_class'])
        tip = self.array.tip_cell_class
        return np.where(tip == '', sampling['rf_fallback_class'], tip)

    @property
    def sampling_mask(self) -> torch.Tensor:
        """Which pixels fall inside each electrode's sampling region: the
        receptive-field center radius, or the single nearest pixel."""
        if self._sampling_mask is None:
            if self._sampling_method == 'receptive_fields':
                radius = center_radius_by_class(
                    self.array.eccentricity_deg, self.params,
                    self.rf_cell_class, self.params['sampling']['rf_form'])
                radius = radius * self.params['sampling']['rf_size_scale']
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
            out = out * self.params['sampling']['stimulus_scale']
        return out
    # Reporting
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Tensors of the last frame, plus the configured and effective
        aggregation modes as strings."""
        return {
            'aggregation_configured': self.aggregation_configured,
            'aggregation_effective': self.aggregation.describe(),
            'equivalent_current_ua': self._last_amplitude_ua,
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
        """The electrode array's configuration, per-electrode flags and
        aggregation mode."""
        return '\n'.join([
            self.array.describe(),
            f"  aggregation       : configured {self.aggregation_configured}"
            f", effective {self.aggregation.describe()}"])

