"""How simultaneously active electrodes combine.

Each mode maps the stimulation currents to an equivalent current per
electrode: the current that electrode alone would need to recruit what
it recruits in company. The charge, threshold, brightness and size
models then run unchanged on that current.

`independent`
    Each electrode recruits as if alone.
`field_superposition`
    Each electrode's signed drive ``(R_j / d)^2`` is summed and only then
    passed through the recruitment kernel. Every electrode counts what
    is recruited in its own neighborhood, so neighbors can only raise
    its equivalent current. Its equivalent radius is its single-electrode
    radius times the cube root of recruited over single-electrode volume,
    capped at the radius for `electrodes.max_current_ua`.

Which electrodes' recruited volumes merge into one phosphene is decided
in `dynaphos_lgn.merging`.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.spatial import cKDTree

from dynaphos.utils import to_tensor
from dynaphos_lgn.electrodes import LGNElectrodeArray

MODES = ('field_superposition', 'independent')


def effective_mode(params: dict) -> str:
    """The mode that actually runs: `independent` whenever pulses are
    not simultaneous, since summation needs them to overlap in time."""
    mode = params['aggregation']['mode']
    if mode not in MODES:
        raise ValueError(f"Unknown aggregation.mode {mode!r}; expected one "
                         f"of {', '.join(MODES)}.")
    if not params['aggregation']['simultaneous_pulses']:
        return 'independent'
    return mode


@dataclass
class AggregationResult:
    """What one aggregation step hands back to the simulator.

    :ivar amplitude_ua: (..., n) equivalent current per electrode.
    :ivar per_layer_weight: (..., n, n_layers) recruitment in each
        electrode's neighborhood, or None to let the simulator compute
        it from `amplitude_ua`.
    :ivar per_layer_cells: Same, in cells.
    """

    amplitude_ua: torch.Tensor
    per_layer_weight: Optional[torch.Tensor] = None
    per_layer_cells: Optional[torch.Tensor] = None


def electrode_positions_mm(array: LGNElectrodeArray) -> np.ndarray:
    """(n, 3) electrode positions in the atlas frame, mm."""
    return array.voxel_indices * array.atlas.voxel_size_mm


def drive_radius(kernel, amplitude_ua: torch.Tensor) -> torch.Tensor:
    """Recruitment radius of |I|, which scales an electrode's drive
    ``(R / d)^2``."""
    return kernel.radius_mm(torch.abs(amplitude_ua))


class Independent:
    """Each electrode recruits as if it were the only one."""

    def __call__(self, amplitude_ua: torch.Tensor) -> AggregationResult:
        return AggregationResult(amplitude_ua)

    def describe(self) -> str:
        return 'independent'


class FieldSuperposition:
    """Field summation from each electrode's distance histogram.

    Electrode i's tissue is binned in shells of radius d. A neighbor j
    at distance D contributes its drive averaged over that shell: on a
    shell, the distance r to j has density ``r / (2 D d)`` on
    ``[|D - d|, D + d]``, so the mean of ``1 / r^2`` over the part within
    j's gather radius Rg is ``ln(min(D + d, Rg) / |D - d|) / (2 D d)``.
    The kernel is applied to that mean drive, which ignores where in the
    shell the tissue lies.
    """

    def __init__(self, array: LGNElectrodeArray, params: dict,
                 data_kwargs: dict, histograms: dict):
        self.kernel = array.kernel
        self.gather_radius_mm = array.max_radius_mm
        self.max_current_ua = array.max_current_ua
        self.max_radius_mm = array.nominal_radius_mm
        self.warned_cap = False
        self.min_distance_mm = 0.5 * array.atlas.voxel_size_mm
        self.merging = params['aggregation']['phosphene_merging']
        self.n_electrodes = array.n_electrodes

        # Neighbors: pairs whose gathered fields can overlap at all.
        self.positions_mm = electrode_positions_mm(array)
        pairs = cKDTree(self.positions_mm).query_pairs(
            2.0 * self.gather_radius_mm, output_type='ndarray')
        self.pairs = pairs.reshape(-1, 2)

        d = np.maximum(histograms['bin_centers_mm'], self.min_distance_mm)
        self.bin_distance = to_tensor(d, **data_kwargs)
        self.counts = to_tensor(histograms['counts'], **data_kwargs)
        self.cells = to_tensor(histograms['cells'], **data_kwargs)

        both = np.concatenate([self.pairs, self.pairs[:, ::-1]])
        D = np.linalg.norm(self.positions_mm[both[:, 0]]
                           - self.positions_mm[both[:, 1]], axis=-1)[:, None]
        reach = self.gather_radius_mm
        near = np.maximum(np.abs(D - d[None]), self.min_distance_mm)
        far = np.minimum(D + d[None], reach)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_inverse_square = np.where(
                D > self.min_distance_mm,
                np.log(far / near) / (2.0 * D * d[None]),
                1.0 / d[None] ** 2)
        shell = np.where(near < reach, mean_inverse_square, 0.0)
        long = dict(device=data_kwargs.get('device'), dtype=torch.long)
        self.target = torch.as_tensor(both[:, 0], **long)
        self.source = torch.as_tensor(both[:, 1], **long)
        self.shell = to_tensor(shell, **data_kwargs)       # (pairs, bins)

    def describe(self) -> str:
        return f'field_superposition ({self.merging} phosphenes)'

    def __call__(self, amplitude_ua: torch.Tensor) -> AggregationResult:
        radius = drive_radius(self.kernel, amplitude_ua)
        sign = torch.sign(amplitude_ua)
        weight, cells, single = self._recruited(radius, sign)
        recruited = weight.sum(dim=-1)
        has_volume = single > 0
        ratio = torch.where(
            has_volume, recruited / torch.where(has_volume, single,
                                                torch.ones_like(single)),
            torch.ones_like(single))
        factor = torch.clamp(ratio, min=1e-12) ** (1.0 / 3.0)
        factor = self._capped(factor, radius)
        magnitude = self.kernel.scale_current(torch.abs(amplitude_ua), factor)
        return AggregationResult(sign * magnitude, weight, cells)

    def _recruited(self, radius, sign):
        """Per-layer recruitment in each stimulated electrode's
        neighborhood under the summed field, and the volume each would
        recruit alone."""
        own = torch.clamp(sign[..., None]
                          * (radius[..., None] / self.bin_distance) ** 2,
                          min=0.0)
        neighbor = (sign[..., self.source, None]
                     * radius[..., self.source, None] ** 2 * self.shell)
        drive = own + torch.zeros_like(own).index_add(-2, self.target,
                                                      neighbor)

        stimulated = (own > 0).to(own.dtype)
        recruited = self.kernel.weights_from_drive(drive) * stimulated
        weight = torch.einsum('...nb,nlb->...nl', recruited, self.counts)
        cells = torch.einsum('...nb,nlb->...nl', recruited, self.cells)
        single = torch.einsum('...nb,nlb->...n',
                              self.kernel.weights_from_drive(own),
                              self.counts)
        return weight, cells, single

    def _capped(self, factor, radius):
        """Cap the equivalent radius at `max_radius_mm`, whose kernel
        tail just fits the gathered neighborhood. An electrode's own
        current is never lowered."""
        on = radius > 0
        cap = torch.clamp(self.max_radius_mm / torch.where(
            on, radius, torch.ones_like(radius)), min=1.0)
        if not self.warned_cap and bool((on & (factor > cap)).any()):
            self.warned_cap = True
            logging.warning(
                "field_superposition: equivalent current capped at "
                "electrodes.max_current_ua (%g uA).", self.max_current_ua)
        return torch.where(on, torch.minimum(factor, cap), factor)


def build_aggregation(array: LGNElectrodeArray, params: dict,
                      data_kwargs: dict, histograms: dict):
    """The aggregation step `aggregation.*` describes."""
    if effective_mode(params) == 'independent':
        return Independent()
    return FieldSuperposition(array, params, data_kwargs, histograms)
