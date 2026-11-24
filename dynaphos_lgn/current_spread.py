"""Current spread and tissue recruitment for LGN microstimulation.

How far the current reaches (Stoney/Tehovnik, ``R = sqrt(I / K)``) and
how strongly each nearby voxel is recruited, from one Gaussian kernel.
The kernel is smooth so that recruitment stays differentiable in the
current.
"""
import math
from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import numpy as np
import torch

Tensorish = Union[float, np.ndarray, torch.Tensor]


def lgn_threshold_ua(params: dict) -> Tuple[float, float]:
    """(mean, sd) single-electrode LGN detection threshold, uA."""
    mean, sd = params['benchmarks']['pezaris_2007']['threshold_ua']
    return float(mean), float(sd)


def stoney_radius_mm(current_ua: Tensorish,
                     k_ua_per_mm2: float) -> Tensorish:
    """Activated-tissue radius R = sqrt(I/K), in mm (not the diameter).

    Zero at I <= 0, with a zero gradient there rather than sqrt's inf.
    """
    if isinstance(current_ua, torch.Tensor):
        # Double where: the sqrt never sees 0, so 0 * inf cannot give NaN.
        on = current_ua > 0
        safe = torch.where(on, current_ua, torch.ones_like(current_ua))
        return torch.where(on, torch.sqrt(safe / k_ua_per_mm2),
                           torch.zeros_like(current_ua))
    return np.sqrt(np.clip(np.asarray(current_ua, float), 0, None)
                   / k_ua_per_mm2)


def excitability_constant_ua_per_mm2(params: dict) -> float:
    """K in uA/mm^2."""
    return float(params['current_spread']['k_ua_per_mm2'])


def k_for_cell_class(cell_class: Union[str, Sequence[str]],
                     params: dict) -> float:
    """K, optionally scaled by the axon diameter of the cell class.

    `current_spread.class_scaling` is 'none', 'diameter' (K ~ 1/d) or
    'diameter_squared' (K ~ 1/d^2). A list of classes uses the mean of
    their diameters.
    """
    p = params['current_spread']
    k_reference = excitability_constant_ua_per_mm2(params)
    scaling = p['class_scaling']
    if scaling == 'none':
        return k_reference
    diameters = p['axon_diameter_um']
    names = [cell_class] if isinstance(cell_class, str) else list(cell_class)
    d = float(np.mean([diameters[name] for name in names]))
    d_ref = diameters[p['class_scaling_reference']]
    if scaling == 'diameter':
        return k_reference * (d_ref / d)
    if scaling == 'diameter_squared':
        return k_reference * (d_ref / d) ** 2
    raise ValueError(f"Unknown class scaling {scaling!r}; expected "
                     f"'none', 'diameter' or 'diameter_squared'.")


@dataclass
class RecruitmentKernel:
    """Gaussian tissue-recruitment weight w(distance, I).

    The weight equals `contour_value` at exactly the spread radius R(I),
    so phosphene size and per-layer recruitment use the same radius.

    :ivar contour_value: Weight at r = R(I). 1/e^2 makes `radius_to_sigma`
        equal V1 Dynaphos's 0.5.
    :ivar k: Excitability constant, uA/mm^2.
    """

    contour_value: float
    k: float

    @classmethod
    def from_params(cls, params: dict) -> 'RecruitmentKernel':
        """The kernel the config describes, with K scaled for
        `electrodes.cell_class` when one is set."""
        cell_class = params['electrodes']['cell_class']
        return cls(
            contour_value=params['current_spread']['contour_value'],
            k=(excitability_constant_ua_per_mm2(params) if cell_class is None
               else k_for_cell_class(cell_class, params)))

    def __post_init__(self):
        if not 0.0 < self.contour_value < 1.0:
            raise ValueError("contour_value must lie strictly in (0, 1).")

    def radius_mm(self, current_ua: Tensorish) -> Tensorish:
        """Nominal activated-tissue radius at this current."""
        return stoney_radius_mm(current_ua, self.k)

    @property
    def radius_to_sigma(self) -> float:
        """sigma / R such that the Gaussian equals `contour_value` at R.

        exp(-0.5 (R/sigma)^2) = c  =>  sigma = R / sqrt(-2 ln c).
        """
        return 1.0 / math.sqrt(-2.0 * math.log(self.contour_value))

    def sigma_mm(self, current_ua: Tensorish) -> Tensorish:
        """Gaussian sigma of the recruitment profile, mm."""
        return self.radius_mm(current_ua) * self.radius_to_sigma

    def weights(self, distance_mm: torch.Tensor,
                current_ua: torch.Tensor) -> torch.Tensor:
        """Recruitment weight in [0, 1], differentiable in `current_ua`.

        :param distance_mm: Distances from the electrode, mm. Any shape
            broadcastable against `current_ua`.
        :param current_ua: Stimulation current per electrode, uA.
        """
        radius = self.radius_mm(current_ua)
        eps = torch.finfo(distance_mm.dtype).tiny
        sigma = torch.clamp(radius * self.radius_to_sigma, min=eps)
        return torch.exp(-0.5 * (distance_mm / sigma) ** 2)

    def effective_radius_mm(self, current_ua: Tensorish,
                            tail_weight: float = 1e-3) -> Tensorish:
        """Radius beyond which the weight drops under `tail_weight`.

        Use this, not `radius_mm`, to size a candidate neighborhood, or
        the kernel's tail gets clipped.
        """
        factor = math.sqrt(-2.0 * math.log(tail_weight)) \
                 * self.radius_to_sigma
        return self.radius_mm(current_ua) * factor

    def weights_from_drive(self, drive: torch.Tensor) -> torch.Tensor:
        """Recruitment weight from a summed, threshold-normalized drive.

        One electrode's drive at distance d is ``(R / d)^2 = I / (K d^2)``,
        a point source's current density over the recruitment threshold,
        so ``drive = 1`` is the R(I) contour and for a single electrode
        this equals ``weights(d, I)``. Drives from several electrodes add.
        """
        drive = torch.clamp(drive, min=1e-12)
        return torch.exp(-0.5 / (self.radius_to_sigma ** 2 * drive))

    def scale_current(self, current_ua: torch.Tensor,
                      radius_factor: torch.Tensor) -> torch.Tensor:
        """The current whose radius is ``radius_factor * R(current)``."""
        return current_ua * radius_factor ** 2
