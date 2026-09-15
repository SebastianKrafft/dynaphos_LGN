"""Current spread and tissue recruitment for LGN microstimulation.

How far the current reaches (Stoney/Tehovnik, ``R = sqrt(I / K)``) and
how strongly each nearby voxel is recruited, from one shared kernel so
the two can never drift apart. The kernel is smooth by necessity: a hard
in/out cutoff has zero gradient with respect to current, which would
break end-to-end optimisation at the first step.

``K`` is transplanted from macaque V1 and is unvalidated for LGN.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple, Union

import numpy as np
import torch

from dynaphos_lgn.params import require

Tensorish = Union[float, np.ndarray, torch.Tensor]


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def excitability_constant_ua_per_mm2(params: Mapping) -> float:
    """K in uA/mm^2 -- transplanted from macaque V1, unvalidated for LGN."""
    return float(require(params, 'current_spread.k_ua_per_mm2'))


def lgn_threshold_ua(params: Mapping) -> Tuple[float, float]:
    """(mean, sd) single-electrode LGN detection threshold, uA."""
    mean, sd = require(params, 'benchmarks.pezaris_2007.threshold_ua')
    return float(mean), float(sd)


def stoney_radius_mm(current_ua: Tensorish,
                     k_ua_per_mm2: float) -> Tensorish:
    """Activated-tissue **radius** R = sqrt(I/K), in mm.

    Half the literature's diameter. Conflating the two is a 2x size
    error, hence the unambiguous name.
    """
    if isinstance(current_ua, torch.Tensor):
        return torch.sqrt(torch.clamp(current_ua, min=0.0) / k_ua_per_mm2)
    return np.sqrt(np.clip(np.asarray(current_ua, float), 0, None)
                   / k_ua_per_mm2)


def bosking_diameter_mm(current_ua: Tensorish, max_diameter_mm: float,
                        slope_per_ua: float,
                        i_half_ua: float) -> Tensorish:
    """Saturating alternative to Stoney's square root.

    Bosking et al. (2017)'s functional shape only; the configured slope
    and half-current are Dynaphos's V1 values, not Bosking's.
    """
    if isinstance(current_ua, torch.Tensor):
        return max_diameter_mm * torch.sigmoid(
            slope_per_ua * (current_ua - i_half_ua))
    return max_diameter_mm / (1.0 + np.exp(
        -slope_per_ua * (np.asarray(current_ua, float) - i_half_ua)))


def k_for_cell_class(cell_class: str, params: Mapping) -> float:
    """Optionally scale K by cell-class soma size.

    `current_spread.class_scaling` picks between 'none' (the default),
    'diameter' (K ~ 1/d) and 'diameter_squared' (K ~ 1/d^2). No
    measurement supports any particular exponent, which is why all three
    stay explicit.
    """
    k_reference = excitability_constant_ua_per_mm2(params)
    reference_class = require(params,
                              'current_spread.class_scaling_reference')
    scaling = require(params, 'current_spread.class_scaling')
    if scaling == 'none':
        return k_reference
    diameters = 'current_spread.nucleolus_diameter_um'
    d = require(params, f'{diameters}.{cell_class}')
    d_ref = require(params, f'{diameters}.{reference_class}')
    if scaling == 'diameter':
        return k_reference * (d_ref / d)
    if scaling == 'diameter_squared':
        return k_reference * (d_ref / d) ** 2
    raise ValueError(f"Unknown class scaling {scaling!r}; expected "
                     f"'none', 'diameter' or 'diameter_squared'.")


# ----------------------------------------------------------------------
# Recruitment kernels
# ----------------------------------------------------------------------

@dataclass
class RecruitmentKernel:
    """Smooth, differentiable tissue-recruitment weight w(distance, I).

    The kernel is anchored so that its weight equals `contour_value` at
    exactly the Stoney radius R(I). That anchoring is what keeps
    "the radius the size model uses" and "the radius the per-layer
    recruitment uses" the same number rather than two independently
    tuned ones.

    :ivar shape: 'gaussian' or 'logistic'.
    :ivar contour_value: Weight at r = R(I). 1/e^2 by default, which
        makes `radius_to_sigma` equal V1 Dynaphos's shipped 0.5.
    :ivar logistic_softness: Transition width of the logistic kernel, as
        a fraction of R(I). Smaller is closer to a hard cutoff, and so
        closer to a vanishing gradient.
    :ivar k: Excitability constant, given in uA/mm^2.
    :ivar spread_model: 'stoney' (sqrt) or 'bosking' (sigmoid).
    """

    shape: str
    contour_value: float
    logistic_softness: float
    k: float
    spread_model: str
    bosking_max_diameter_mm: float
    bosking_slope_per_ua: float
    bosking_i_half_ua: float

    @classmethod
    def from_params(cls, params: Mapping) -> 'RecruitmentKernel':
        """Build the kernel the config describes.

        When `electrodes.cell_class` names a class, K goes through
        `k_for_cell_class`, so the configured class scaling applies.
        """
        cell_class = require(params, 'electrodes.cell_class')
        bosking = require(params, 'current_spread.bosking')
        return cls(
            shape=require(params, 'current_spread.kernel_shape'),
            contour_value=require(params, 'current_spread.contour_value'),
            logistic_softness=require(params,
                                      'current_spread.logistic_softness'),
            k=(excitability_constant_ua_per_mm2(params) if cell_class is None
               else k_for_cell_class(cell_class, params)),
            spread_model=require(params, 'current_spread.spread_model'),
            bosking_max_diameter_mm=bosking['max_diameter_mm'],
            bosking_slope_per_ua=bosking['slope_per_ua'],
            bosking_i_half_ua=bosking['i_half_ua'])

    @property
    def _logistic_offset(self) -> float:
        return _logit(self.contour_value)

    def __post_init__(self):
        if self.shape not in ('gaussian', 'logistic'):
            raise ValueError("shape must be 'gaussian' or 'logistic'.")
        if not 0.0 < self.contour_value < 1.0:
            raise ValueError("contour_value must lie strictly in (0, 1).")
        if self.spread_model not in ('stoney', 'bosking'):
            raise ValueError("spread_model must be 'stoney' or 'bosking'.")

    # -- radii --------------------------------------------------------
    def radius_mm(self, current_ua: Tensorish) -> Tensorish:
        """Nominal activated-tissue radius at this current."""
        if self.spread_model == 'stoney':
            return stoney_radius_mm(current_ua, self.k)
        return 0.5 * bosking_diameter_mm(
            current_ua, self.bosking_max_diameter_mm,
            self.bosking_slope_per_ua, self.bosking_i_half_ua)

    @property
    def radius_to_sigma(self) -> float:
        """sigma / R such that the Gaussian equals `contour_value` at R.

        exp(-0.5 (R/sigma)^2) = c  =>  sigma = R / sqrt(-2 ln c).
        """
        return 1.0 / math.sqrt(-2.0 * math.log(self.contour_value))

    def sigma_mm(self, current_ua: Tensorish) -> Tensorish:
        """Gaussian sigma of the recruitment profile, mm."""
        return self.radius_mm(current_ua) * self.radius_to_sigma

    # -- weights ------------------------------------------------------
    def weights(self, distance_mm: torch.Tensor,
                current_ua: torch.Tensor) -> torch.Tensor:
        """Recruitment weight in [0, 1], differentiable in `current_ua`.

        :param distance_mm: Distances from the electrode, mm. Any shape
            broadcastable against `current_ua`.
        :param current_ua: Stimulation current per electrode, uA.
        """
        radius = self.radius_mm(current_ua)
        eps = torch.finfo(distance_mm.dtype).tiny
        if self.shape == 'gaussian':
            sigma = torch.clamp(radius * self.radius_to_sigma, min=eps)
            return torch.exp(-0.5 * (distance_mm / sigma) ** 2)
        softness = torch.clamp(radius * self.logistic_softness, min=eps)
        # Anchored so w(R) == contour_value exactly: at d = R the
        # argument reduces to logit(contour_value).
        offset = self._logistic_offset
        return torch.sigmoid(-(distance_mm - radius) / softness + offset)

    def effective_radius_mm(self, current_ua: Tensorish,
                            tail_weight: float = 1e-3) -> Tensorish:
        """Radius beyond which the kernel's weight drops under
        `tail_weight`.

        Setup code sizing a candidate neighbourhood must use this, not
        `radius_mm`, or it clips the smooth kernel's tail.
        """
        if self.shape == 'gaussian':
            factor = math.sqrt(-2.0 * math.log(tail_weight)) \
                     * self.radius_to_sigma
        else:
            # sigmoid(-(d - R)/s + offset) = tail
            #   =>  d = R + s * (offset - logit(tail))
            factor = 1.0 + self.logistic_softness * (
                self._logistic_offset - _logit(tail_weight))
        return self.radius_mm(current_ua) * factor


def aggregate_activation(per_electrode_weights: torch.Tensor,
                         mode: str = 'independent') -> torch.Tensor:
    """Combine simultaneously-active electrodes' recruitment weights.

    :param per_electrode_weights: (..., n_electrodes, n_voxels) weights.
    :param mode: 'independent' (default, linear summation),
        'field_superposition' (same sum, documented as additive fields)
        or 'probability_summation' (``1 - prod(1 - w)``).

    There is no LGN multi-electrode data: the modes bracket a real
    disagreement in the non-LGN literature rather than approximating one
    known answer, so report across modes, not from one.
    """
    if mode in ('independent', 'field_superposition'):
        return per_electrode_weights.sum(dim=-2)
    if mode == 'probability_summation':
        return 1.0 - torch.prod(1.0 - per_electrode_weights.clamp(0, 1),
                                dim=-2)
    raise ValueError(f"Unknown aggregation mode {mode!r}.")


def threshold_reduction_bracket(n_electrodes: int, spacing_mm: float,
                                params: Mapping) -> dict:
    """Bracket the expected multi-electrode threshold reduction.

    The two mechanistic extremes from the non-LGN literature, plus the
    geometric quantity that decides which should dominate: electrode
    spacing relative to the current-spread radius. This is a bracket,
    not a prediction.

    :return: dict with ``spread_radius_mm``, ``spacing_over_radius``,
        ``floor_fraction`` (Callier-style probability summation),
        ``ceiling_fraction`` (Kunigk-style field summation),
        ``regime`` and ``note``.
    """
    current_ua = lgn_threshold_ua(params)[0]
    k_ua_per_mm2 = excitability_constant_ua_per_mm2(params)
    cfg = require(params, 'benchmarks.multi_electrode')

    radius = float(stoney_radius_mm(current_ua, k_ua_per_mm2))
    ratio = spacing_mm / radius if radius > 0 else float('inf')

    saturates_at = max(float(cfg['saturates_at_n_electrodes']), 1.0)

    # log2-scaled ramp from 0 (at 1 electrode) to 1 (at n_electrodes >= saturates_at)
    saturation_fraction = min(1.0, math.log2(max(n_electrodes, 1))
                              / max(math.log2(saturates_at), 1e-9))

    # Kim & Callier 2015: spacing-independent and small;
    floor = cfg['probability_summation_max_reduction'] * saturation_fraction
    # Kunigk 2022: large, and vanishing once the electrodes' fields stop overlapping.
    vanish = float(cfg['field_summation_vanishes_at_radii'])
    overlap = max(0.0, 1.0 - ratio / vanish)
    ceiling = (cfg['field_summation_max_reduction'] * saturation_fraction
              * overlap)

    if ratio <= 1.0:
        regime = 'field-summation dominated (spacing <= spread radius)'
    elif ratio <= vanish:
        regime = 'mixed'
    else:
        regime = (f'probability-summation dominated '
                  f'(spacing > {vanish:g}x radius)')

    return {
        'n_electrodes': n_electrodes,
        'spacing_mm': spacing_mm,
        'spread_radius_mm': radius,
        'spacing_over_radius': ratio,
        'floor_fraction': floor,
        'ceiling_fraction': max(floor, ceiling),
        'regime': regime,
        'note': ('Bracket, not a prediction: no LGN multi-electrode '
                 'threshold data exists. Floor from Callier et al. (2015), '
                 'ceiling from Kunigk et al. (2022), both non-LGN.'),
    }
