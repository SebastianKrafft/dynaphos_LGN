"""Receptive-field geometry for LGN cell classes.

Centre-surround difference-of-Gaussians sizes as a function of
eccentricity and cell class. The simulator uses these for stimulus
sampling -- which patch of image an electrode "sees" -- not for
phosphene appearance.

Verification is unusually uneven here: the M/P coefficients are an
in-project fit to digitised Croner & Kaplan (1995) points, and the
koniocellular defaults are owl monkey, good for the K > M > P ordering
and not for absolute sizes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Union

import numpy as np

from dynaphos_lgn.params import require

ArrayLike = Union[float, np.ndarray]


def cell_classes(params: Mapping) -> Sequence[str]:
    """The cell classes this configuration defines parameters for."""
    return list(require(params, 'receptive_fields.cell_classes'))


@dataclass
class ReceptiveField:
    """Difference-of-Gaussians receptive-field parameters.

    :ivar center_radius_deg: rc, the 1/e radius of the centre Gaussian
        -- Croner & Kaplan's convention, NOT a standard deviation
        (rc = sqrt(2) * sigma).
    :ivar surround_radius_deg: rs, same convention.
    :ivar surround_center_volume_ratio: The INTEGRATED surround/centre
        ratio ``(Ks*rs^2)/(Kc*rc^2)``, as the literature reports it. The
        peak ratio the DoG equation needs is derived from it in
        `surround_center_peak_ratio`.
    :ivar cell_class: Which class these describe.
    """

    center_radius_deg: np.ndarray
    surround_radius_deg: np.ndarray
    surround_center_volume_ratio: float
    cell_class: str

    @property
    def surround_center_peak_ratio(self) -> np.ndarray:
        """Ks/Kc, derived from the integrated ratio and the two radii.

        volume_ratio = (Ks * rs^2) / (Kc * rc^2)
          =>  Ks/Kc = volume_ratio * (rc/rs)^2
        """
        return (self.surround_center_volume_ratio
                / np.asarray(self.surround_center_ratio, dtype=float) ** 2)

    @property
    def center_sigma_deg(self) -> np.ndarray:
        """The centre Gaussian's standard deviation, rc / sqrt(2).

        Kept distinct from rc: collapsing the two is a factor of 1.41 in
        every derived size.
        """
        return np.asarray(self.center_radius_deg) / np.sqrt(2.0)

    @property
    def surround_center_ratio(self) -> np.ndarray:
        return (np.asarray(self.surround_radius_deg)
                / np.asarray(self.center_radius_deg))

    def profile(self, r_deg: ArrayLike) -> np.ndarray:
        """DoG sensitivity at radial distance `r_deg` from the centre.

        The centre is normalised to peak at 1, so the net DoG peaks
        slightly below 1 and crosses zero near the centre/surround
        boundary. That is the shape, not an error.
        """
        r = np.asarray(r_deg, dtype=float)
        rc = np.asarray(self.center_radius_deg, dtype=float)
        rs = np.asarray(self.surround_radius_deg, dtype=float)
        return (np.exp(-(r / rc) ** 2)
                - self.surround_center_peak_ratio
                * np.exp(-(r / rs) ** 2))


def center_radius_deg(eccentricity_deg: ArrayLike, params: Mapping,
                      cell_class: str = 'parvo',
                      form: str = 'exponential') -> np.ndarray:
    """Receptive-field centre radius rc(E), degrees.

    :param eccentricity_deg: Eccentricity, degrees.
    :param params: Parameter dictionary.
    :param cell_class: 'magno', 'parvo' or 'konio'. Defaults to 'parvo'.
    :param form: 'exponential' or 'power', for the M/P classes only. The
        koniocellular fit is linear and ignores this. Defaults to
        'exponential'.
    """
    e = np.asarray(eccentricity_deg, dtype=float)
    if cell_class == 'konio':
        k = require(params, 'receptive_fields.xu_konio')
        rc = k['rc_slope_per_deg'] * e + k['rc_intercept_deg']
        # The linear fit goes negative below ~4.9 deg; clamp to the same
        # paper's measured mean below 15 deg, not to zero.
        return np.maximum(rc, k['rc_below_15deg'][0])
    if cell_class not in ('magno', 'parvo'):
        raise ValueError(f"Unknown cell_class {cell_class!r}; expected one "
                         f"of {tuple(cell_classes(params))}.")
    if form == 'exponential':
        c = require(
            params,
            f'receptive_fields.croner_kaplan.exponential.{cell_class}')
        return c['a_deg'] * np.exp(c['b_per_deg'] * e)
    if form == 'power':
        c = require(params,
                    f'receptive_fields.croner_kaplan.power.{cell_class}')
        return c['a_deg'] * np.maximum(e, 1e-6) ** c['b']
    raise ValueError(f"Unknown form {form!r}; expected 'exponential' or "
                     f"'power'.")


def surround_radius_deg(eccentricity_deg: ArrayLike, params: Mapping,
                        cell_class: str = 'parvo',
                        form: str = 'exponential') -> np.ndarray:
    """Receptive-field surround radius rs(E), degrees.

    rs/rc for the M and P classes is
    ``receptive_fields.croner_kaplan.surround_center_radius_ratio``.
    Applying it as a scalar multiplier forces rs to share rc's
    eccentricity dependence, which the data does not establish -- a
    flagged assumption. The koniocellular
    class ignores it and uses Xu et al.'s own rs(E).
    """
    if cell_class == 'konio':
        k = require(params, 'receptive_fields.xu_konio')
        e = np.asarray(eccentricity_deg, dtype=float)
        rs = k['rs_slope_per_deg'] * e + k['rs_intercept_deg']
        return np.maximum(rs, k['rs_below_15deg'][0])
    surround_center_ratio = require(
        params,
        'receptive_fields.croner_kaplan.surround_center_radius_ratio')
    return surround_center_ratio * center_radius_deg(
        eccentricity_deg, params, cell_class, form)


def receptive_field(eccentricity_deg: ArrayLike, params: Mapping,
                    cell_class: str = 'parvo',
                    form: str = 'exponential') -> ReceptiveField:
    """Full DoG parameters at the given eccentricity."""
    return ReceptiveField(
        center_radius_deg=center_radius_deg(eccentricity_deg, params,
                                            cell_class, form),
        surround_radius_deg=surround_radius_deg(
            eccentricity_deg, params, cell_class, form),
        surround_center_volume_ratio=require(
            params,
            f'receptive_fields.surround_center_volume_ratio.{cell_class}'),
        cell_class=cell_class)
