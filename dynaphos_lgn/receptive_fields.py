"""Receptive-field geometry for LGN cell classes.

Difference-of-Gaussians center and surround sizes by eccentricity, for
magno and parvo cells (Croner & Kaplan 1995).
"""
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def cell_classes(params: dict) -> Sequence[str]:
    """The cell classes this configuration defines parameters for."""
    return list(params['receptive_fields']['cell_classes'])


@dataclass
class ReceptiveField:
    """Difference-of-Gaussians receptive-field parameters.

    :ivar center_radius_deg: rc, the 1/e radius of the center Gaussian
        (rc = sqrt(2) * sigma).
    :ivar surround_radius_deg: rs, same convention.
    :ivar surround_center_volume_ratio: The integrated ratio
        ``(Ks*rs^2)/(Kc*rc^2)``; see `surround_center_peak_ratio`.
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
    def surround_center_ratio(self) -> np.ndarray:
        return (np.asarray(self.surround_radius_deg)
                / np.asarray(self.center_radius_deg))

    def profile(self, r_deg: ArrayLike) -> np.ndarray:
        """DoG sensitivity at radial distance `r_deg`, center peak 1."""
        r = np.asarray(r_deg, dtype=float)
        rc = np.asarray(self.center_radius_deg, dtype=float)
        rs = np.asarray(self.surround_radius_deg, dtype=float)
        return (np.exp(-(r / rc) ** 2)
                - self.surround_center_peak_ratio
                * np.exp(-(r / rs) ** 2))


def center_radius_deg(eccentricity_deg: ArrayLike, params: dict,
                      cell_class: str = 'parvo',
                      form: str = 'exponential') -> np.ndarray:
    """Receptive-field center radius rc(E), degrees.

    :param eccentricity_deg: Eccentricity, degrees.
    :param params: Parameter dictionary.
    :param cell_class: 'magno' or 'parvo'.
    :param form: 'exponential' or 'power'.
    """
    e = np.asarray(eccentricity_deg, dtype=float)
    if cell_class not in ('magno', 'parvo'):
        raise ValueError(f"Unknown cell_class {cell_class!r}; expected one "
                         f"of {tuple(cell_classes(params))}.")
    fits = params['receptive_fields']['croner_kaplan']
    if form == 'exponential':
        c = fits['exponential'][cell_class]
        return c['a_deg'] * np.exp(c['b_per_deg'] * e)
    if form == 'power':
        c = fits['power'][cell_class]
        return c['a_deg'] * np.maximum(e, 1e-6) ** c['b']
    raise ValueError(f"Unknown form {form!r}; expected 'exponential' or "
                     f"'power'.")


def center_radius_by_class(eccentricity_deg: ArrayLike, params: dict,
                           cell_class: ArrayLike,
                           form: str = 'exponential') -> np.ndarray:
    """rc(E), degrees, with a cell class per element of `eccentricity_deg`."""
    e = np.asarray(eccentricity_deg, dtype=float)
    cell_class = np.broadcast_to(np.asarray(cell_class), e.shape)
    radius = np.empty(e.shape)
    for name in np.unique(cell_class):
        here = cell_class == name
        radius[here] = center_radius_deg(e[here], params, str(name), form)
    return radius


def surround_radius_deg(eccentricity_deg: ArrayLike, params: dict,
                        cell_class: str = 'parvo',
                        form: str = 'exponential') -> np.ndarray:
    """Receptive-field surround radius rs(E), degrees: rc(E) times the
    class's `surround_center_radius_ratio`."""
    ratio = params['receptive_fields']['croner_kaplan'][
        'surround_center_radius_ratio'][cell_class]
    return ratio * center_radius_deg(eccentricity_deg, params, cell_class,
                                     form)


def receptive_field(eccentricity_deg: ArrayLike, params: dict,
                    cell_class: str = 'parvo',
                    form: str = 'exponential') -> ReceptiveField:
    """Full DoG parameters at the given eccentricity."""
    return ReceptiveField(
        center_radius_deg=center_radius_deg(eccentricity_deg, params,
                                            cell_class, form),
        surround_radius_deg=surround_radius_deg(
            eccentricity_deg, params, cell_class, form),
        surround_center_volume_ratio=params['receptive_fields'][
            'surround_center_volume_ratio'][cell_class],
        cell_class=cell_class)
