"""Retinotopic magnification of the macaque LGN.

Magnification is in degrees of visual field per millimeter of tissue,
which turns a current-spread radius into a phosphene size. It comes from
the per-voxel Jacobian of the atlas's retinotopy (`JacobianMagnification`).
The Jacobian is fitted in the flat chart (E cos I, E sin I) the renderer
draws in: radial distances are exact there, tangential ones are
stretched by E / sin E (2% at 20 deg, 9% at 40 deg).

Also here: the cell-density and volume fits of Malpeli, Lee & Baker
(1996).
"""
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def malpeli_parvo_density(eccentricity_deg: ArrayLike,
                          params: Mapping) -> np.ndarray:
    """Parvocellular density, Eq. 3: ``a * (E + e0) ** exponent``."""
    c = params['magnification']['malpeli']['parvo']
    e = np.asarray(eccentricity_deg, dtype=float)
    return c['a'] * (e + c['e0']) ** c['exponent']


def malpeli_magno_density(eccentricity_deg: ArrayLike,
                          params: Mapping) -> np.ndarray:
    """Magnocellular density, Eq. 4: ``a * ((E - e0)^2 + c) ** exponent``.

    Non-monotonic near the fovea.
    """
    k = params['magnification']['malpeli']['magno']
    e = np.asarray(eccentricity_deg, dtype=float)
    return k['a'] * ((e - k['e0']) ** 2 + k['c']) ** k['exponent']


def malpeli_volume_magnification(eccentricity_deg: ArrayLike,
                                 params: Mapping) -> np.ndarray:
    """Laminar volume per deg^2 of hemifield, Eq. 2, mm^3/deg^2."""
    c = params['magnification']['malpeli']['volume']
    e = np.asarray(eccentricity_deg, dtype=float)
    return c['a'] * (e + c['e0']) ** c['exponent']


def malpeli_density(eccentricity_deg: ArrayLike, params: Mapping,
                    cell_class: str = 'parvo') -> np.ndarray:
    if cell_class == 'parvo':
        return malpeli_parvo_density(eccentricity_deg, params)
    if cell_class == 'magno':
        return malpeli_magno_density(eccentricity_deg, params)
    if cell_class == 'both':
        return (malpeli_parvo_density(eccentricity_deg, params)
                + malpeli_magno_density(eccentricity_deg, params))
    raise ValueError(f"Unknown cell_class {cell_class!r}; expected "
                     f"'parvo', 'magno' or 'both'.")


def isotropic_construction_radius_deg(params: Mapping) -> float:
    """Radius within which the atlas's eccentricity comes from a
    rotationally symmetric formula, so local anisotropy is treated as
    isotropic."""
    return float(params['atlas']['isotropic_construction_radius_deg'])


@dataclass
class LocalMagnification:
    """Local tissue -> visual-field scaling, in degrees per millimeter.

    :ivar major_deg_per_mm: Larger principal magnification.
    :ivar minor_deg_per_mm: Smaller principal magnification.
    :ivar orientation_rad: Direction of the major axis in the visual
        field. Zero for isotropic models.
    :ivar isotropic_by_construction: Inside the atlas's isotropic center,
        or from an isotropic model.
    :ivar unreliable_beyond_neighborhood: The local linear map should not
        be extrapolated across a whole current-spread footprint; these
        electrodes are rendered by scatter instead of as an ellipse.
    :ivar valid: A magnification could be estimated at all.
    """

    major_deg_per_mm: np.ndarray
    minor_deg_per_mm: np.ndarray
    orientation_rad: np.ndarray
    isotropic_by_construction: np.ndarray
    unreliable_beyond_neighborhood: np.ndarray
    valid: np.ndarray

    @property
    def anisotropy(self) -> np.ndarray:
        """Ratio of the two principal magnifications, >= 1."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(self.minor_deg_per_mm > 0,
                            self.major_deg_per_mm / self.minor_deg_per_mm,
                            np.nan)

    @property
    def areal_deg2_per_mm2(self) -> np.ndarray:
        """Product of the principal magnifications, deg^2 / mm^2."""
        return self.major_deg_per_mm * self.minor_deg_per_mm

    def __len__(self):
        return int(np.size(self.major_deg_per_mm))


class JacobianMagnification:
    """Per-voxel anisotropic magnification, from the cached Jacobian atlas.

    The Jacobian maps a physical displacement (mm, ML/DV/AP) to a
    visual-field displacement (deg, x/y). Its two singular values are the
    principal magnifications; the leading left singular vector gives the
    major axis's direction.

    :param jacobian_atlas: A computed or loaded `JacobianAtlas`.
    :param params: Parameter dictionary.
    :param atlas: Built `ErwinAtlas`, needed for `at_eccentricity`.
    """

    def __init__(self, jacobian_atlas, params: Mapping, atlas=None):
        self.jacobian_atlas = jacobian_atlas
        self.params = params
        self.atlas = atlas
        self._table_cache = {}

    @staticmethod
    def decompose(jacobian: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SVD of a stack of (..., 2, 3) Jacobians.

        :return: (major, minor, orientation_rad), NaN where the input is
            not finite.
        """
        j = np.asarray(jacobian, dtype=float)
        finite = np.isfinite(j).all(axis=(-2, -1))
        safe = np.where(finite[..., None, None], j, 0.0)
        u, s, _ = np.linalg.svd(safe, full_matrices=False)
        major = np.where(finite, s[..., 0], np.nan)
        minor = np.where(finite, s[..., 1], np.nan)
        orientation = np.where(finite,
                               np.arctan2(u[..., 1, 0], u[..., 0, 0]),
                               np.nan)
        return major, minor, orientation

    def at_voxels(self, voxel_indices: np.ndarray) -> LocalMagnification:
        """Magnification at the given (n, 3) atlas indices."""
        idx = tuple(np.asarray(voxel_indices, dtype=int).T)
        jacobian_atlas = self.jacobian_atlas
        major, minor, orientation = self.decompose(
            jacobian_atlas.jacobian_at(voxel_indices))
        return LocalMagnification(
            major_deg_per_mm=major,
            minor_deg_per_mm=minor,
            orientation_rad=orientation,
            isotropic_by_construction=jacobian_atlas.isotropic_flag[idx],
            unreliable_beyond_neighborhood=(
                jacobian_atlas.unreliable_beyond_neighborhood_flag[idx]),
            valid=jacobian_atlas.jacobian_valid[idx])

    def at_eccentricity(self, eccentricity_deg: ArrayLike,
                        cell_class: Optional[str] = 'parvo'
                        ) -> LocalMagnification:
        """Inclination-averaged magnification of a cell class (None: all
        tissue), binned at
        `magnification.eccentricity_table_bin_width_deg`."""
        centers, major_t, minor_t, orient_t = self.eccentricity_table(
            cell_class)
        ecc_query = np.atleast_1d(np.asarray(eccentricity_deg, dtype=float))
        # Interpolate the doubled angle, so the axis never wraps.
        sin2 = np.interp(ecc_query, centers, np.sin(2 * orient_t))
        cos2 = np.interp(ecc_query, centers, np.cos(2 * orient_t))
        return LocalMagnification(
            major_deg_per_mm=np.interp(ecc_query, centers, major_t),
            minor_deg_per_mm=np.interp(ecc_query, centers, minor_t),
            orientation_rad=0.5 * np.arctan2(sin2, cos2),
            isotropic_by_construction=(
                ecc_query < isotropic_construction_radius_deg(self.params)),
            unreliable_beyond_neighborhood=np.zeros_like(ecc_query, bool),
            valid=np.ones_like(ecc_query, bool))

    def _table_voxels(self, cell_class: Optional[str]):
        """Indices and SVD of every fit-ok voxel of `cell_class`."""
        atlas, jacobian_atlas = self.atlas, self.jacobian_atlas
        if atlas is None:
            raise ValueError("Tabulating needs the ErwinAtlas; pass "
                             "atlas= to the JacobianMagnification.")
        idx = np.argwhere(atlas.eligible(cell_class)
                          & jacobian_atlas.jacobian_valid)
        major, minor, orientation = self.decompose(
            jacobian_atlas.jacobian_at(idx))
        return tuple(idx.T), major, minor, orientation

    def eccentricity_table(self, cell_class: Optional[str]):
        """(centers, major, minor, orientation): per-bin medians over
        eccentricity (orientation: doubled-angle mean), cached per cell
        class."""
        if cell_class in self._table_cache:
            return self._table_cache[cell_class]

        bin_width_deg = self.params['magnification'][
            'eccentricity_table_bin_width_deg']
        sel, major, minor, orientation = self._table_voxels(cell_class)
        ecc = self.atlas.eccentricity_deg[sel]

        edges = np.arange(0, np.nanmax(ecc) + bin_width_deg, bin_width_deg)
        centers = 0.5 * (edges[:-1] + edges[1:])
        which = np.digitize(ecc, edges) - 1

        def binned(values, reduce=np.median):
            out = np.full(len(centers), np.nan)
            for b in range(len(centers)):
                sel = values[which == b]
                sel = sel[np.isfinite(sel)]
                if sel.size:
                    out[b] = reduce(sel)
            return out

        major_t = binned(major)
        minor_t = binned(minor)
        # Orientation is an axis (period pi): average the doubled angle.
        orient_t = 0.5 * np.arctan2(binned(np.sin(2 * orientation), np.mean),
                                    binned(np.cos(2 * orientation), np.mean))
        keep = np.isfinite(major_t)
        table = (centers[keep], major_t[keep], minor_t[keep], orient_t[keep])
        self._table_cache[cell_class] = table
        return table
