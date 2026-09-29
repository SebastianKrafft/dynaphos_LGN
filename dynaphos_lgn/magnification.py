"""Retinotopic magnification models for the macaque LGN.

Every model returns degrees of visual field per millimetre of tissue,
which turns a current-spread radius into a phosphene size.

`JacobianMagnification`
    Per-voxel and anisotropic, from the cached `JacobianAtlas`.
`AtlasGradientMagnification`
    Isotropic M(E), tabulated by eccentricity.
`AnisotropicGradientMagnification`
    The Jacobian's principal magnifications tabulated over
    (eccentricity, inclination).
`MalpeliDensityMagnification`
    Malpeli et al. (1996)'s cell density converted to a linear
    magnification, as a (low, central, high) interval.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import numpy as np

from dynaphos_lgn.atlas import mirror_inclination, mirror_orientation
from dynaphos_lgn.params import optional, require

ArrayLike = Union[float, np.ndarray]


# ----------------------------------------------------------------------
# Malpeli, Lee & Baker (1996) cell density, cells/deg^2
# ----------------------------------------------------------------------

def malpeli_parvo_density(eccentricity_deg: ArrayLike,
                          params: Mapping) -> np.ndarray:
    """Parvocellular density, Eq. 1: ``a * (E + e0) ** exponent``."""
    c = require(params, 'magnification.malpeli.parvo')
    e = np.asarray(eccentricity_deg, dtype=float)
    return c['a'] * (e + c['e0']) ** c['exponent']


def malpeli_magno_density(eccentricity_deg: ArrayLike,
                          params: Mapping) -> np.ndarray:
    """Magnocellular density, Eq. 2: ``a * ((E - e0)^2 + c) ** exponent``.

    Non-monotonic near the fovea.
    """
    k = require(params, 'magnification.malpeli.magno')
    e = np.asarray(eccentricity_deg, dtype=float)
    return k['a'] * ((e - k['e0']) ** 2 + k['c']) ** k['exponent']


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
    """Radius within which the atlas's retinotopy is isotropic by
    construction."""
    return float(require(params, 'atlas.isotropic_construction_radius_deg'))


# ----------------------------------------------------------------------
# The common return type
# ----------------------------------------------------------------------

@dataclass
class LocalMagnification:
    """Local tissue -> visual-field scaling, in degrees per millimetre.

    :ivar major_deg_per_mm: Larger principal magnification.
    :ivar minor_deg_per_mm: Smaller principal magnification.
    :ivar orientation_rad: Direction of the major axis in the visual
        field. Zero for isotropic models.
    :ivar isotropic_by_construction: Inside the atlas's isotropic centre,
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

    @classmethod
    def isotropic(cls, deg_per_mm: np.ndarray) -> 'LocalMagnification':
        """A circular magnification, the same along both axes."""
        return cls(
            major_deg_per_mm=deg_per_mm,
            minor_deg_per_mm=deg_per_mm,
            orientation_rad=np.zeros_like(deg_per_mm),
            isotropic_by_construction=np.ones_like(deg_per_mm, bool),
            unreliable_beyond_neighborhood=np.zeros_like(deg_per_mm, bool),
            valid=np.isfinite(deg_per_mm))

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

    @property
    def linear_mm_per_deg(self) -> np.ndarray:
        """Isotropic-equivalent linear magnification, 1/sqrt(areal)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return 1.0 / np.sqrt(self.areal_deg2_per_mm2)

    def __len__(self):
        return int(np.size(self.major_deg_per_mm))


class MagnificationModel:
    """Interface: turn a location into a `LocalMagnification`."""

    def at_eccentricity(self, eccentricity_deg: ArrayLike,
                        cell_class: str = 'parvo',
                        inclination_deg: Optional[ArrayLike] = None
                        ) -> LocalMagnification:
        """:param inclination_deg: Needed by models that depend on more
            than eccentricity; isotropic models ignore it."""
        raise NotImplementedError

    def at_voxels(self, voxel_indices: np.ndarray, erwin_atlas=None
                  ) -> LocalMagnification:
        """Magnification at (n, 3) atlas indices, looked up by each
        voxel's (eccentricity, inclination) in `erwin_atlas`."""
        if erwin_atlas is None:
            raise ValueError(f"{type(self).__name__} needs the ErwinAtlas "
                             f"to turn voxel indices into visual-field "
                             f"positions.")
        idx = tuple(np.asarray(voxel_indices, dtype=int).T)
        return self.at_eccentricity(
            erwin_atlas.eccentricity_deg[idx],
            inclination_deg=erwin_atlas.inclination_deg[idx])


def _table_voxels(jacobian_magnification: 'JacobianMagnification',
                  cell_class: Optional[str]):
    """Indices and SVD of every fit-ok voxel of `cell_class`, for the
    table-based models."""
    ea = jacobian_magnification.erwin_atlas
    ja = jacobian_magnification.jacobian_atlas
    if ea is None:
        raise ValueError("Tabulating needs the ErwinAtlas -- pass "
                         "erwin_atlas= to the JacobianMagnification.")
    idx = np.argwhere(ea.eligible(cell_class) & ja.jacobian_valid)
    major, minor, orientation = JacobianMagnification.decompose(
        ja.jacobian_at(idx))
    return tuple(idx.T), major, minor, orientation


class JacobianMagnification(MagnificationModel):
    """Per-voxel anisotropic magnification, from the cached Jacobian atlas.

    The Jacobian maps a physical displacement (mm, ML/DV/AP) to a
    visual-field displacement (deg, x/y). Its two singular values are the
    principal magnifications; the leading left singular vector gives the
    major axis's direction.

    :param jacobian_atlas: A computed or loaded `JacobianAtlas`.
    :param params: Parameter dictionary.
    :param erwin_atlas: Built `ErwinAtlas`, needed for `at_eccentricity`
        and for building the table models.
    """

    def __init__(self, jacobian_atlas, params: Mapping, erwin_atlas=None):
        self.jacobian_atlas = jacobian_atlas
        self.params = params
        self.erwin_atlas = erwin_atlas
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

    def at_voxels(self, voxel_indices: np.ndarray, erwin_atlas=None
                  ) -> LocalMagnification:
        """Magnification at the given (n, 3) atlas indices."""
        idx = tuple(np.asarray(voxel_indices, dtype=int).T)
        ja = self.jacobian_atlas
        major, minor, orientation = self.decompose(
            ja.jacobian_at(voxel_indices))
        return LocalMagnification(
            major_deg_per_mm=major,
            minor_deg_per_mm=minor,
            orientation_rad=orientation,
            isotropic_by_construction=ja.isotropic_flag[idx],
            unreliable_beyond_neighborhood=(
                ja.unreliable_beyond_neighborhood_flag[idx]),
            valid=ja.jacobian_valid[idx])

    def at_eccentricity(self, eccentricity_deg: ArrayLike,
                        cell_class: str = 'parvo',
                        inclination_deg: Optional[ArrayLike] = None
                        ) -> LocalMagnification:
        """Inclination-averaged magnification, binned at
        `magnification.eccentricity_table_bin_width_deg`. For comparison
        with literature curves; `inclination_deg` is ignored."""
        centers, major_t, minor_t, orient_t = self.eccentricity_table(
            cell_class)
        ecc_query = np.atleast_1d(np.asarray(eccentricity_deg, dtype=float))
        return LocalMagnification(
            major_deg_per_mm=np.interp(ecc_query, centers, major_t),
            minor_deg_per_mm=np.interp(ecc_query, centers, minor_t),
            orientation_rad=np.interp(ecc_query, centers, orient_t),
            isotropic_by_construction=(
                ecc_query < isotropic_construction_radius_deg(self.params)),
            unreliable_beyond_neighborhood=np.zeros_like(ecc_query, bool),
            valid=np.ones_like(ecc_query, bool))

    def eccentricity_table(self, cell_class: Optional[str]):
        """(centers, major, minor, orientation): per-bin medians over
        eccentricity, cached per cell class."""
        if cell_class in self._table_cache:
            return self._table_cache[cell_class]

        bin_width_deg = require(
            self.params, 'magnification.eccentricity_table_bin_width_deg')
        sel, major, minor, orientation = _table_voxels(self, cell_class)
        ecc = self.erwin_atlas.eccentricity_deg[sel]

        edges = np.arange(0, np.nanmax(ecc) + bin_width_deg, bin_width_deg)
        centers = 0.5 * (edges[:-1] + edges[1:])
        which = np.digitize(ecc, edges) - 1

        def binned_median(values):
            out = np.full(len(centers), np.nan)
            for b in range(len(centers)):
                sel = values[which == b]
                sel = sel[np.isfinite(sel)]
                if sel.size:
                    out[b] = np.median(sel)
            return out

        major_t = binned_median(major)
        minor_t = binned_median(minor)
        orient_t = binned_median(orientation)
        keep = np.isfinite(major_t)
        table = (centers[keep], major_t[keep], minor_t[keep], orient_t[keep])
        self._table_cache[cell_class] = table
        return table


class AtlasGradientMagnification(MagnificationModel):
    """Isotropic scalar magnification M(E), tabulated in mm/deg."""

    def __init__(self, bin_centers_deg: np.ndarray,
                 mm_per_deg: np.ndarray, params: Optional[Mapping] = None):
        self.params = params
        order = np.argsort(np.asarray(bin_centers_deg, dtype=float))
        self.bin_centers_deg = np.asarray(bin_centers_deg, float)[order]
        self.mm_per_deg = np.asarray(mm_per_deg, float)[order]
        finite = np.isfinite(self.mm_per_deg) & (self.mm_per_deg > 0)
        if not finite.any():
            raise ValueError("No usable (positive, finite) bins.")
        self.bin_centers_deg = self.bin_centers_deg[finite]
        self.mm_per_deg = self.mm_per_deg[finite]

    @classmethod
    def from_jacobian(cls, jacobian_magnification: JacobianMagnification,
                      cell_class: Optional[str]
                      ) -> 'AtlasGradientMagnification':
        centers, major, minor, _ = \
            jacobian_magnification.eccentricity_table(cell_class)
        with np.errstate(divide='ignore', invalid='ignore'):
            mm_per_deg = 1.0 / np.sqrt(major * minor)
        return cls(centers, mm_per_deg)

    def at_eccentricity(self, eccentricity_deg: ArrayLike,
                        cell_class: str = 'parvo',
                        inclination_deg: Optional[ArrayLike] = None
                        ) -> LocalMagnification:
        e = np.atleast_1d(np.asarray(eccentricity_deg, dtype=float))
        mm_per_deg = np.interp(e, self.bin_centers_deg, self.mm_per_deg)
        return LocalMagnification.isotropic(1.0 / mm_per_deg)


def _bin_reliability(value_grid: np.ndarray, n_grid: np.ndarray,
                     min_voxels_per_bin: int, outlier_z_threshold: float
                     ) -> np.ndarray:
    """True for bins with enough voxels whose value is not a robust
    outlier against the rest of its eccentricity row.

    :param value_grid: (n_ecc, n_incl), NaN where a bin has no data.
    :param n_grid: Voxel count per bin, same shape.
    :param outlier_z_threshold: Robust z-score (value - row median) /
        (1.4826 * row MAD) above which a bin is flagged.
    """
    reliable = (n_grid >= min_voxels_per_bin) & np.isfinite(value_grid)
    for e in range(value_grid.shape[0]):
        row_ok = reliable[e]
        if row_ok.sum() < 3:
            continue  # too few points on this row to judge an outlier
        row_vals = value_grid[e, row_ok]
        median = np.median(row_vals)
        mad = np.median(np.abs(row_vals - median))
        if mad == 0:
            # Most of the row agrees exactly: anything else is an outlier.
            reliable[e] &= (value_grid[e] == median)
            continue
        z = np.abs(value_grid[e] - median) / (1.4826 * mad)
        reliable[e] &= (z <= outlier_z_threshold)
    return reliable


class AnisotropicGradientMagnification(MagnificationModel):
    """The Jacobian's major, minor and orientation tabulated over
    (eccentricity, inclination), so simulation never touches the
    per-voxel Jacobian.

    Unreliable bins (too few voxels, or outliers along their eccentricity
    row, see `_bin_reliability`) are left out of the interpolation.
    Orientation is stored as 0 in near-isotropic bins and interpolated
    via the doubled angle, so it never wraps.
    """

    def __init__(self, ecc_edges: np.ndarray, incl_edges: np.ndarray,
                 major_grid: np.ndarray, minor_grid: np.ndarray,
                 orientation_grid: np.ndarray, n_grid: np.ndarray,
                 reliable_grid: np.ndarray, isotropic_grid: np.ndarray,
                 orientation_reliable_min_anisotropy: float,
                 params: Optional[Mapping] = None):
        self.params = params
        self.ecc_edges = np.asarray(ecc_edges, dtype=float)
        self.incl_edges = np.asarray(incl_edges, dtype=float)
        self.ecc_centers = 0.5 * (self.ecc_edges[:-1] + self.ecc_edges[1:])
        self.incl_centers = 0.5 * (self.incl_edges[:-1] + self.incl_edges[1:])
        if (len(self.ecc_centers) < 2) or (len(self.incl_centers) < 2):
            raise ValueError("Need at least 2 bins on each axis to "
                             "interpolate; got "
                             f"{len(self.ecc_centers)} x "
                             f"{len(self.incl_centers)}.")

        self.n_grid = np.asarray(n_grid, dtype=float)
        self.reliable_grid = np.asarray(reliable_grid, dtype=bool)
        self.isotropic_grid = np.asarray(isotropic_grid, dtype=bool)
        self.orientation_reliable_min_anisotropy = float(
            orientation_reliable_min_anisotropy)

        # NaN out unreliable bins so interpolation routes around them.
        self.major_grid = np.where(self.reliable_grid, major_grid, np.nan)
        self.minor_grid = np.where(self.reliable_grid, minor_grid, np.nan)
        self.orientation_grid = np.asarray(orientation_grid, dtype=float)
        self._cos2 = np.where(self.reliable_grid,
                              np.cos(2 * self.orientation_grid), np.nan)
        self._sin2 = np.where(self.reliable_grid,
                              np.sin(2 * self.orientation_grid), np.nan)

        reliable_e, reliable_i = np.nonzero(self.reliable_grid)
        if reliable_e.size == 0:
            raise ValueError("No (eccentricity, inclination) bin passed "
                             "the reliability gate; the table is empty. "
                             "Loosen min_voxels_per_bin or "
                             "outlier_z_threshold.")
        self._reliable_ecc = self.ecc_centers[reliable_e]
        self._reliable_incl = self.incl_centers[reliable_i]
        self._reliable_e_idx = reliable_e
        self._reliable_i_idx = reliable_i

    @classmethod
    def from_jacobian(cls, jacobian_magnification: JacobianMagnification,
                      cell_class: Optional[str],
                      params: Optional[Mapping] = None
                      ) -> 'AnisotropicGradientMagnification':
        """Bin every fit-ok voxel's decomposed Jacobian over (E, I).

        Build it from the left nucleus and wrap it in
        `MirroredMagnification` for the right one, whose horizontal
        meridian sits on the +-180 deg edge of this table.

        :param jacobian_magnification: Must carry `erwin_atlas`.
        :param cell_class: 'parvo', 'magno', or None for every lamina.
        :param params: Defaults to `jacobian_magnification.params`.
        """
        params = (jacobian_magnification.params if params is None
                  else params)
        cfg = require(params, 'magnification.anisotropic_gradient')
        ecc_width = float(cfg['eccentricity_bin_width_deg'])
        incl_width = float(cfg['inclination_bin_width_deg'])
        min_voxels_per_bin = int(cfg['min_voxels_per_bin'])
        outlier_z_threshold = float(cfg['outlier_z_threshold'])
        aniso_gate = float(cfg['orientation_reliable_min_anisotropy'])

        sel, major, minor, orientation = _table_voxels(
            jacobian_magnification, cell_class)
        ea = jacobian_magnification.erwin_atlas
        ecc = ea.eccentricity_deg[sel]
        incl = ea.inclination_deg[sel]

        finite = np.isfinite(major) & np.isfinite(minor) & np.isfinite(
            orientation)
        ecc, incl, major, minor, orientation = (
            ecc[finite], incl[finite], major[finite], minor[finite],
            orientation[finite])
        if ecc.size == 0:
            raise ValueError(
                f"No finite fit-ok voxels for cell_class={cell_class!r}; "
                f"cannot build a table.")

        ecc_edges = np.arange(0.0, float(ecc.max()) + ecc_width, ecc_width)
        incl_lo = incl_width * np.floor(float(incl.min()) / incl_width)
        incl_hi = incl_width * np.ceil(float(incl.max()) / incl_width)
        incl_edges = np.arange(incl_lo, incl_hi + incl_width, incl_width)

        n_e, n_i = len(ecc_edges) - 1, len(incl_edges) - 1
        ecc_bin = np.clip(np.digitize(ecc, ecc_edges) - 1, 0, n_e - 1)
        incl_bin = np.clip(np.digitize(incl, incl_edges) - 1, 0, n_i - 1)
        flat_bin = ecc_bin * n_i + incl_bin
        n_bins = n_e * n_i

        n_grid = np.bincount(flat_bin, minlength=n_bins
                             ).reshape(n_e, n_i).astype(float)

        # Group by bin with one sort, then loop over populated bins only.
        order = np.argsort(flat_bin, kind='stable')
        sorted_bin = flat_bin[order]
        sorted_major = major[order]
        sorted_minor = minor[order]
        sorted_cos2 = np.cos(2 * orientation[order])
        sorted_sin2 = np.sin(2 * orientation[order])
        unique_bins, starts, counts = np.unique(
            sorted_bin, return_index=True, return_counts=True)

        major_flat = np.full(n_bins, np.nan)
        minor_flat = np.full(n_bins, np.nan)
        cos2_flat = np.full(n_bins, np.nan)
        sin2_flat = np.full(n_bins, np.nan)
        for b, start, count in zip(unique_bins, starts, counts):
            sl = slice(start, start + count)
            major_flat[b] = np.median(sorted_major[sl])
            minor_flat[b] = np.median(sorted_minor[sl])
            cos2_flat[b] = np.mean(sorted_cos2[sl])
            sin2_flat[b] = np.mean(sorted_sin2[sl])

        major_grid = major_flat.reshape(n_e, n_i)
        minor_grid = minor_flat.reshape(n_e, n_i)
        orientation_grid = 0.5 * np.arctan2(sin2_flat.reshape(n_e, n_i),
                                            cos2_flat.reshape(n_e, n_i))

        with np.errstate(divide='ignore', invalid='ignore'):
            anisotropy_grid = major_grid / minor_grid
        ecc_centers = 0.5 * (ecc_edges[:-1] + ecc_edges[1:])
        isotropic_radius = isotropic_construction_radius_deg(params)
        isotropic_grid = ((anisotropy_grid < aniso_gate)
                          | (ecc_centers[:, None] < isotropic_radius))
        orientation_grid = np.where(isotropic_grid, 0.0, orientation_grid)

        reliable_grid = (
            _bin_reliability(major_grid, n_grid, min_voxels_per_bin,
                             outlier_z_threshold)
            & _bin_reliability(minor_grid, n_grid, min_voxels_per_bin,
                               outlier_z_threshold))

        return cls(ecc_edges, incl_edges, major_grid, minor_grid,
                   orientation_grid, n_grid, reliable_grid, isotropic_grid,
                   aniso_gate, params=params)

    def _interpolate(self, ecc: np.ndarray, incl: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]:
        """Bilinear interpolation of major/minor/cos2/sin2 at (ecc, incl).

        Unreliable corners are dropped and the weights renormalised; a
        point with no reliable corner takes the nearest reliable bin.

        :return: (major, minor, cos2, sin2, unreliable), with `unreliable`
            True wherever a corner was dropped.
        """
        ecc = np.clip(ecc, self.ecc_centers[0], self.ecc_centers[-1])
        incl = np.clip(incl, self.incl_centers[0], self.incl_centers[-1])

        e_idx = np.clip(np.searchsorted(self.ecc_centers, ecc) - 1,
                        0, len(self.ecc_centers) - 2)
        i_idx = np.clip(np.searchsorted(self.incl_centers, incl) - 1,
                        0, len(self.incl_centers) - 2)
        e0, e1 = self.ecc_centers[e_idx], self.ecc_centers[e_idx + 1]
        i0, i1 = self.incl_centers[i_idx], self.incl_centers[i_idx + 1]
        we = np.where(e1 > e0, (ecc - e0) / (e1 - e0), 0.0)
        wi = np.where(i1 > i0, (incl - i0) / (i1 - i0), 0.0)

        corners = [(e_idx, i_idx, (1 - we) * (1 - wi)),
                   (e_idx, i_idx + 1, (1 - we) * wi),
                   (e_idx + 1, i_idx, we * (1 - wi)),
                   (e_idx + 1, i_idx + 1, we * wi)]

        major_num = np.zeros_like(ecc, dtype=float)
        minor_num = np.zeros_like(ecc, dtype=float)
        cos2_num = np.zeros_like(ecc, dtype=float)
        sin2_num = np.zeros_like(ecc, dtype=float)
        weight_sum = np.zeros_like(ecc, dtype=float)
        for ei, ii, w in corners:
            major_val = self.major_grid[ei, ii]
            usable = np.isfinite(major_val) & (w > 0)
            contribution = np.where(usable, w, 0.0)
            major_num += contribution * np.where(usable, major_val, 0.0)
            minor_num += contribution * np.where(
                usable, self.minor_grid[ei, ii], 0.0)
            cos2_num += contribution * np.where(
                usable, self._cos2[ei, ii], 0.0)
            sin2_num += contribution * np.where(
                usable, self._sin2[ei, ii], 0.0)
            weight_sum += contribution

        clean = weight_sum > (1.0 - 1e-6)  # all 4 corners reliable
        no_data = weight_sum <= 1e-9       # zero reliable corners
        with np.errstate(divide='ignore', invalid='ignore'):
            major = np.where(no_data, np.nan, major_num / weight_sum)
            minor = np.where(no_data, np.nan, minor_num / weight_sum)
            cos2 = np.where(no_data, np.nan, cos2_num / weight_sum)
            sin2 = np.where(no_data, np.nan, sin2_num / weight_sum)

        if np.any(no_data):
            qe, qi = ecc[no_data], incl[no_data]
            d2 = ((qe[:, None] - self._reliable_ecc[None, :]) ** 2
                  + (qi[:, None] - self._reliable_incl[None, :]) ** 2)
            nearest = np.argmin(d2, axis=1)
            re = self._reliable_e_idx[nearest]
            ri = self._reliable_i_idx[nearest]
            major[no_data] = self.major_grid[re, ri]
            minor[no_data] = self.minor_grid[re, ri]
            cos2[no_data] = self._cos2[re, ri]
            sin2[no_data] = self._sin2[re, ri]

        return major, minor, cos2, sin2, ~clean

    def at_eccentricity(self, eccentricity_deg: ArrayLike,
                        cell_class: str = 'parvo',
                        inclination_deg: Optional[ArrayLike] = None
                        ) -> LocalMagnification:
        if inclination_deg is None:
            raise ValueError(
                "AnisotropicGradientMagnification needs inclination_deg. "
                "Use AtlasGradientMagnification where only eccentricity "
                "is available.")
        ecc, incl = np.broadcast_arrays(
            np.atleast_1d(np.asarray(eccentricity_deg, dtype=float)),
            np.atleast_1d(np.asarray(inclination_deg, dtype=float)))
        ecc, incl = ecc.ravel(), incl.ravel()

        major, minor, cos2, sin2, unreliable = self._interpolate(ecc, incl)
        orientation = 0.5 * np.arctan2(sin2, cos2)

        isotropic_radius = (isotropic_construction_radius_deg(self.params)
                            if self.params is not None else 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            anisotropy = major / minor
        isotropic = ((ecc < isotropic_radius)
                     | ~(anisotropy >= self.orientation_reliable_min_anisotropy))
        orientation = np.where(isotropic, 0.0, orientation)
        valid = np.isfinite(major) & np.isfinite(minor)

        return LocalMagnification(
            major_deg_per_mm=major,
            minor_deg_per_mm=minor,
            orientation_rad=orientation,
            isotropic_by_construction=isotropic,
            unreliable_beyond_neighborhood=unreliable,
            valid=valid)


class MirroredMagnification(MagnificationModel):
    """A model of the left LGN, answering for its mirror image.

    Inclination is reflected into the left nucleus's frame before the
    lookup and the orientation reflected back afterwards; magnitudes are
    unchanged by a reflection.

    :param base: Any `MagnificationModel` built from the left nucleus.
    """

    def __init__(self, base: MagnificationModel):
        self.base = base
        self.params = getattr(base, 'params', None)

    def at_eccentricity(self, eccentricity_deg: ArrayLike,
                        cell_class: Optional[str] = None,
                        inclination_deg: Optional[ArrayLike] = None
                        ) -> LocalMagnification:
        # Forward cell_class only when given: the wrapped models have
        # different defaults.
        kwargs = {} if cell_class is None else {'cell_class': cell_class}
        if inclination_deg is not None:
            kwargs['inclination_deg'] = mirror_inclination(
                np.asarray(inclination_deg, dtype=float))
        local = self.base.at_eccentricity(eccentricity_deg, **kwargs)
        return dataclasses.replace(
            local, orientation_rad=mirror_orientation(local.orientation_rad))


class MalpeliDensityMagnification(MagnificationModel):
    """Linear magnification from Malpeli et al. (1996)'s cell density.

    Density (cells/deg^2) divided by a volumetric cell density rho_v
    (cells/mm^3) and a projection-column thickness T (mm) gives an areal
    magnification; its square root is the linear one. T is configured as
    a (low, central, high) triple, so `interval` returns three values and
    `at_eccentricity` the central one.

    :param cells_per_mm3: rho_v, as a scalar or a callable of
        eccentricity. Defaults to the config value; build with
        `from_atlas` when that is null.
    """

    def __init__(self, params: Mapping, cell_class: str = 'parvo',
                 cells_per_mm3=None):
        if cell_class not in ('parvo', 'magno'):
            raise ValueError("cell_class must be 'parvo' or 'magno'.")
        self.params = params
        self.cell_class = cell_class

        if cells_per_mm3 is None:
            cells_per_mm3 = optional(
                params,
                f'magnification.density_derived.cells_per_mm3.{cell_class}')
        if cells_per_mm3 is None:
            raise ValueError(
                f"magnification.density_derived.cells_per_mm3.{cell_class} "
                f"is null (derive it from the atlas), so build this model "
                f"with `from_atlas` or set the value in the config.")
        if callable(cells_per_mm3):
            self.cells_per_mm3_profile = cells_per_mm3
            self.cells_per_mm3 = float(np.mean(
                np.atleast_1d(cells_per_mm3(np.array([1.0, 10.0, 40.0])))))
        else:
            self.cells_per_mm3_profile = None
            self.cells_per_mm3 = float(cells_per_mm3)

        thickness = tuple(float(t) for t
                          in self._configured_thickness(params, cell_class))
        if len(thickness) != 3 or not (thickness[0] <= thickness[1]
                                       <= thickness[2]):
            raise ValueError("column_thickness_mm must be an ascending "
                             "(low, central, high) triple, in mm.")
        self.column_thickness_mm = thickness

    @staticmethod
    def _configured_thickness(params: Mapping, cell_class: str):
        """`column_thickness_mm`: one shared triple, or one per class."""
        configured = require(
            params, 'magnification.density_derived.column_thickness_mm')
        if isinstance(configured, Mapping):
            if cell_class not in configured:
                raise ValueError(
                    f"column_thickness_mm is split by class but has no "
                    f"{cell_class!r} entry; it has {sorted(configured)}.")
            return configured[cell_class]
        return configured

    @classmethod
    def from_atlas(cls, erwin_atlas, params: Mapping,
                   cell_class: str = 'parvo',
                   eccentricity_resolved: bool = False,
                   bin_width_deg: float = 1.0,
                   **kwargs) -> 'MalpeliDensityMagnification':
        """Take rho_v from the atlas's own mean cells per voxel.

        :param eccentricity_resolved: Tabulate rho_v(E) per eccentricity
            bin instead of using one global mean.
        """
        mask = erwin_atlas.eligible(cell_class)
        voxel_volume_mm3 = float(
            require(params, 'atlas.voxel_size_mm')) ** 3
        if not eccentricity_resolved:
            mean_cells_per_voxel = float(
                erwin_atlas.cells_per_voxel[mask].mean())
            return cls(params, cell_class,
                       cells_per_mm3=mean_cells_per_voxel / voxel_volume_mm3,
                       **kwargs)

        sel = tuple(np.argwhere(mask).T)
        ecc = erwin_atlas.eccentricity_deg[sel]
        cells = erwin_atlas.cells_per_voxel[sel]
        edges = np.arange(0.0, float(np.nanmax(ecc)) + bin_width_deg,
                          bin_width_deg)
        which = np.digitize(ecc, edges) - 1
        centers, values = [], []
        for b in range(len(edges) - 1):
            in_bin = which == b
            if in_bin.sum() < 50:
                continue
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            values.append(float(cells[in_bin].mean()) / voxel_volume_mm3)
        if len(centers) < 2:
            raise ValueError(
                "Not enough populated eccentricity bins to build a "
                "rho_v profile; use eccentricity_resolved=False.")
        centers = np.asarray(centers)
        values = np.asarray(values)

        def profile(eccentricity_deg):
            # np.interp extrapolates flat, which keeps rho_v from running
            # to zero past the last bin.
            return np.interp(np.asarray(eccentricity_deg, dtype=float),
                             centers, values)

        profile.centers_deg = centers
        profile.cells_per_mm3 = values
        return cls(params, cell_class, cells_per_mm3=profile, **kwargs)

    def cells_per_mm3_at(self, eccentricity_deg: ArrayLike) -> np.ndarray:
        """rho_v at one or more eccentricities."""
        e = np.asarray(eccentricity_deg, dtype=float)
        if self.cells_per_mm3_profile is None:
            return np.full_like(e, self.cells_per_mm3, dtype=float)
        return np.asarray(self.cells_per_mm3_profile(e), dtype=float)

    def volume_magnification_mm3_per_deg2(self, eccentricity_deg: ArrayLike
                                          ) -> np.ndarray:
        """Malpeli's density divided by volumetric cell density."""
        return (malpeli_density(eccentricity_deg, self.params,
                                self.cell_class)
                / self.cells_per_mm3_at(eccentricity_deg))

    def interval(self, eccentricity_deg: ArrayLike
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(low, central, high) linear magnification in mm/deg."""
        vol = self.volume_magnification_mm3_per_deg2(eccentricity_deg)
        t_lo, t_mid, t_hi = self.column_thickness_mm
        # A thicker column spreads the same volume over less area, so the
        # low magnification comes from the high thickness.
        return (np.sqrt(vol / t_hi), np.sqrt(vol / t_mid),
                np.sqrt(vol / t_lo))

    def at_eccentricity(self, eccentricity_deg: ArrayLike,
                        cell_class: Optional[str] = None,
                        inclination_deg: Optional[ArrayLike] = None
                        ) -> LocalMagnification:
        if cell_class is not None and cell_class != self.cell_class:
            raise ValueError(
                f"This model was built for {self.cell_class!r}; construct a "
                f"separate one for {cell_class!r}.")
        e = np.atleast_1d(np.asarray(eccentricity_deg, dtype=float))
        _, mm_per_deg, _ = self.interval(e)
        return LocalMagnification.isotropic(1.0 / mm_per_deg)
