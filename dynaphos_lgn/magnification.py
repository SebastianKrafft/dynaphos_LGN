"""Retinotopic magnification models for the macaque LGN.

All three models answer one question: How much visual field (deg) does
a patch of tissue (mm) cover? All return it in the unit of
degrees per millimetre, which is what turns a current-spread radius into
a phosphene size.

`JacobianMagnification`
    Per-voxel and anisotropic, from the cached `JacobianAtlas`. Two
    principal magnifications plus the major axis's orientation, so a
    circular patch of tissue can render as an ellipse. The default.

`AtlasGradientMagnification`
    Isotropic scalar M(E) tabulated by eccentricity bin. Cheap, smooth,
    and needs no voxel index.

`MalpeliDensityMagnification`
    Malpeli et al. (1996)'s closed-form cell density converted to a
    linear magnification. The conversion needs two quantities the
    formula does not supply, so it returns an interval.

None of the three is an independent measurement of the others -- they
all trace back to one macaque's 415 recording sites, so agreement
between them is not a cross-check.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import numpy as np

from dynaphos_lgn.params import optional, require, resolve_class_codes

ArrayLike = Union[float, np.ndarray]

# NumPy 2.0 renamed `trapz` to `trapezoid` and removed the old spelling.
# requirements.txt still pins numpy==1.26.4, where only `trapz` exists,
# so bind whichever is present rather than picking one and breaking the
# other. Same function, same signature -- purely a rename.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


# ----------------------------------------------------------------------
# Malpeli, Lee & Baker (1996) closed-form cell-density functions
# ----------------------------------------------------------------------
# Eqs 1-2, coefficients in `magnification.malpeli`. These are CELL
# DENSITY (cells/deg^2), not a tissue magnification (mm^2/deg^2).

def malpeli_parvo_density(eccentricity_deg: ArrayLike,
                          params: Mapping) -> np.ndarray:
    """Parvocellular cell density, cells/deg^2. Malpeli et al. (1996) Eq. 1.

    ``a * (E + e0) ** exponent``
    """
    c = require(params, 'magnification.malpeli.parvo')
    e = np.asarray(eccentricity_deg, dtype=float)
    return c['a'] * (e + c['e0']) ** c['exponent']


def malpeli_magno_density(eccentricity_deg: ArrayLike,
                          params: Mapping) -> np.ndarray:
    """Magnocellular cell density, cells/deg^2. Malpeli et al. (1996) Eq. 2.

    ``a * ((E - e0)^2 + c) ** exponent``

    Non-monotonic near the fovea by construction, not by numerical
    artefact -- magnocellular density genuinely does not fall
    monotonically with eccentricity.
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
    construction, not by measurement."""
    return float(require(params, 'atlas.isotropic_construction_radius_deg'))


# ----------------------------------------------------------------------
# The common return type
# ----------------------------------------------------------------------

@dataclass
class LocalMagnification:
    """Local tissue -> visual-field scaling at one or more points.

    All magnitudes are degrees of visual angle per millimetre of tissue,
    i.e. the forward direction (tissue -> percept).

    :ivar major_deg_per_mm: Larger principal magnification (sigma_1).
    :ivar minor_deg_per_mm: Smaller principal magnification (sigma_2).
        Equal to `major_deg_per_mm` for isotropic models.
    :ivar orientation_rad: Direction of the major axis in visual-field
        coordinates, radians. Zero (and meaningless) for isotropic
        models.
    :ivar isotropic_by_construction: True where the atlas's retinotopy
        is isotropic because Erwin et al. (1999) wrote an isotropic
        formula in, not because anisotropy was measured and found
        absent.
    :ivar unreliable_beyond_neighborhood: True where this single local
        linear map should not be extrapolated across a whole
        current-spread footprint. Route these electrodes to per-voxel
        scatter rendering instead of the ellipse shortcut.
    :ivar valid: True where a magnification could be estimated at all.
    """

    major_deg_per_mm: np.ndarray
    minor_deg_per_mm: np.ndarray
    orientation_rad: np.ndarray
    isotropic_by_construction: np.ndarray
    unreliable_beyond_neighborhood: np.ndarray
    valid: np.ndarray

    @property
    def anisotropy(self) -> np.ndarray:
        """Ratio of the two principal magnifications, >= 1.

        A validation target only -- never a simulator input.
        """
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
        """Isotropic-equivalent linear magnification, mm/deg.

        1/sqrt(areal): the side of the square patch of tissue covering
        one square degree. For an anisotropic model this DISCARDS the
        anisotropy, so do not use it where the ellipse matters.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return 1.0 / np.sqrt(self.areal_deg2_per_mm2)

    def __len__(self):
        return int(np.size(self.major_deg_per_mm))


class MagnificationModel:
    """Interface: turn a location into a `LocalMagnification`."""

    def at_voxels(self, voxel_indices: np.ndarray) -> LocalMagnification:
        raise NotImplementedError

    def at_eccentricity(self, eccentricity_deg: ArrayLike,
                        cell_class: str = 'parvo') -> LocalMagnification:
        raise NotImplementedError


class JacobianMagnification(MagnificationModel):
    """Per-voxel anisotropic magnification, from the cached Jacobian atlas.

    The Jacobian is a 2x3 matrix mapping a physical displacement (mm, in
    ML/DV/AP) to a visual-field displacement (deg, in a Cartesian
    x = E*cos(I), y = E*sin(I) frame). Its two singular values are the
    principal magnifications; the leading left singular vector gives the
    major axis's direction in the visual field. LGN layers are thin
    curved sheets, so the third physical direction carries no
    retinotopic gradient.
    """

    def __init__(self, jacobian_atlas, params: Mapping, erwin_atlas=None):
        """
        :param jacobian_atlas: A `dynaphos_lgn.atlas.JacobianAtlas` that
            has been computed or loaded.
        :param params: Parameter dictionary.
        :param erwin_atlas: Optional built `ErwinAtlas`, only needed for
            `at_eccentricity` (which averages over voxels at a given
            eccentricity).
        """
        self.jacobian_atlas = jacobian_atlas
        self.params = params
        self.erwin_atlas = erwin_atlas
        self._svd_cache = {}

    # -- core ---------------------------------------------------------
    @staticmethod
    def decompose(jacobian: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SVD of a stack of 2x3 Jacobians.

        :param jacobian: (..., 2, 3) array, deg/mm.
        :return: (major, minor, orientation_rad). `major >= minor >= 0`
            are the singular values in deg/mm; `orientation_rad` is
            atan2 of the leading left singular vector, i.e. the major
            axis's direction in visual-field (x, y) coordinates.
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
        """Magnification at the given (ml, dv, ap) atlas indices.

        :param voxel_indices: (n, 3) integer array.
        """
        idx = tuple(np.asarray(voxel_indices, dtype=int).T)
        # Through `jacobian_at`, not `jacobian[idx]`, so a mirrored
        # atlas can serve the lookup from the left LGN's cached fit
        # instead of materialising a reflected copy of the whole field.
        jac = self.jacobian_atlas.jacobian_at(voxel_indices)
        major, minor, orientation = self.decompose(jac)
        valid = self.jacobian_atlas.jacobian_valid[idx]
        return LocalMagnification(
            major_deg_per_mm=major,
            minor_deg_per_mm=minor,
            orientation_rad=orientation,
            isotropic_by_construction=self.jacobian_atlas.isotropic_flag[idx],
            unreliable_beyond_neighborhood=(
                self.jacobian_atlas.unreliable_beyond_neighborhood_flag[idx]),
            valid=valid)

    def at_eccentricity(self, eccentricity_deg: ArrayLike,
                        cell_class: str = 'parvo') -> LocalMagnification:
        """Inclination-averaged magnification at the given eccentricities,
        binned at `magnification.eccentricity_table_bin_width_deg`.

        Requires `erwin_atlas`. Averaging over inclination discards the
        anisotropy the Jacobian exists to capture, so this is for
        validation against literature curves, not for rendering.
        """
        if self.erwin_atlas is None:
            raise ValueError("at_eccentricity needs the ErwinAtlas "
                             "(pass erwin_atlas= to the constructor).")
        bin_width_deg = require(
            self.params, 'magnification.eccentricity_table_bin_width_deg')
        ecc_query = np.atleast_1d(np.asarray(eccentricity_deg, dtype=float))
        table = self._eccentricity_table(cell_class, bin_width_deg)
        centers, major_t, minor_t, orient_t = table
        return LocalMagnification(
            major_deg_per_mm=np.interp(ecc_query, centers, major_t),
            minor_deg_per_mm=np.interp(ecc_query, centers, minor_t),
            orientation_rad=np.interp(ecc_query, centers, orient_t),
            isotropic_by_construction=(
                ecc_query
                < isotropic_construction_radius_deg(self.params)),
            unreliable_beyond_neighborhood=np.zeros_like(ecc_query, bool),
            valid=np.ones_like(ecc_query, bool))

    def _eccentricity_table(self, cell_class: str, bin_width_deg: float):
        key = (cell_class, bin_width_deg)
        if key in self._svd_cache:
            return self._svd_cache[key]

        ea = self.erwin_atlas
        ja = self.jacobian_atlas
        mask = ea.valid & ja.jacobian_valid
        if cell_class in ('parvo', 'magno'):
            mask = mask & np.isin(
                ea.layer, resolve_class_codes(self.params, cell_class))

        idx = np.argwhere(mask)
        major, minor, orientation = self.decompose(ja.jacobian_at(idx))
        ecc = ea.eccentricity_deg[tuple(idx.T)]

        edges = np.arange(0, np.nanmax(ecc) + bin_width_deg, bin_width_deg)
        centers = 0.5 * (edges[:-1] + edges[1:])
        which = np.digitize(ecc, edges) - 1
        n_bins = len(centers)

        def binned_median(values):
            out = np.full(n_bins, np.nan)
            for b in range(n_bins):
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
        self._svd_cache[key] = table
        return table


class AtlasGradientMagnification(MagnificationModel):
    """Isotropic scalar magnification M(E), tabulated in mm/deg.

    A lookup table over eccentricity bins. Every phosphene it sizes
    comes out circular, and it sets `isotropic_by_construction`
    unconditionally so that choice stays visible downstream.
    """

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
                      cell_class: str = 'parvo'
                      ) -> 'AtlasGradientMagnification':
        bin_width_deg = require(
            jacobian_magnification.params,
            'magnification.eccentricity_table_bin_width_deg')
        centers, major, minor, _ = jacobian_magnification._eccentricity_table(
            cell_class, bin_width_deg)
        with np.errstate(divide='ignore', invalid='ignore'):
            mm_per_deg = 1.0 / np.sqrt(major * minor)
        return cls(centers, mm_per_deg)

    def at_eccentricity(self, eccentricity_deg: ArrayLike,
                        cell_class: str = 'parvo') -> LocalMagnification:
        e = np.atleast_1d(np.asarray(eccentricity_deg, dtype=float))
        mm_per_deg = np.interp(e, self.bin_centers_deg, self.mm_per_deg)
        deg_per_mm = 1.0 / mm_per_deg
        return LocalMagnification(
            major_deg_per_mm=deg_per_mm,
            minor_deg_per_mm=deg_per_mm,
            orientation_rad=np.zeros_like(deg_per_mm),
            isotropic_by_construction=np.ones_like(deg_per_mm, bool),
            unreliable_beyond_neighborhood=np.zeros_like(deg_per_mm, bool),
            valid=np.isfinite(deg_per_mm))

    def at_voxels(self, voxel_indices, erwin_atlas=None
                  ) -> LocalMagnification:
        if erwin_atlas is None:
            raise ValueError("AtlasGradientMagnification needs an "
                             "ErwinAtlas to turn voxel indices into "
                             "eccentricities.")
        idx = tuple(np.asarray(voxel_indices, dtype=int).T)
        return self.at_eccentricity(erwin_atlas.eccentricity_deg[idx])


class MalpeliDensityMagnification(MagnificationModel):
    """Linear magnification derived from Malpeli et al. (1996)'s density.

    Malpeli publishes CELL DENSITY in cells/deg^2. Converting that to a
    tissue magnification needs a volumetric cell density rho_v and a
    projection-column thickness T, neither of which the formula fixes,
    plus an isotropy assumption in the final square root. So this class
    returns an INTERVAL: `at_eccentricity` gives the central estimate,
    `interval` gives (low, central, high).

    T is unresolved by roughly an order of magnitude between two checks
    -- treat this model as the MAGNITUDE estimate and the
    Jacobian/atlas-gradient route as the SHAPE estimate, and do not
    report either as agreeing with the other.
    """

    def __init__(self, params: Mapping, cell_class: str = 'parvo',
                 cells_per_mm3: Optional[float] = None):
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
                f"is null, which means 'derive it from the atlas' -- so "
                f"build this model with `from_atlas`, or set the value in "
                f"the config.")
        if callable(cells_per_mm3):
            # An eccentricity-resolved profile. Packing density is NOT
            # uniform across the nucleus: it falls monotonically with
            # eccentricity, by 1.5x (parvo) and 2.6x (magno) from centre
            # to far periphery, and volume magnification is rho_v's
            # direct reciprocal, so a single scalar is wrong by up to 2x
            # for magno in the central field. Still a scalar by default;
            # this only makes the better thing expressible.
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
        """`column_thickness_mm`, per class if the config splits it.

        Two accepted shapes, because the measured value turns out to be
        strongly class-dependent while the original config was a single
        shared triple:

        - one ascending triple, shared by both classes (the original);
        - a mapping with a `parvo` and a `magno` triple.

        Measured from the atlas, summed parvo T is 0.56-1.77 mm and
        summed magno T 0.20-0.74 mm -- ~2.4x apart, so one shared triple
        is necessarily wrong for one of the two classes. This accessor
        makes adopting a split value a config edit rather than a code
        change; it changes no default.
        """
        configured = require(
            params, 'magnification.density_derived.column_thickness_mm')
        if isinstance(configured, Mapping):
            if cell_class not in configured:
                raise ValueError(
                    f"column_thickness_mm is split by class but has no "
                    f"{cell_class!r} entry; it has "
                    f"{sorted(configured)}. A partial split would "
                    f"silently fall back to the other class's "
                    f"thickness, so it is refused.")
            return configured[cell_class]
        return configured

    @property
    def map_length_anchor_mm(self) -> Tuple[float, float, float]:
        """Approximate extent of the macaque LGN along the eccentricity
        axis, used only by `calibrate_column_thickness_mm` as a coarse
        anchor."""
        return tuple(require(
            self.params,
            'magnification.density_derived.map_length_anchor_mm'))

    # -- calibration --------------------------------------------------
    def map_length_mm(self, column_thickness_mm: float) -> float:
        """int M_linear(E) dE -- the map's extent along the eccentricity
        axis, in mm, for one assumed column thickness.

        Integrated over the range
        `magnification.density_derived.map_length_integration` names.
        Checkable against the nucleus's real physical size, and what
        `calibrate_column_thickness_mm` inverts.
        """
        cfg = require(
            self.params,
            'magnification.density_derived.map_length_integration')
        e = np.linspace(cfg['min_eccentricity_deg'],
                        cfg['max_eccentricity_deg'], cfg['n_samples'])
        vol = self.volume_magnification_mm3_per_deg2(e)
        return float(_trapezoid(np.sqrt(vol / column_thickness_mm), e))

    def calibrate_column_thickness_mm(self,
                                      target_map_length_mm: float) -> float:
        """Back-solve the column thickness implied by a target map length.

        Closed-form, since M_linear scales as 1/sqrt(T):
        T = T_ref * (L_ref / L_target)^2. The configured
        `thickness_plausibility_bracket_mm` is a sanity check only -- an
        out-of-range result is logged and returned unclamped.
        """
        bracket = require(
            self.params,
            'magnification.density_derived.'
            'thickness_plausibility_bracket_mm')
        reference_t = 1.0
        reference_length = self.map_length_mm(reference_t)
        implied = reference_t * (reference_length / target_map_length_mm) ** 2
        if not (bracket[0] <= implied <= bracket[1]):
            logging.warning(
                "Implied column thickness %.3f mm falls outside the "
                "plausibility bracket %s -- reporting it unclamped, since "
                "an implausible value is itself the finding.",
                implied, bracket)
        return implied

    @classmethod
    def from_atlas(cls, erwin_atlas, params: Mapping,
                   cell_class: str = 'parvo',
                   eccentricity_resolved: bool = False,
                   bin_width_deg: float = 1.0,
                   **kwargs) -> 'MalpeliDensityMagnification':
        """Take rho_v from the atlas's own mean cells/voxel.

        Preferred over an external packing figure: CELLS.DAT already
        counts real cells in real voxels.

        :param eccentricity_resolved: Build a tabulated rho_v(E) instead
            of one global mean. Packing density measurably is not
            uniform -- it falls by 1.5x (parvo) to 2.6x (magno) from the
            centre to the far periphery, so the global mean is wrong by
            up to 2x for magno inside 2 deg. Defaults to False, which
            keeps the historical behaviour.
        """
        mask = (erwin_atlas.valid
                & np.isin(erwin_atlas.layer,
                          resolve_class_codes(params, cell_class)))
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
            # Flat extrapolation outside the tabulated range rather than
            # a linear one: rho_v falls monotonically, so extrapolating
            # the trend past the last bin would run it towards zero and
            # send the magnification to infinity.
            return np.interp(np.asarray(eccentricity_deg, dtype=float),
                             centers, values)

        profile.centers_deg = centers
        profile.cells_per_mm3 = values
        return cls(params, cell_class, cells_per_mm3=profile, **kwargs)

    def cells_per_mm3_at(self, eccentricity_deg: ArrayLike) -> np.ndarray:
        """Volumetric cell density at one or more eccentricities.

        The configured scalar, unless the model was built with a
        profile, in which case the profile is evaluated. `from_atlas`
        builds one with `eccentricity_resolved=True`.
        """
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
        """(low, central, high) linear magnification in mm/deg.

        The width comes from the layer-thickness interval alone; rho_v
        shifts all three bounds together.
        """
        vol = self.volume_magnification_mm3_per_deg2(eccentricity_deg)
        t_lo, t_mid, t_hi = self.column_thickness_mm
        # areal = volume / thickness; thicker layer -> smaller area per
        # deg^2 -> smaller linear magnification. So the LOW magnification
        # comes from the HIGH thickness.
        return (np.sqrt(vol / t_hi), np.sqrt(vol / t_mid),
                np.sqrt(vol / t_lo))

    def at_eccentricity(self, eccentricity_deg: ArrayLike,
                        cell_class: Optional[str] = None
                        ) -> LocalMagnification:
        if cell_class is not None and cell_class != self.cell_class:
            raise ValueError(
                f"This model was built for {self.cell_class!r}; construct a "
                f"separate one for {cell_class!r} (the two Malpeli "
                f"equations have different functional forms).")
        e = np.atleast_1d(np.asarray(eccentricity_deg, dtype=float))
        _, mm_per_deg, _ = self.interval(e)
        deg_per_mm = 1.0 / mm_per_deg
        return LocalMagnification(
            major_deg_per_mm=deg_per_mm,
            minor_deg_per_mm=deg_per_mm,
            orientation_rad=np.zeros_like(deg_per_mm),
            isotropic_by_construction=np.ones_like(deg_per_mm, bool),
            unreliable_beyond_neighborhood=np.zeros_like(deg_per_mm, bool),
            valid=np.isfinite(deg_per_mm))

    def at_voxels(self, voxel_indices, erwin_atlas=None
                  ) -> LocalMagnification:
        if erwin_atlas is None:
            raise ValueError("MalpeliDensityMagnification needs an "
                             "ErwinAtlas to turn voxel indices into "
                             "eccentricities.")
        idx = tuple(np.asarray(voxel_indices, dtype=int).T)
        return self.at_eccentricity(erwin_atlas.eccentricity_deg[idx])
