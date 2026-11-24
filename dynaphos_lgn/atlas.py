"""The Erwin et al. (1999) macaque LGN atlas and the per-voxel Jacobian
of its retinotopy.

The published atlas is a single left LGN, covering the right hemifield.
`MirroredAtlas` and `MirroredJacobianAtlas` reflect it across the
midline to stand in for the right LGN.
"""
import itertools
import logging
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import numpy as np

from dynaphos_lgn.params import resolve_class_codes
from dynaphos_lgn.fovea import apply_central_eccentricity
from dynaphos_lgn.magnification import JacobianMagnification


class Atlas:
    #: Which nucleus this volume represents. Only `MirroredAtlas`
    #: changes it.
    hemisphere = 'left'

    #: True for the closed-form stand-in (`SyntheticAtlas`).
    is_synthetic = False

    #: (n, 3) (ML, DV, AP) voxel indices of the foveola's projection
    #: column, or None when not loaded.
    foveola_column = None

    #: Which central 1 deg `eccentricity_deg` holds: 'published' or
    #: 'rebuilt' (see `fovea.py`).
    central_eccentricity = 'published'

    def __init__(self, params: Optional[Mapping] = None):
        self.params = params


class ErwinAtlas(Atlas):
    """The Erwin, Baker, Busen & Malpeli (1999) macaque LGN atlas.

    Grid geometry, file format, sentinels, origin and layer codes are all
    read from the `atlas` config section.
    """

    def __init__(self, params: Mapping):
        super().__init__(params)
        self.atlas_shape = tuple(params['atlas']['shape_ml_dv_ap'])

        sentinels = params['atlas']['sentinels']
        self.missing_ecc_incl = sentinels['eccentricity_inclination']
        self.missing_layer = sentinels['layer']

        files = params['atlas']['files']
        self._files = files
        self._file_order = params['atlas']['file_order']
        self.ecc_scale = files['eccentricity']['scale']
        self.incl_scale = files['inclination']['scale']
        self.cells_scale = files['cells']['scale']

        # index = (coord - origin) / voxel_size on all three axes
        origin = params['atlas']['horsley_clarke_origin_mm']
        self._origin_ml_mm = origin['ml']
        self._origin_dv_mm = origin['dv']
        self._origin_ap_mm = origin['ap']
        self.voxel_size_mm = params['atlas']['voxel_size_mm']

    @property
    def ipsi_sentinel_values(self) -> list:
        """Inclination values that mark the ipsilateral placeholder."""
        flat = self.params['atlas']['ipsi_flat_inclination_deg']
        return [-flat, flat]

    def is_ipsi_sentinel(self, inclination: Optional[np.ndarray] = None
                         ) -> np.ndarray:
        """True where the ipsilateral placeholder is set.

        :param inclination: Raw inclination values to test. Defaults to
            the whole loaded volume.
        """
        if inclination is None:
            inclination = self.inclination
        return np.isin(inclination, self.ipsi_sentinel_values)

    def eligible(self, cell_class=None,
                 exclude_ipsi: bool = False) -> np.ndarray:
        """Valid voxels, optionally of a cell class (or a list of them)
        and without the ipsilateral placeholder."""
        mask = self.valid.copy()
        if cell_class is not None:
            mask &= np.isin(self.layer,
                            resolve_class_codes(self.params, cell_class))
        if exclude_ipsi:
            mask &= ~self.is_ipsi_sentinel()
        return mask

    def read(self, atlas_dir: Optional[Union[str, Path]] = None
             ) -> 'ErwinAtlas':
        """Load the four .DAT files named in `atlas.files`.

        :param atlas_dir: Directory containing them. Defaults to
            `atlas.directory`.
        """
        if atlas_dir is None:
            atlas_dir = self.params['atlas']['directory']
        atlas_dir = Path(atlas_dir)

        def load(key):
            spec = self._files[key]
            raw = np.fromfile(atlas_dir / spec['name'], dtype=spec['dtype'])
            return raw.reshape(self.atlas_shape, order=self._file_order)

        self.eccentricity = load('eccentricity')
        self.inclination = load('inclination')
        self.layer = load('layer')
        self.cells = load('cells')
        self.clear_foveal_placeholder()
        self.update_valid()
        self.foveola_column = self.load_foveola_column(atlas_dir)

        self.eccentricity_deg = (self.eccentricity
                                 / np.float32(self.ecc_scale))
        self.inclination_deg = (self.inclination
                                / np.float32(self.incl_scale))
        self.cells_per_voxel = (self.cells
                                / np.float32(self.cells_scale))

        return apply_central_eccentricity(self, self.params)

    def clear_foveal_placeholder(self):
        """Move the ipsilateral placeholder off zero eccentricity.

        Inclination is undefined at the fovea, which belongs to both
        hemifields, so +-135 deg there would wrongly exclude foveal
        tissue. It becomes +-90 deg: the vertical meridian, same quadrant.
        """
        at_fovea = (self.eccentricity == 0) & self.is_ipsi_sentinel()
        self.inclination[at_fovea] = (np.sign(self.inclination[at_fovea])
                                      * 90 * self.incl_scale)

    def load_foveola_column(self, atlas_dir: Union[str, Path]
                            ) -> Optional[np.ndarray]:
        """Read `atlas.foveola_file`, checking every voxel is in bounds and
        at zero eccentricity.

        :return: (n, 3) int (ML, DV, AP) indices, or None, with a warning,
            when the file is absent.
        """
        name = self.params['atlas'].get('foveola_file')
        if name is None:
            return None
        path = Path(atlas_dir) / name
        if not path.exists():
            logging.warning("No foveola column at %s; foveola_column is "
                            "None.", path)
            return None

        column = np.loadtxt(path, dtype=int, ndmin=2, encoding='utf-8-sig')
        if (column.shape[1] != 3 or np.any(column < 0)
                or np.any(column >= self.atlas_shape)):
            raise ValueError(f"{path} is not a list of (ML, DV, AP) indices "
                             f"inside the grid {self.atlas_shape}.")
        off_fovea = self.eccentricity[tuple(column.T)] != 0
        if off_fovea.any():
            raise ValueError(f"{off_fovea.sum()} of the {len(column)} voxels "
                             f"in {path} are not at zero eccentricity; the "
                             f"file does not match this atlas.")
        return column

    @property
    def foveola_column_mm(self) -> Optional[np.ndarray]:
        """`foveola_column` in Horsley-Clarke mm, (n, 3) (ML, DV, AP)."""
        if self.foveola_column is None:
            return None
        return np.column_stack(
            self.index_to_horsley_clarke(*self.foveola_column.T))

    @property
    def layer_codes(self) -> np.ndarray:
        """The layer codes present on valid voxels."""
        return np.unique(self.layer[self.valid])

    def update_valid(self):
        """Recompute `valid`: retinotopy known and a layer assigned."""
        self.valid = ((self.eccentricity != self.missing_ecc_incl)
                      & (self.inclination != self.missing_ecc_incl)
                      & (self.layer != self.missing_layer))

    def horsley_clarke_to_index(self, ml_mm: Union[float, np.ndarray],
                                dv_mm: Union[float, np.ndarray],
                                ap_mm: Union[float, np.ndarray],
                                validate: bool = True,
                                hemisphere: Optional[str] = None
                                ) -> tuple:
        """Horsley-Clarke coordinates (mm) to the nearest (ML, DV, AP)
        voxel index.

        :param ml_mm: Lateral distance from the midline, always positive.
        :param dv_mm: Dorsal-ventral coordinate(s), mm.
        :param ap_mm: Anterior-posterior coordinate(s), mm.
        :param validate: Raise IndexError when an index falls outside the
            volume.
        :param hemisphere: 'left' or 'right'. When given it must match
            this atlas's own `hemisphere`.
        :return: (ml_idx, dv_idx, ap_idx), ints for scalar input and
            integer arrays otherwise.
        """
        if hemisphere is not None and hemisphere != self.hemisphere:
            raise ValueError(
                f"This atlas is the {self.hemisphere} LGN, but the "
                f"coordinate was given for the {hemisphere!r} one.")

        def to_index(mm, origin):
            return np.round((np.asarray(mm) - origin)
                            / self.voxel_size_mm).astype(int)

        ml_idx = self._ml_index(to_index(ml_mm, self._origin_ml_mm))
        dv_idx = to_index(dv_mm, self._origin_dv_mm)
        ap_idx = to_index(ap_mm, self._origin_ap_mm)

        if validate:
            for idx, size, name in zip((ml_idx, dv_idx, ap_idx),
                                       self.atlas_shape, ('ML', 'DV', 'AP')):
                if np.any((idx < 0) | (idx >= size)):
                    raise IndexError(
                        f"{name} index out of bounds for atlas shape "
                        f"{self.atlas_shape} (input ml={ml_mm}, dv={dv_mm}, "
                        f"ap={ap_mm} mm).")

        if ml_idx.ndim == 0:
            return int(ml_idx), int(dv_idx), int(ap_idx)
        return ml_idx, dv_idx, ap_idx

    def index_to_horsley_clarke(self, ml_idx, dv_idx, ap_idx) -> tuple:
        """The inverse of `horsley_clarke_to_index`, in both nuclei.

        :return: (ml_mm, dv_mm, ap_mm), floats for scalar input and float
            arrays otherwise. `ml_mm` is lateral distance from the midline.
        """
        def to_mm(index, origin):
            return np.asarray(index) * self.voxel_size_mm + origin

        ml_mm = to_mm(self._ml_index(np.asarray(ml_idx)), self._origin_ml_mm)
        dv_mm = to_mm(dv_idx, self._origin_dv_mm)
        ap_mm = to_mm(ap_idx, self._origin_ap_mm)
        if ml_mm.ndim == 0:
            return float(ml_mm), float(dv_mm), float(ap_mm)
        return ml_mm, dv_mm, ap_mm

    def _ml_index(self, index: np.ndarray) -> np.ndarray:
        """Hook for `MirroredAtlas`, whose ML axis runs the other way."""
        return index


class JacobianAtlas(Atlas):
    """Per-voxel Jacobian of the physical (ML, DV, AP) -> visual-field
    (x, y) mapping, fitted once and cached to disk.

    Fields, all of the atlas's shape:

    :ivar jacobian: (*shape, 2, 3) deg/mm, NaN where the fit failed.
    :ivar jacobian_valid: Whether the fit was accepted.
    :ivar n_neighbors: Valid neighbors used per voxel.
    :ivar residual_rms: In-neighborhood fit residual, deg.
    :ivar transition_probe_residual: Extrapolation mismatch at the probe
        distance, deg.
    :ivar isotropic_flag: Inside `atlas.isotropic_construction_radius_deg`.
    :ivar unreliable_beyond_neighborhood_flag: The Jacobian should not
        be extrapolated past its fit neighborhood.
    """

    def __init__(self, params: Mapping):
        super().__init__(params)
        self.atlas_shape = None
        self.jacobian = None
        self.jacobian_valid = None
        self.n_neighbors = None
        self.residual_rms = None
        self.transition_probe_residual = None
        self.isotropic_flag = None
        self.unreliable_beyond_neighborhood_flag = None
        self._svd = None                # filled by `singular_values`

    def compute(self, atlas: 'ErwinAtlas') -> 'JacobianAtlas':
        """Fit a local linear map, mm offset -> visual-field (x, y)
        offset, at every valid voxel of `atlas`.

        Fits in Cartesian (x, y) rather than (eccentricity, inclination)
        to avoid the polar singularity at the fovea. Parameters come from
        `atlas.jacobian`.
        """
        cfg = self.params['atlas']['jacobian']
        voxel_size = atlas.voxel_size_mm
        valid = atlas.eligible(exclude_ipsi=self.params['atlas'][
            'exclude_ipsi_flat_inclination'])
        incl_rad = np.deg2rad(atlas.inclination_deg)
        x = (atlas.eccentricity_deg * np.cos(incl_rad)).astype(
            np.float32)
        y = (atlas.eccentricity_deg * np.sin(incl_rad)).astype(
            np.float32)

        ata, atb, btb, n_neighbors = self._normal_equations(
            valid, x, y, voxel_size, cfg['neighborhood_radius_vox'])
        fit_ok = valid & (n_neighbors >= cfg['min_neighbors'])
        jacobian = self._solve(ata, atb, cfg['ridge'])
        jacobian[~fit_ok] = np.nan

        # RSS via the OLS identity y'y - beta'(X'y), reusing the sums.
        rss = np.maximum(sum(
            btb[k] - (jacobian[..., k, 0] * atb[0, k]
                      + jacobian[..., k, 1] * atb[1, k]
                      + jacobian[..., k, 2] * atb[2, k])
            for k in range(2)), 0)       # guard tiny float negatives
        residual_rms = np.sqrt(rss / np.maximum(n_neighbors, 1))
        residual_rms[~fit_ok] = np.nan
        del ata, atb, btb, rss

        probe_residual, probe_valid = self._probe_residual(
            jacobian, valid, x, y, voxel_size,
            cfg['transition_probe_distance_vox'])

        self.atlas_shape = atlas.atlas_shape
        self.jacobian = jacobian
        self.jacobian_valid = fit_ok
        self.n_neighbors = n_neighbors
        self.residual_rms = residual_rms
        self.transition_probe_residual = probe_residual
        self.isotropic_flag = fit_ok & (
            atlas.eccentricity_deg
            < self.params['atlas']['isotropic_construction_radius_deg'])
        self.unreliable_beyond_neighborhood_flag = (
            fit_ok & probe_valid
            & (probe_residual > cfg['transition_residual_threshold_deg']))
        return self

    def _normal_equations(self, valid, x, y, voxel_size: float, radius: int):
        """Sums of the least-squares fit over each voxel's neighbors.

        With A the neighbor offsets in mm and b their visual-field
        offsets, returns A^T A (its 6 unique entries, keyed (i, j)),
        A^T b (keyed (axis, x or y)), b^T b (per x, y) and the neighbor
        count. Each is a scalar field of the atlas's shape, to avoid
        multi-GB (*shape, 3, 3) temporaries.
        """
        shape = valid.shape

        def zeros():
            return np.zeros(shape, dtype=np.float32)

        pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        ata = {pair: zeros() for pair in pairs}
        atb = {(i, k): zeros() for i in range(3) for k in range(2)}
        btb = [zeros(), zeros()]
        n_neighbors = zeros()

        steps = range(-radius, radius + 1)
        for offset in itertools.product(steps, steps, steps):
            if offset == (0, 0, 0):
                continue
            shifted_valid, shifted_x, shifted_y = self._shift(
                offset, shape, valid, x, y)
            w = shifted_valid.astype(np.float32)
            a = [step * voxel_size for step in offset]
            dq = (shifted_x - x, shifted_y - y)
            for i, j in pairs:
                ata[i, j] += w * (a[i] * a[j])
            for i in range(3):
                for k in range(2):
                    atb[i, k] += w * a[i] * dq[k]
            for k in range(2):
                btb[k] += w * dq[k] * dq[k]
            n_neighbors += w
        return ata, atb, btb, n_neighbors

    @staticmethod
    def _solve(ata: dict, atb: dict, ridge: float) -> np.ndarray:
        """(*shape, 2, 3) Jacobians from the normal equations, by the
        closed-form inverse of the ridged, symmetric 3x3 A^T A."""
        m00, m01, m02 = ata[0, 0] + ridge, ata[0, 1], ata[0, 2]
        m11, m12, m22 = ata[1, 1] + ridge, ata[1, 2], ata[2, 2] + ridge

        # Adjugate
        c00 = m11 * m22 - m12 * m12
        c01 = m02 * m12 - m01 * m22
        c02 = m01 * m12 - m02 * m11
        c11 = m00 * m22 - m02 * m02
        c12 = m01 * m02 - m00 * m12
        c22 = m00 * m11 - m01 * m01
        det = m00 * c00 + m01 * c01 + m02 * c02
        det = np.where(det == 0, np.finfo(np.float32).tiny, det)
        inverse = [[c00 / det, c01 / det, c02 / det],
                   [c01 / det, c11 / det, c12 / det],
                   [c02 / det, c12 / det, c22 / det]]
        del c00, c01, c02, c11, c12, c22, det

        jacobian = np.empty(m00.shape + (2, 3), dtype=np.float32)
        for k in range(2):
            for j in range(3):
                jacobian[..., k, j] = (inverse[0][j] * atb[0, k]
                                       + inverse[1][j] * atb[1, k]
                                       + inverse[2][j] * atb[2, k])
        return jacobian

    def _probe_residual(self, jacobian, valid, x, y, voxel_size: float,
                        distance_vox: int):
        """Largest mismatch between each voxel's Jacobian, extrapolated
        `distance_vox` along every axis, and the atlas there.

        A 1-voxel fit cannot see a tear in the map; this can.

        :return: (residual in deg, NaN where no probe point is valid;
            whether any probe point is valid).
        """
        shape = valid.shape
        residual = np.zeros(shape, dtype=np.float32)
        any_valid = np.zeros(shape, dtype=bool)
        for axis in range(3):
            for sign in (1, -1):
                offset = [0, 0, 0]
                offset[axis] = sign * distance_vox
                shifted_valid, shifted_x, shifted_y = self._shift(
                    tuple(offset), shape, valid, x, y)
                p = [step * voxel_size for step in offset]
                predicted = [
                    q + jacobian[..., k, 0] * p[0]
                    + jacobian[..., k, 1] * p[1] + jacobian[..., k, 2] * p[2]
                    for k, q in enumerate((x, y))]
                mismatch = np.sqrt((shifted_x - predicted[0]) ** 2
                                   + (shifted_y - predicted[1]) ** 2)
                residual = np.maximum(
                    residual, np.where(shifted_valid, mismatch, 0))
                any_valid |= shifted_valid
        residual[~any_valid] = np.nan
        return residual, any_valid

    def jacobian_at(self, voxel_indices: np.ndarray) -> np.ndarray:
        """The (n, 2, 3) Jacobians at the given (ml, dv, ap) indices."""
        idx = tuple(np.asarray(voxel_indices, dtype=int).T)
        return self.jacobian[idx]

    @property
    def singular_values(self):
        """(major, minor, orientation_rad) of every voxel's Jacobian,
        computed once."""
        if self._svd is None:
            self._svd = self.decompose_valid()
        return self._svd

    def decompose_valid(self):
        """SVD of the fit-ok voxels only, in chunks of
        `atlas.jacobian.svd_chunk_size`.

        :return: (major, minor, orientation_rad), each a float32 array of
            the atlas's shape, NaN where `jacobian_valid` is False.
        """
        chunk_size = self.params['atlas']['jacobian']['svd_chunk_size']
        shape = self.jacobian_valid.shape
        flat_jacobian = self.jacobian.reshape(-1, 2, 3)
        index = np.flatnonzero(self.jacobian_valid.ravel())

        major = np.full(flat_jacobian.shape[0], np.nan, dtype=np.float32)
        minor = np.full_like(major, np.nan)
        orientation = np.full_like(major, np.nan)

        for start in range(0, index.size, chunk_size):
            part = index[start:start + chunk_size]
            a, b, o = JacobianMagnification.decompose(flat_jacobian[part])
            major[part] = a.astype(np.float32)
            minor[part] = b.astype(np.float32)
            orientation[part] = o.astype(np.float32)

        return (major.reshape(shape), minor.reshape(shape),
                orientation.reshape(shape))

    @staticmethod
    def _shift(offset: tuple, shape: tuple, valid: np.ndarray,
               x: np.ndarray, y: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(valid, x, y) re-indexed so position p holds the values of
        p + offset. Unlike np.roll, out-of-bounds positions are invalid
        rather than wrapped."""
        def slices(d, size):
            if d >= 0:
                return slice(d, size), slice(0, size - d)
            return slice(0, size + d), slice(-d, size)

        src = tuple(slices(d, s)[0] for d, s in zip(offset, shape))
        dst = tuple(slices(d, s)[1] for d, s in zip(offset, shape))

        shifted_valid = np.zeros(shape, dtype=bool)
        shifted_x = np.zeros(shape, dtype=x.dtype)
        shifted_y = np.zeros(shape, dtype=y.dtype)
        shifted_valid[dst] = valid[src]
        shifted_x[dst] = x[src]
        shifted_y[dst] = y[src]
        return shifted_valid, shifted_x, shifted_y

    def save(self, path: Union[str, Path]):
        np.savez_compressed(path, jacobian=self.jacobian,
                jacobian_valid=self.jacobian_valid,
                n_neighbors=self.n_neighbors,
                residual_rms=self.residual_rms,
                transition_probe_residual=self.transition_probe_residual,
                isotropic_flag=self.isotropic_flag,
                unreliable_beyond_neighborhood_flag=(
                    self.unreliable_beyond_neighborhood_flag))

    @classmethod
    def load(cls, path: Union[str, Path], params: Mapping) -> 'JacobianAtlas':
        data = np.load(path)
        atlas = cls(params)
        atlas.jacobian = data['jacobian']
        atlas.jacobian_valid = data['jacobian_valid']
        atlas.n_neighbors = data['n_neighbors']
        atlas.residual_rms = data['residual_rms']
        atlas.transition_probe_residual = data['transition_probe_residual']
        atlas.isotropic_flag = data['isotropic_flag']
        atlas.unreliable_beyond_neighborhood_flag = \
            data['unreliable_beyond_neighborhood_flag']
        atlas.atlas_shape = atlas.jacobian_valid.shape
        return atlas


def mirror_inclination(inclination, scale: float = 1.0,
                       sentinel: Optional[float] = None):
    """Reflect inclination across the vertical meridian: I -> 180 - I,
    i.e. visual-field x -> -x.

    :param inclination: Values in the atlas's stored units (deg x `scale`).
    :param scale: The stored-integer scale factor.
    :param sentinel: A "not applicable" value to pass through untouched.
    :return: The reflected values, in the input's dtype.
    """
    values = np.asarray(inclination)
    half_turn = 180.0 * scale
    full_turn = 360.0 * scale
    mirrored = half_turn - values.astype(np.float64)
    # Inputs lie in [-180, 180] x scale, so one subtraction wraps back.
    mirrored = np.where(mirrored > half_turn, mirrored - full_turn, mirrored)
    if sentinel is not None:
        mirrored = np.where(values == sentinel, sentinel, mirrored)
    if np.issubdtype(values.dtype, np.integer):
        mirrored = np.round(mirrored)
    return mirrored.astype(values.dtype)


def mirror_orientation(orientation_rad):
    """Reflect a visual-field axis angle across the vertical meridian.

    x -> -x sends an axis at theta to -theta (period pi). Returned in
    (-pi, pi], like `JacobianMagnification.decompose`.
    """
    theta = np.asarray(orientation_rad)
    mirrored = -theta
    return np.where(mirrored <= -np.pi, mirrored + 2 * np.pi,
                    mirrored).astype(theta.dtype)


def hemifield_of(hemisphere: str) -> str:
    """Which half of the visual field a nucleus represents."""
    if hemisphere == 'left':
        return 'right'
    if hemisphere == 'right':
        return 'left'
    raise ValueError(f"Unknown hemisphere {hemisphere!r}; expected 'left' "
                     f"or 'right'.")


def _ml_reversed(ndim: int, ml_axis: int) -> tuple:
    """Index tuple reversing the ML axis, as a view."""
    return tuple(slice(None, None, -1) if axis == ml_axis else slice(None)
                 for axis in range(ndim))


class MirroredAtlas(ErwinAtlas):
    """The right LGN, as the mirror image of the published left one.

    Reflects the ML grid axis, the inclination (I -> 180 - I) and with it
    the ipsilateral placeholder (+-135 deg becomes +-45 deg). Eccentricity,
    layers and cell counts are NumPy views of the source atlas.

    :param atlas: A built `ErwinAtlas` (or `SyntheticAtlas`) of the left
        LGN.
    :param params: Parameter dictionary. Defaults to the source atlas's.
    """

    hemisphere = 'right'

    def __init__(self, atlas: ErwinAtlas, params: Optional[Mapping] = None):
        params = atlas.params if params is None else params
        super().__init__(params)
        if atlas.hemisphere != 'left':
            raise ValueError(
                f"MirroredAtlas reflects the left LGN, but was given a "
                f"{atlas.hemisphere!r} one.")
        self.base = atlas
        # The source wins on grid geometry: a SyntheticAtlas has its own.
        self.atlas_shape = atlas.atlas_shape
        self.voxel_size_mm = atlas.voxel_size_mm
        self._ml_axis = int(params['atlas']['mirror']['ml_axis'])
        self._reflect()

    @property
    def is_synthetic(self) -> bool:
        return self.base.is_synthetic

    @property
    def _reverse(self) -> tuple:
        return _ml_reversed(len(self.atlas_shape), self._ml_axis)

    def _reflect(self) -> 'MirroredAtlas':
        """Reflect the source atlas."""
        base = self.base
        flip = self._reverse

        self.eccentricity = base.eccentricity[flip]
        self.layer = base.layer[flip]
        self.cells = base.cells[flip]
        self.eccentricity_deg = base.eccentricity_deg[flip]
        self.cells_per_voxel = base.cells_per_voxel[flip]

        self.inclination = mirror_inclination(
            base.inclination[flip], scale=self.incl_scale,
            sentinel=self.missing_ecc_incl)
        self.inclination_deg = (self.inclination
                                / np.float32(self.incl_scale))
        self.update_valid()

        self.central_eccentricity = base.central_eccentricity
        if base.foveola_column is not None:
            self.foveola_column = base.foveola_column.copy()
            ml = self.foveola_column[:, self._ml_axis]
            ml[:] = self.atlas_shape[self._ml_axis] - 1 - ml
        return self

    @property
    def ipsi_sentinel_values(self) -> list:
        """+-135 deg in the left nucleus reflects to +-45 deg here."""
        return [float(mirror_inclination(v))
                for v in super().ipsi_sentinel_values]

    def _ml_index(self, index: np.ndarray) -> np.ndarray:
        """Reflect an ML index, so one lateral distance picks out
        mirror-image tissue in the two nuclei."""
        return self.atlas_shape[self._ml_axis] - 1 - index

    def analytic_magnification_deg_per_mm(self, eccentricity_deg):
        """Forwarded from a synthetic source atlas; a reflection leaves
        both principal magnifications unchanged."""
        return self.base.analytic_magnification_deg_per_mm(eccentricity_deg)


class MirroredJacobianAtlas(JacobianAtlas):
    """The Jacobian field of a `MirroredAtlas`, derived without refitting.

    Reflecting ML and visual-field x turns each Jacobian J into
    ``diag(-1, 1) J diag(-1, 1, 1)``, so the singular values (phosphene
    sizes) are unchanged and only the orientation is reflected.

    :param jacobian_atlas: A computed or loaded `JacobianAtlas` of the
        left LGN.
    :param params: Parameter dictionary.
    """

    def __init__(self, jacobian_atlas: JacobianAtlas, params: Mapping):
        super().__init__(params)
        self.base = jacobian_atlas
        self._ml_axis = int(params['atlas']['mirror']['ml_axis'])
        self.atlas_shape = jacobian_atlas.atlas_shape
        flip = self._reverse

        self.jacobian_valid = jacobian_atlas.jacobian_valid[flip]
        self.n_neighbors = jacobian_atlas.n_neighbors[flip]
        self.residual_rms = jacobian_atlas.residual_rms[flip]
        self.transition_probe_residual = \
            jacobian_atlas.transition_probe_residual[flip]
        self.isotropic_flag = jacobian_atlas.isotropic_flag[flip]
        self.unreliable_beyond_neighborhood_flag = \
            jacobian_atlas.unreliable_beyond_neighborhood_flag[flip]

        # Rows are visual-field (x, y); columns physical (ML, DV, AP).
        # x and ML are the two reflected axes.
        self._sign = np.array([[1.0, -1.0, -1.0],
                               [-1.0, 1.0, 1.0]], dtype=np.float32)
        self._reversed_jacobian = jacobian_atlas.jacobian[flip]
        self._materialized = None

    @property
    def _reverse(self) -> tuple:
        return _ml_reversed(len(self.atlas_shape), self._ml_axis)

    def jacobian_at(self, voxel_indices: np.ndarray) -> np.ndarray:
        idx = tuple(np.asarray(voxel_indices, dtype=int).T)
        return self._reversed_jacobian[idx] * self._sign

    @property
    def jacobian(self) -> np.ndarray:
        """The whole reflected field, materialized on first access.

        Costs a second full-grid copy; the pipeline itself only uses
        `jacobian_at` and `decompose_valid`.
        """
        if self._materialized is None:
            logging.warning(
                "Materializing the full mirrored Jacobian (%s float32, "
                "about %.1f GB).", self._reversed_jacobian.shape,
                self._reversed_jacobian.size * 4 / 1e9)
            self._materialized = self._reversed_jacobian * self._sign
        return self._materialized

    @jacobian.setter
    def jacobian(self, value):
        # JacobianAtlas.__init__ assigns None; refuse anything else so
        # the mirror cannot drift from its source.
        if value is not None:
            raise AttributeError(
                "A MirroredJacobianAtlas derives its Jacobian from the "
                "atlas it reflects; assign to that one instead.")

    def decompose_valid(self):
        """The source's singular values, re-indexed, with the orientation
        reflected."""
        major, minor, orientation = self.base.singular_values
        flip = self._reverse
        return (major[flip], minor[flip],
                mirror_orientation(orientation[flip]))

    def compute(self, atlas) -> 'MirroredJacobianAtlas':
        raise NotImplementedError(
            "A mirrored Jacobian is derived from the left LGN's fit. "
            "Compute the left one and wrap it.")

    def save(self, path):
        raise NotImplementedError(
            "Nothing to cache: save the left LGN's Jacobian instead.")
