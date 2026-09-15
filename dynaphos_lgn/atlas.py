"""Loading the Erwin et al. (1999) macaque LGN atlas, and fitting the
per-voxel Jacobian of its retinotopy.
"""
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import numpy as np

from dynaphos_lgn.params import require


class Atlas:
    def __init__(self, params: Optional[Mapping] = None):
        self.params = params

    def load_mapping(self, path : Union[str, Path], shape: Optional,
                     dtype: Optional = np.int16, order: str = 'C'):
        raw = np.fromfile(path, dtype=dtype)
        if shape is not None:
            map = raw.reshape(shape, order=order)
            return map


class ErwinAtlas(Atlas):
    """The Erwin, Baker, Busen & Malpeli (1999) macaque LGN atlas.

    Every property of the grid and of the file format -- shape, voxel
    size, Fortran ordering, per-file dtype and scale factor, sentinel
    values, the Horsley-Clarke origin and the laminar code table -- is
    read from the `atlas` section of the parameter dictionary rather
    than hardcoded here, so a refined or differently-gridded atlas is a
    config change instead of a code change.

    `LAYER_CODES` and `IPSI_FLAT_INCLINATION_DEG` are config-backed
    views kept under their original names for existing callers.
    """

    def __init__(self, params: Mapping):
        super().__init__(params)
        self.atlas_shape = tuple(require(params, 'atlas.shape_ml_dv_ap'))

        sentinels = require(params, 'atlas.sentinels')
        self._missing_ecc_incl = sentinels['eccentricity_inclination']
        self._missing_layer = sentinels['layer']

        files = require(params, 'atlas.files')
        self._files = files
        self._file_order = require(params, 'atlas.file_order')
        self._ecc_scale = files['eccentricity']['scale']
        self._incl_scale = files['inclination']['scale']
        self._cells_scale = files['cells']['scale']

        # index = (coord - origin) / voxel_size on all three axes; the
        # atlas's documentation fixes the origin and the directions.
        origin = require(params, 'atlas.horsley_clarke_origin_mm')
        self._origin_ml_mm = origin['ml']
        self._origin_dv_mm = origin['dv']
        self._origin_ap_mm = origin['ap']
        self._voxel_size_mm = require(params, 'atlas.voxel_size_mm')

    # -- config-backed views of what used to be class constants --------
    @property
    def LAYER_CODES(self) -> dict:
        """{'magno': {1, 2}, 'parvo': {3, 4}} as the config defines it."""
        return {name: set(codes) for name, codes
                in require(self.params, 'atlas.layers.classes').items()}

    @property
    def IPSI_FLAT_INCLINATION_DEG(self) -> float:
        """Coarse +-135 deg placeholder for the two ipsilateral-hemifield
        quadrants, not a resolved value."""
        return require(self.params, 'atlas.ipsi_flat_inclination_deg')

    def is_ipsi_sentinel(self, inclination: Optional[np.ndarray] = None
                         ) -> np.ndarray:
        """True where the coarse +-135 deg ipsilateral placeholder is set.

        :param inclination: Raw inclination values to test. Defaults to
            the whole loaded volume.
        """
        if inclination is None:
            inclination = self.inclination
        flat = self.IPSI_FLAT_INCLINATION_DEG
        return np.isin(inclination, [-flat, flat])

    def build_atlas(self, atlas_dir: Optional[Union[str, Path]] = None
                    ) -> 'ErwinAtlas':
        """Load the four .DAT files named in `atlas.files`.

        :param atlas_dir: Directory containing them. Defaults to
            `atlas.directory` from the config.
        :return: self, with the raw and scaled volumes populated.
        """
        # `atlas.file_order` is Fortran: ML is the fastest-varying axis
        # in the flat file, not NumPy's default C order.
        if atlas_dir is None:
            atlas_dir = require(self.params, 'atlas.directory')
        atlas_dir = Path(atlas_dir)

        def load(key):
            spec = self._files[key]
            return self.load_mapping(atlas_dir / spec['name'],
                                     self.atlas_shape, dtype=spec['dtype'],
                                     order=self._file_order)

        self.eccentricity = load('eccentricity')
        self.inclination = load('inclination')
        self.layer = load('layer')
        self.cells = load('cells')

        self.valid = (self.eccentricity != self._missing_ecc_incl) & \
                    (self.inclination != self._missing_ecc_incl) & \
                    (self.layer != self._missing_layer)

        self.eccentricity_deg = (self.eccentricity
                                 / np.float32(self._ecc_scale))
        self.inclination_deg = (self.inclination
                                / np.float32(self._incl_scale))
        self.cells_per_voxel = (self.cells
                                / np.float32(self._cells_scale))

        return self

    def horsley_clarke_to_index(self, ml_mm: Union[float, np.ndarray],
                                dv_mm: Union[float, np.ndarray],
                                ap_mm: Union[float, np.ndarray],
                                validate: bool = True
                                ) -> tuple:
        """Convert Horsley-Clarke stereotaxic coordinates (mm) to the
        nearest integer index/indices into the atlas's (ML, DV, AP) voxel
        array.

        :param ml_mm: Medial-lateral coordinate(s), mm.
        :param dv_mm: Dorsal-ventral coordinate(s), mm.
        :param ap_mm: Anterior-posterior coordinate(s), mm.
        :param validate: If True (default), raise IndexError when any
            resulting index falls outside the atlas volume. If False,
            indices are returned unclipped and out-of-bounds values are
            left to the caller to handle.
        :return: (ml_idx, dv_idx, ap_idx), each an int (scalar input) or
            an integer np.ndarray (array input), usable to directly index
            self.eccentricity / self.inclination / self.layer / self.cells.
        """
        def to_index(mm, origin):
            return np.round((np.asarray(mm) - origin)
                            / self._voxel_size_mm).astype(int)

        ml_idx = to_index(ml_mm, self._origin_ml_mm)
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


class JacobianAtlas(Atlas):
    """Per-voxel local Jacobian of the physical (ML, DV, AP) position ->
    visual-field (x, y) mapping, fitted once and cached to disk.

    Derived rather than published, so the cache is a self-describing
    .npz rather than the raw headerless layout the .DAT files use.

    Fields, all of the atlas's shape:

    :ivar jacobian: (*shape, 2, 3) deg/mm, NaN where the fit failed.
    :ivar jacobian_valid: Whether the fit was accepted.
    :ivar n_neighbors: Valid neighbours used per voxel; diagnostic.
    :ivar residual_rms: In-neighborhood fit residual, deg.
    :ivar transition_probe_residual: Longer-range extrapolation
        mismatch, deg, behind `unreliable_beyond_neighborhood_flag`.
    :ivar isotropic_flag: Inside the radius where the atlas's retinotopy
        is isotropic by construction.
    :ivar unreliable_beyond_neighborhood_flag: Do not extrapolate this
        voxel's Jacobian past its fit neighbourhood. Not a tear
        detector.
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
        self._svd = None                # lazily filled by `singular_values`

    @property
    def ISOTROPIC_CONSTRUCTION_RADIUS_DEG(self) -> float:
        """Eccentricity within which the atlas's retinotopy is isotropic
        by construction rather than by measurement."""
        return require(self.params, 'atlas.isotropic_construction_radius_deg')

    def compute(self, erwin_atlas: 'ErwinAtlas') -> 'JacobianAtlas':
        """Fit a local linear map, physical mm offset -> visual-field
        (x, y) offset, at every valid voxel of `erwin_atlas`.

        Differentiates the Cartesian (x, y) representation of the visual
        field rather than raw (eccentricity, inclination), which is
        polar and so has a foveal singularity and a locally varying
        scale. The body fits by batched normal equations to stay
        within memory.

        All fit parameters -- neighborhood radius, minimum neighbor
        count, the ridge regularizer, the transition-probe distance and
        residual threshold, and whether to exclude the coarse ipsilateral
        placeholder -- come from the `atlas.jacobian` config section.

        :param erwin_atlas: A built ErwinAtlas.
        :return: self, with every field in the class docstring populated.
        """
        cfg = require(self.params, 'atlas.jacobian')
        neighborhood_radius_vox = cfg['neighborhood_radius_vox']
        min_neighbors = cfg['min_neighbors']
        ridge = cfg['ridge']
        transition_probe_distance_vox = cfg['transition_probe_distance_vox']
        transition_residual_threshold_deg = cfg[
            'transition_residual_threshold_deg']
        exclude_ipsi_flat_inclination = require(
            self.params, 'atlas.exclude_ipsi_flat_inclination')

        shape = erwin_atlas.atlas_shape
        voxel_size = erwin_atlas._voxel_size_mm
        dtype = np.float32

        valid = erwin_atlas.valid.copy()
        if exclude_ipsi_flat_inclination:
            valid &= ~erwin_atlas.is_ipsi_sentinel()

        incl_rad = np.deg2rad(erwin_atlas.inclination_deg)
        x = (erwin_atlas.eccentricity_deg * np.cos(incl_rad)).astype(dtype)
        y = (erwin_atlas.eccentricity_deg * np.sin(incl_rad)).astype(dtype)

        # A^T A (symmetric, 6 unique entries) and A^T b, accumulated as
        # separate scalar fields rather than (*shape, 3, 3) arrays, to
        # avoid multi-GB broadcast temporaries.
        m00 = np.zeros(shape, dtype=dtype)
        m01 = np.zeros(shape, dtype=dtype)
        m02 = np.zeros(shape, dtype=dtype)
        m11 = np.zeros(shape, dtype=dtype)
        m12 = np.zeros(shape, dtype=dtype)
        m22 = np.zeros(shape, dtype=dtype)
        r0x = np.zeros(shape, dtype=dtype)
        r0y = np.zeros(shape, dtype=dtype)
        r1x = np.zeros(shape, dtype=dtype)
        r1y = np.zeros(shape, dtype=dtype)
        r2x = np.zeros(shape, dtype=dtype)
        r2y = np.zeros(shape, dtype=dtype)
        n_neighbors = np.zeros(shape, dtype=dtype)
        sum_dqx2 = np.zeros(shape, dtype=dtype)
        sum_dqy2 = np.zeros(shape, dtype=dtype)

        r = neighborhood_radius_vox
        for dm in range(-r, r + 1):
            for dd in range(-r, r + 1):
                for da in range(-r, r + 1):
                    if dm == 0 and dd == 0 and da == 0:
                        continue

                    shifted_valid, shifted_x, shifted_y = self._shift(
                        (dm, dd, da), shape, valid, x, y)

                    w = shifted_valid.astype(dtype)
                    am, ad, aa = (dm * voxel_size, dd * voxel_size,
                                 da * voxel_size)
                    dqx = shifted_x - x
                    dqy = shifted_y - y

                    m00 += w * (am * am)
                    m01 += w * (am * ad)
                    m02 += w * (am * aa)
                    m11 += w * (ad * ad)
                    m12 += w * (ad * aa)
                    m22 += w * (aa * aa)

                    r0x += w * am * dqx
                    r0y += w * am * dqy
                    r1x += w * ad * dqx
                    r1y += w * ad * dqy
                    r2x += w * aa * dqx
                    r2y += w * aa * dqy

                    sum_dqx2 += w * dqx * dqx
                    sum_dqy2 += w * dqy * dqy

                    n_neighbors += w

        fit_ok = valid & (n_neighbors >= min_neighbors)

        m00 += ridge
        m11 += ridge
        m22 += ridge

        # Closed-form inverse of the symmetric 3x3 via its adjugate and
        # determinant
        c00 = m11 * m22 - m12 * m12
        c01 = m02 * m12 - m01 * m22
        c02 = m01 * m12 - m02 * m11
        c11 = m00 * m22 - m02 * m02
        c12 = m01 * m02 - m00 * m12
        c22 = m00 * m11 - m01 * m01
        det = m00 * c00 + m01 * c01 + m02 * c02
        det = np.where(det == 0, np.finfo(dtype).tiny, det)

        inv00, inv01, inv02 = c00 / det, c01 / det, c02 / det
        inv11, inv12 = c11 / det, c12 / det
        inv22 = c22 / det
        del c00, c01, c02, c11, c12, c22, det
        del m00, m01, m02, m11, m12, m22

        jacobian = np.empty(shape + (2, 3), dtype=dtype)
        jacobian[..., 0, 0] = inv00 * r0x + inv01 * r1x + inv02 * r2x
        jacobian[..., 0, 1] = inv01 * r0x + inv11 * r1x + inv12 * r2x
        jacobian[..., 0, 2] = inv02 * r0x + inv12 * r1x + inv22 * r2x
        jacobian[..., 1, 0] = inv00 * r0y + inv01 * r1y + inv02 * r2y
        jacobian[..., 1, 1] = inv01 * r0y + inv11 * r1y + inv12 * r2y
        jacobian[..., 1, 2] = inv02 * r0y + inv12 * r1y + inv22 * r2y
        jacobian[~fit_ok] = np.nan
        del inv00, inv01, inv02, inv11, inv12, inv22

        # RSS via the OLS identity y'y - beta'(X'y), reusing what was
        # accumulated above instead of a second pass over the offsets.
        rss_x = sum_dqx2 - (jacobian[..., 0, 0] * r0x
                           + jacobian[..., 0, 1] * r1x
                           + jacobian[..., 0, 2] * r2x)
        rss_y = sum_dqy2 - (jacobian[..., 1, 0] * r0y
                           + jacobian[..., 1, 1] * r1y
                           + jacobian[..., 1, 2] * r2y)
        rss = np.maximum(rss_x + rss_y, 0)  # guard tiny float negatives
        del rss_x, rss_y, sum_dqx2, sum_dqy2
        del r0x, r0y, r1x, r1y, r2x, r2y
        residual_rms = np.sqrt(rss / np.maximum(n_neighbors, 1))
        residual_rms[~fit_ok] = np.nan
        del rss

        isotropic_flag = fit_ok & (erwin_atlas.eccentricity_deg
                                   < self.ISOTROPIC_CONSTRUCTION_RADIUS_DEG)

        # The unreliability flag cannot use residual_rms: at a 1-voxel
        # neighborhood the fit sits entirely on one side of a ~20-voxel
        # tear and never sees it. Instead extrapolate each voxel's own
        # Jacobian out to a probe point and compare against the atlas.
        # Smooth curvature contributes mismatch there too, which is why
        # the threshold sits well above the noise floor.
        probe_vox = transition_probe_distance_vox
        probe_offsets = [(probe_vox, 0, 0), (-probe_vox, 0, 0),
                         (0, probe_vox, 0), (0, -probe_vox, 0),
                         (0, 0, probe_vox), (0, 0, -probe_vox)]
        max_probe_residual = np.zeros(shape, dtype=dtype)
        any_probe_valid = np.zeros(shape, dtype=bool)
        for offset in probe_offsets:
            shifted_valid_p, shifted_x_p, shifted_y_p = self._shift(
                offset, shape, valid, x, y)
            pm, pd, pa = (offset[0] * voxel_size, offset[1] * voxel_size,
                         offset[2] * voxel_size)
            pred_x = (x + jacobian[..., 0, 0] * pm
                     + jacobian[..., 0, 1] * pd + jacobian[..., 0, 2] * pa)
            pred_y = (y + jacobian[..., 1, 0] * pm
                     + jacobian[..., 1, 1] * pd + jacobian[..., 1, 2] * pa)
            probe_residual = np.sqrt((shifted_x_p - pred_x) ** 2
                                     + (shifted_y_p - pred_y) ** 2)
            probe_residual = np.where(shifted_valid_p, probe_residual, 0)
            max_probe_residual = np.maximum(max_probe_residual, probe_residual)
            any_probe_valid |= shifted_valid_p
        max_probe_residual[~any_probe_valid] = np.nan

        unreliable = (fit_ok & any_probe_valid
                      & (max_probe_residual
                         > transition_residual_threshold_deg))

        self.atlas_shape = shape
        self.jacobian = jacobian
        self.jacobian_valid = fit_ok
        self.n_neighbors = n_neighbors
        self.residual_rms = residual_rms
        self.transition_probe_residual = max_probe_residual
        self.isotropic_flag = isotropic_flag
        self.unreliable_beyond_neighborhood_flag = unreliable
        return self

    @property
    def singular_values(self):
        """(major, minor, orientation_rad) of every voxel's Jacobian.

        Lazy and cached: the SVD is not free, and `decompose_valid`
        touches only the fit-ok voxels to keep LAPACK's temporaries out
        of the gigabytes.
        """
        if getattr(self, '_svd', None) is None:
            self._svd = self.decompose_valid()
        return self._svd

    def decompose_valid(self):
        """SVD the fit-ok voxels only, in chunks of
        `atlas.jacobian.svd_chunk_size`.

        :return: (major, minor, orientation_rad), each a float32 array of
            the atlas's shape, NaN wherever `jacobian_valid` is False.
        """
        from dynaphos_lgn.magnification import JacobianMagnification

        chunk_size = require(self.params, 'atlas.jacobian.svd_chunk_size')
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
        """Return (valid, x, y) re-indexed so that position p holds the
        values of position p + offset, with out-of-bounds positions
        marked invalid -- unlike np.roll, this does not wrap around."""
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
                # On-disk key kept as the historical name so older
                # caches still load.
                near_transition_flag=self.unreliable_beyond_neighborhood_flag)

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
            data['near_transition_flag']
        atlas.atlas_shape = atlas.jacobian_valid.shape
        return atlas


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(
        "Building and caching the Jacobian atlas lives in "
        "dynaphos_lgn/build.py, which also prints a validation report:\n"
        "    python -m dynaphos_lgn.build --cache-jacobian")
