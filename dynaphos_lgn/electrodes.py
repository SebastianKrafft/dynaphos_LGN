"""Setup-time electrode geometry for the LGN simulator.

Voxel lookup, the candidate neighborhood, per-candidate layer and
position, the local magnification and the render flags are computed
once here. The forward pass only weights these fixed arrays by a smooth
function of the current.
"""
import logging
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from dynaphos.utils import Map
from dynaphos_lgn.atlas import hemifield_of
from dynaphos_lgn.current_spread import RecruitmentKernel
from dynaphos_lgn.lgn_utils import get_electrode_layout
from dynaphos_lgn.magnification import (JacobianMagnification,
                                        LocalMagnification,
                                        isotropic_construction_radius_deg)


def layer_names(params: Mapping, present=None) -> dict:
    """{code: name} from `atlas.layers.names`, optionally only for the
    codes in `present`."""
    return {int(code): name for code, name
            in params['atlas']['layers']['names'].items()
            if present is None or int(code) in present}


@dataclass
class ElectrodeFlags:
    """Per-electrode flags.

    :ivar central_isotropic: Inside the atlas's isotropic center.
    :ivar unreliable_beyond_neighborhood: The center Jacobian should not
        be extrapolated across the footprint; rendered by scatter.
    :ivar touches_placeholder: The candidate neighborhood includes
        placeholder or ipsilateral-sentinel voxels.
    :ivar class_heterogeneous: The neighborhood spans more than one
        cell class.
    :ivar scalar_magnification: No Jacobian at the electrode; sized by
        the inclination-averaged magnification at its eccentricity.
    """

    central_isotropic: np.ndarray
    unreliable_beyond_neighborhood: np.ndarray
    touches_placeholder: np.ndarray
    class_heterogeneous: np.ndarray
    scalar_magnification: np.ndarray

    def any_flag(self) -> np.ndarray:
        return (self.central_isotropic | self.unreliable_beyond_neighborhood
                | self.touches_placeholder | self.class_heterogeneous
                | self.scalar_magnification)

    def summary(self) -> str:
        n = len(self.central_isotropic)
        rows = [('central-1 deg, isotropic by construction',
                 self.central_isotropic),
                ('Jacobian unreliable beyond its fit neighborhood',
                 self.unreliable_beyond_neighborhood),
                ('footprint touches placeholder/sentinel voxels',
                 self.touches_placeholder),
                ('footprint spans more than one cell class',
                 self.class_heterogeneous),
                ('sized by the eccentricity table, not the Jacobian',
                 self.scalar_magnification)]
        lines = [f"Electrode confidence flags ({n} electrodes):"]
        for label, mask in rows:
            lines.append(f"  {int(mask.sum()):>4} / {n}  {label}")
        return '\n'.join(lines)


def cloud_bandwidth_deg(points: np.ndarray, center: np.ndarray,
                        extent_percentile: float) -> float:
    """Splat bandwidth that hides a point cloud's lattice, degrees.

    :param points: (n, 2) visual-field positions of one cloud.
    :param center: (2,) the electrode's own position.
    :param extent_percentile: Percentile of distance from `center` that
        counts as the cloud's extent.
    """
    if len(points) < 2:
        return 0.0
    sub = points if len(points) <= 256 else points[
        np.linspace(0, len(points) - 1, 256).astype(int)]
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nearest = np.min(d, axis=1)
    # The atlas's quantization makes distinct voxels share a retinotopic
    # position, so drop the zero-spacing pairs.
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    nn_spacing = float(np.median(nearest)) if nearest.size else 0.0

    # Nearest-neighbor spacing alone leaves lattice artifacts, so also
    # compare against a uniform cloud of the same count and area.
    extent = float(np.percentile(np.linalg.norm(points - center, axis=-1),
                                 extent_percentile))
    uniform_spacing = (np.sqrt(np.pi * extent ** 2 / len(points))
                       if extent > 0 else 0.0)
    return max(nn_spacing, uniform_spacing)


class LGNElectrodeArray:
    """A fixed set of LGN electrodes, with everything precomputed.

    The candidate neighborhood is sized for `electrodes.max_current_ua`,
    so the array is valid for any current up to it. The hemisphere is
    taken from the atlas.

    :param atlas: A built `ErwinAtlas` (or `MirroredAtlas`).
    :param voxel_indices: (n_electrodes, 3) integer (ml, dv, ap) indices.
    :param kernel: The `RecruitmentKernel` used at runtime; it also sizes
        the neighborhood.
    :param jacobian_atlas: Computed/loaded `JacobianAtlas`.
    :param rng: Used only for the scatter subsample.
    """

    def __init__(self, atlas, params: Mapping, voxel_indices: np.ndarray,
                 kernel: Optional[RecruitmentKernel] = None,
                 jacobian_atlas=None,
                 rng: Optional[np.random.Generator] = None):
        self.atlas = atlas
        self.params = params
        self.kernel = kernel or RecruitmentKernel.from_params(params)
        self.max_current_ua = float(
            params['electrodes']['max_current_ua'])
        self.rng = np.random.default_rng() if rng is None else rng
        self.exclude_ipsi_flat_inclination = params['atlas'][
            'exclude_ipsi_flat_inclination']
        tail_weight = params['current_spread']['tail_weight']
        max_candidate_voxels = params['electrodes'][
            'max_candidate_voxels']
        n_scatter_points = params['electrodes']['n_scatter_points']

        self.hemisphere = atlas.hemisphere

        self.voxel_indices = np.asarray(voxel_indices, dtype=int)
        if self.voxel_indices.ndim != 2 or self.voxel_indices.shape[1] != 3:
            raise ValueError("voxel_indices must have shape "
                             "(n_electrodes, 3).")
        self.n_electrodes = len(self.voxel_indices)

        self._resolve_positions()
        self._build_candidate_neighborhood(tail_weight, max_candidate_voxels)
        self._gather_candidate_properties()
        self._resolve_magnification(jacobian_atlas)
        self._select_scatter_points(n_scatter_points)
        self._precompute_scatter_bandwidth()

    # Constructors
    @classmethod
    def spread_in_visual_field(cls, atlas, params: Mapping,
                               rng: Optional[np.random.Generator] = None,
                               n_electrodes: Optional[int] = None,
                               **kwargs) -> 'LGNElectrodeArray':
        """Farthest-point-sample electrodes for even visual-field coverage.

        :param n_electrodes: Defaults to `electrodes.n_electrodes`.
        """
        rng = np.random.default_rng() if rng is None else rng
        if n_electrodes is None:
            n_electrodes = params['electrodes']['n_electrodes']
        _, voxels = get_electrode_layout(
            atlas, n_electrodes,
            cell_class=params['electrodes']['cell_class'],
            min_eccentricity=params['electrodes']['min_eccentricity_deg'],
            max_eccentricity=params['electrodes']['max_eccentricity_deg'],
            bin_size_deg=params['electrodes']['bin_size_deg'],
            exclude_ipsi_flat_inclination=params['atlas'][
                'exclude_ipsi_flat_inclination'],
            depth_selection=params['electrodes']['depth_selection'],
            rng=rng)
        return cls(atlas, params, voxels, rng=rng, **kwargs)

    @classmethod
    def from_visual_field_targets(cls, atlas, params: Mapping,
                                  eccentricity_deg: Sequence[float],
                                  inclination_deg: Sequence[float],
                                  **kwargs) -> 'LGNElectrodeArray':
        """Put an electrode at the voxel nearest each requested percept
        location; `position_error_deg` records the miss."""
        ecc = np.atleast_1d(np.asarray(eccentricity_deg, dtype=float))
        incl = np.atleast_1d(np.asarray(inclination_deg, dtype=float))
        valid = atlas.eligible(
            params['electrodes']['cell_class'],
            params['atlas']['exclude_ipsi_flat_inclination'])
        idx = np.argwhere(valid)
        vx = atlas.eccentricity_deg[valid] * np.cos(
            np.deg2rad(atlas.inclination_deg[valid]))
        vy = atlas.eccentricity_deg[valid] * np.sin(
            np.deg2rad(atlas.inclination_deg[valid]))
        tx = ecc * np.cos(np.deg2rad(incl))
        ty = ecc * np.sin(np.deg2rad(incl))
        chosen = np.empty((len(tx), 3), dtype=int)
        errors = np.empty(len(tx))
        for i, (x0, y0) in enumerate(zip(tx, ty)):
            d2 = (vx - x0) ** 2 + (vy - y0) ** 2
            j = int(np.argmin(d2))
            chosen[i] = idx[j]
            errors[i] = float(np.sqrt(d2[j]))
        array = cls(atlas, params, chosen, **kwargs)
        array.position_error_deg = errors
        return array

    # Setup steps
    def _resolve_positions(self):
        """Look up each electrode's retinotopic position, with a small
        neighborhood fallback where it lands on a placeholder voxel."""
        atlas = self.atlas
        idx = tuple(self.voxel_indices.T)
        ecc = np.asarray(atlas.eccentricity_deg[idx], dtype=float).copy()
        incl = np.asarray(atlas.inclination_deg[idx], dtype=float).copy()
        on_tissue = np.asarray(atlas.valid[idx], dtype=bool).copy()

        self.fell_on_placeholder = ~on_tissue
        n_bad = int(self.fell_on_placeholder.sum())
        if n_bad:
            logging.info("%d of %d electrodes landed on a placeholder voxel; "
                         "falling back to a local neighborhood average.",
                         n_bad, self.n_electrodes)
            for i in np.flatnonzero(self.fell_on_placeholder):
                mean = self._neighborhood_position(self.voxel_indices[i])
                if mean is None:
                    ecc[i], incl[i] = np.nan, np.nan
                else:
                    ecc[i], incl[i] = mean

        self.eccentricity_deg = ecc
        self.inclination_deg = incl
        self.visual_field = Map(r=ecc, phi=np.deg2rad(incl))
        self.vf_xy = np.stack(self.visual_field.cartesian, axis=-1)

    def _neighborhood_position(self, voxel):
        """Mean (ecc, incl) of the valid voxels within
        `electrodes.position_fallback_radius_vox`, averaged in Cartesian
        coordinates to avoid the +-180 deg wrap."""
        atlas = self.atlas
        radius = self.params['electrodes']['position_fallback_radius_vox']
        lo = np.maximum(voxel - radius, 0)
        hi = np.minimum(voxel + radius + 1, atlas.atlas_shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        valid = atlas.valid[sl]
        if not valid.any():
            return None
        ecc = atlas.eccentricity_deg[sl][valid]
        incl = np.deg2rad(atlas.inclination_deg[sl][valid])
        x, y = np.mean(ecc * np.cos(incl)), np.mean(ecc * np.sin(incl))
        return float(np.hypot(x, y)), float(np.rad2deg(np.arctan2(y, x)))

    def _build_candidate_neighborhood(self, tail_weight, max_candidate_voxels):
        """Fixed sphere of candidate voxels around every electrode.

        Sized from the effective kernel radius at `max_current_ua`. A
        stride keeps the count within `electrodes.max_candidate_voxels`;
        `voxels_per_candidate` records what each candidate stands for.
        """
        voxel_size = self.atlas.voxel_size_mm
        self.max_radius_mm = float(self.kernel.effective_radius_mm(
            self.max_current_ua, tail_weight))
        self.nominal_radius_mm = float(
            self.kernel.radius_mm(self.max_current_ua))
        radius_vox = self.max_radius_mm / voxel_size

        stride = 1
        while True:
            r_int = int(np.ceil(radius_vox / stride))
            estimate = (2 * r_int + 1) ** 3 * np.pi / 6.0
            if estimate <= max_candidate_voxels or stride >= 16:
                break
            stride += 1
        self.candidate_stride_vox = stride
        self.voxels_per_candidate = stride ** 3

        r_int = int(np.ceil(radius_vox / stride))
        axis = np.arange(-r_int, r_int + 1) * stride
        grid = np.meshgrid(axis, axis, axis, indexing='ij')
        offsets = np.stack([g.ravel() for g in grid], axis=1)
        offsets_mm = offsets * voxel_size
        distance_mm = np.linalg.norm(offsets_mm, axis=1)
        keep = distance_mm <= self.max_radius_mm

        self.candidate_offsets_vox = offsets[keep]
        self.candidate_offsets_mm = offsets_mm[keep]
        self.candidate_distance_mm = distance_mm[keep]
        self.n_candidates = int(keep.sum())
        logging.info("Candidate neighborhood: %d voxels per electrode "
                     "(stride %d, radius %.3f mm at I_max = %.1f uA).",
                     self.n_candidates, stride, self.max_radius_mm,
                     self.max_current_ua)

    def _gather_candidate_properties(self):
        """Layer, class, validity and position for every candidate."""
        atlas = self.atlas
        shape = np.asarray(atlas.atlas_shape)
        idx = (self.voxel_indices[:, None, :]
               + self.candidate_offsets_vox[None, :, :])

        in_bounds = np.all((idx >= 0) & (idx < shape), axis=-1)
        clipped = np.clip(idx, 0, shape - 1)
        flat = tuple(clipped.reshape(-1, 3).T)

        valid = atlas.valid[flat].reshape(in_bounds.shape) & in_bounds
        if self.exclude_ipsi_flat_inclination:
            sentinel = atlas.is_ipsi_sentinel(
                atlas.inclination[flat].reshape(in_bounds.shape))
            self.candidate_is_sentinel = sentinel & in_bounds
            valid &= ~sentinel
        else:
            self.candidate_is_sentinel = np.zeros_like(valid)
        # Placeholder: a labeled layer, but no retinotopy.
        self.candidate_is_placeholder = in_bounds & (
            atlas.layer[flat].reshape(in_bounds.shape)
            != atlas.missing_layer) & ~atlas.valid[flat].reshape(
            in_bounds.shape)

        layer = atlas.layer[flat].reshape(in_bounds.shape).astype(np.int8)
        layer = np.where(valid, layer, 0)

        ecc = atlas.eccentricity_deg[flat].reshape(in_bounds.shape)
        incl_rad = np.deg2rad(
            atlas.inclination_deg[flat].reshape(in_bounds.shape))
        vf_x = np.where(valid, ecc * np.cos(incl_rad),
                        self.vf_xy[:, 0:1])
        vf_y = np.where(valid, ecc * np.sin(incl_rad),
                        self.vf_xy[:, 1:2])

        cells = atlas.cells_per_voxel[flat].reshape(in_bounds.shape)

        self.candidate_voxels = clipped
        self.candidate_valid = valid
        self.candidate_layer = layer
        self.candidate_vf_xy = np.stack([vf_x, vf_y], axis=-1)
        self.candidate_cells_per_voxel = np.where(valid, cells, 0.0)

        self.layer_names_by_code = layer_names(self.params,
                                               atlas.layer_codes)
        self.layer_ids = np.array(sorted(self.layer_names_by_code),
                                  dtype=np.int8)
        self.candidate_layer_onehot = (
            layer[..., None] == self.layer_ids[None, None, :]).astype(
            np.float32)

        classes = self.params['atlas']['layers']['classes']
        self.touches_class = {name: np.isin(layer, codes).any(axis=1)
                              for name, codes in classes.items()}
        self.tip_cell_class = self._nearest_cell_class(layer)

    def _nearest_cell_class(self, layer: np.ndarray) -> np.ndarray:
        """Receptive-field cell class at each tip: that of the nearest
        candidate voxel in a class, '' when the neighborhood has none."""
        classes = self.params['atlas']['layers']['classes']
        names = [''] + list(self.params['receptive_fields']['cell_classes'])
        index = np.zeros(layer.shape, dtype=int)
        for i, name in enumerate(names[1:], start=1):
            index[np.isin(layer, classes[name])] = i
        distance = np.where(index > 0, self.candidate_distance_mm, np.inf)
        nearest = np.argmin(distance, axis=1)
        return np.array(names)[index[np.arange(len(layer)), nearest]]

    def _resolve_magnification(self, jacobian_atlas):
        """Local magnification at each electrode center, plus flags.

        Electrodes without a Jacobian take the inclination-averaged
        magnification at their eccentricity. Anything still missing is
        reported by `LGNSize` and renders at zero size.
        """
        if jacobian_atlas is None:
            raise ValueError("LGNElectrodeArray needs a jacobian_atlas.")
        self.jacobian_atlas = jacobian_atlas
        self.magnification_model = JacobianMagnification(
            jacobian_atlas, self.params, self.atlas)
        local = self.magnification_model.at_voxels(self.voxel_indices)
        missing = ~local.valid | ~np.isfinite(local.major_deg_per_mm)

        if missing.any():
            cell_class = self.params['electrodes']['cell_class']
            fallback = self.magnification_model.at_eccentricity(
                self.eccentricity_deg,
                cell_class if isinstance(cell_class, str) else 'parvo')
            for attr in ('major_deg_per_mm', 'minor_deg_per_mm',
                         'orientation_rad'):
                setattr(local, attr,
                        np.where(missing, getattr(fallback, attr),
                                 getattr(local, attr)))
            local.valid = local.valid | fallback.valid

        self.local_magnification: LocalMagnification = local
        self.flags = ElectrodeFlags(
            central_isotropic=np.asarray(
                local.isotropic_by_construction, dtype=bool)
            | (self.eccentricity_deg
               < isotropic_construction_radius_deg(self.params)),
            unreliable_beyond_neighborhood=np.asarray(
                local.unreliable_beyond_neighborhood, dtype=bool),
            touches_placeholder=np.any(
                self.candidate_is_sentinel | self.candidate_is_placeholder,
                axis=1),
            class_heterogeneous=np.sum(list(self.touches_class.values()),
                                       axis=0) > 1,
            scalar_magnification=missing)

    def _select_scatter_points(self, n_scatter_points: int):
        """Subsample candidates for scatter rendering, by distance rank
        with a stride so the outer shell is not over-represented."""
        n_points = min(n_scatter_points, self.n_candidates)
        # Stable: grid distances tie, and numpy versions break ties apart.
        order = np.argsort(self.candidate_distance_mm, kind='stable')
        step = max(1, self.n_candidates // n_points)
        chosen = order[::step][:n_points]
        self.scatter_index = np.sort(chosen)
        self.n_scatter_points = len(self.scatter_index)

    def _precompute_scatter_bandwidth(self):
        """Floor for the scatter splat kernel, in degrees, from the point
        cloud's own spacing."""
        pts = self.candidate_vf_xy[:, self.scatter_index, :]
        valid = self.candidate_valid[:, self.scatter_index]
        percentile = self.params['rendering']['scatter_extent_percentile']
        # Masked-out candidates sit at the electrode center; including
        # them would drive the median spacing to zero.
        self.scatter_bandwidth_floor_deg = np.array([
            cloud_bandwidth_deg(pts[i][valid[i]], self.vf_xy[i], percentile)
            for i in range(self.n_electrodes)])

    # Derived views
    @property
    def render_mode(self) -> np.ndarray:
        """'scatter' where the center Jacobian should not be extrapolated
        across the footprint, 'ellipse' otherwise (always 'ellipse' in
        the isotropic center)."""
        mode = np.where(self.flags.unreliable_beyond_neighborhood,
                        'scatter', 'ellipse')
        mode = np.where(self.flags.central_isotropic, 'ellipse', mode)
        return mode

    def phosphene_sigma_deg(self, current_ua: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Phosphene sigmas (major, minor) in degrees: the recruitment
        sigma in mm times the two principal magnifications."""
        sigma_mm = np.asarray(self.kernel.sigma_mm(current_ua), dtype=float)
        return (sigma_mm * self.local_magnification.major_deg_per_mm,
                sigma_mm * self.local_magnification.minor_deg_per_mm)

    def distance_histograms(self) -> dict:
        """Per-(electrode, layer) histograms of candidate distance, into
        `recruitment.distance_histogram_bins` bins.

        The weight depends only on distance and current, so the forward
        pass can weight bins instead of every candidate.

        :return: dict with ``bin_centers_mm`` (n_bins,), ``counts``
            (n_electrodes, n_layers, n_bins), ``cells`` (same shape,
            weighted by cells per voxel and by `voxels_per_candidate`).
        """
        n_bins = self.params['recruitment']['distance_histogram_bins']
        edges = np.linspace(0.0, self.max_radius_mm, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        which = np.clip(np.digitize(self.candidate_distance_mm, edges) - 1,
                        0, n_bins - 1)

        n_layers = len(self.layer_ids)
        counts = np.zeros((self.n_electrodes, n_layers, n_bins))
        cells = np.zeros_like(counts)
        for li, code in enumerate(self.layer_ids):
            mask = (self.candidate_layer == code) & self.candidate_valid
            for e in range(self.n_electrodes):
                sel = mask[e]
                if not sel.any():
                    continue
                counts[e, li] = np.bincount(which[sel], minlength=n_bins)
                cells[e, li] = np.bincount(
                    which[sel], weights=self.candidate_cells_per_voxel[e][sel],
                    minlength=n_bins)
        return {'bin_centers_mm': centers,
                'counts': counts * self.voxels_per_candidate,
                'cells': cells * self.voxels_per_candidate,
                'layer_ids': self.layer_ids,
                'layer_names': [self.layer_names_by_code[int(c)]
                                for c in self.layer_ids]}

    @property
    def scatter_points_xy(self) -> np.ndarray:
        """(n_electrodes, n_scatter_points, 2) point positions, degrees."""
        return self.candidate_vf_xy[:, self.scatter_index, :]

    @property
    def scatter_distance_mm(self) -> np.ndarray:
        """(n_scatter_points,) distance of each scatter point from its
        electrode, shared across electrodes."""
        return self.candidate_distance_mm[self.scatter_index]

    @property
    def scatter_valid(self) -> np.ndarray:
        """(n_electrodes, n_scatter_points) tissue mask for scatter points."""
        return self.candidate_valid[:, self.scatter_index]

    def scatter_patch_radius_deg(self) -> np.ndarray:
        """Half-width of each electrode's scatter render patch, degrees,
        from the spread of its points plus a margin."""
        pts = self.scatter_points_xy
        valid = self.scatter_valid
        radius = np.zeros(self.n_electrodes)
        for i in range(self.n_electrodes):
            p = pts[i][valid[i]]
            if len(p) == 0:
                radius[i] = 0.0
                continue
            radius[i] = float(np.percentile(
                np.linalg.norm(p - self.vf_xy[i], axis=-1),
                self.params['rendering']['scatter_patch_radius_percentile']))
        return radius * self.params['rendering']['scatter_patch_radius_margin']

    def per_layer_cell_counts(self) -> np.ndarray:
        """(n_electrodes, n_layers) reachable cells at maximum current."""
        cells = (self.candidate_cells_per_voxel[..., None]
                 * self.candidate_layer_onehot)
        return cells.sum(axis=1) * self.voxels_per_candidate

    def describe(self) -> str:
        hemifield = hemifield_of(self.hemisphere)
        mirrored = ('' if self.hemisphere == 'left'
                    else ', mirrored from the published left atlas')
        lines = [
            f"LGNElectrodeArray: {self.n_electrodes} electrodes",
            f"  nucleus           : {self.hemisphere} LGN -> {hemifield} "
            f"hemifield{mirrored}",
            f"  eccentricity      : "
            f"{np.nanmin(self.eccentricity_deg):.2f} - "
            f"{np.nanmax(self.eccentricity_deg):.2f} deg",
            f"  I_max             : {self.max_current_ua:.1f} uA",
            f"  spread radius     : {self.nominal_radius_mm * 1e3:.0f} um "
            f"nominal, {self.max_radius_mm * 1e3:.0f} um gathered",
            f"  candidates        : {self.n_candidates} per electrode "
            f"(stride {self.candidate_stride_vox}, each standing for "
            f"{self.voxels_per_candidate} voxels)",
            f"  scatter points    : {self.n_scatter_points}",
            f"  render mode       : "
            f"{int((self.render_mode == 'ellipse').sum())} ellipse, "
            f"{int((self.render_mode == 'scatter').sum())} scatter",
            self.flags.summary(),
        ]
        return '\n'.join(lines)
