"""Setup-time electrode geometry for the LGN simulator.

Everything expensive and everything categorical happens here once, at
construction, and never in the forward pass: voxel lookup, the candidate
neighbourhood, per-candidate layer/class/position, the local Jacobian,
and the flags that decide how each electrode renders. The forward pass
then only multiplies those fixed tensors by a smooth function of the
optimised current, which is what makes per-layer recruitment
differentiable at all.

Electrode positions are assumed fixed (implant time, not optimised).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from dynaphos.utils import Map
from dynaphos_lgn.current_spread import RecruitmentKernel
from dynaphos_lgn.magnification import (JacobianMagnification,
                                        LocalMagnification, MagnificationModel,
                                        isotropic_construction_radius_deg)
from dynaphos_lgn.params import require, resolve_class_codes


def layer_names(params: Mapping) -> dict:
    """{code: name} from `atlas.layers.names`.

    "contra"/"ipsi" in these names is EYE OF ORIGIN -- a different axis
    from INCL.DAT's contralateral/ipsilateral HEMIFIELD sentinel. The
    two are independent, and conflating them is the classic mistake with
    this dataset.
    """
    return {int(code): name for code, name
            in require(params, 'atlas.layers.names').items()}


def layer_to_class(params: Mapping) -> dict:
    """{code: 'magno'|'parvo'} from `atlas.layers.classes`."""
    out = {}
    for name, codes in require(params,
                                       'atlas.layers.classes').items():
        for code in codes:
            out[int(code)] = name
    return out


@dataclass
class ElectrodeFlags:
    """Per-electrode confidence flags, carried through to every output.

    :ivar central_isotropic: The electrode's recruited tissue reaches
        into the central 1 deg, where Erwin et al. (1999) assigned
        eccentricity from Malpeli's isotropic formula rather than from
        interpolated data. Any anisotropy reported there is an artefact
        of construction, not a measurement.
    :ivar unreliable_beyond_neighborhood: A single point-evaluated
        Jacobian should not be extrapolated across this electrode's
        whole footprint. Routes the electrode to scatter rendering.
    :ivar touches_placeholder: The candidate neighbourhood includes
        placeholder or ipsilateral-sentinel voxels, so the recruited
        footprint is masked and incomplete at that boundary.
    :ivar class_heterogeneous: The candidate neighbourhood spans both
        magnocellular and parvocellular tissue, so the single
        centre-voxel Jacobian used by the ellipse shortcut cannot be
        class-correct for all of it.
    :ivar scalar_magnification: This electrode was sized by the cheap
        scalar-magnification fallback, i.e. forced circular.
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
                ('Jacobian unreliable beyond its fit neighbourhood',
                 self.unreliable_beyond_neighborhood),
                ('footprint touches placeholder/sentinel voxels',
                 self.touches_placeholder),
                ('footprint spans both magno and parvo tissue',
                 self.class_heterogeneous),
                ('sized by scalar magnification (forced circular)',
                 self.scalar_magnification)]
        lines = [f"Electrode confidence flags ({n} electrodes):"]
        for label, mask in rows:
            lines.append(f"  {int(mask.sum()):>4} / {n}  {label}")
        return '\n'.join(lines)


class LGNElectrodeArray:
    """A fixed set of LGN electrodes, with everything precomputed.

    :param atlas: A built `dynaphos_lgn.atlas.ErwinAtlas`.
    :param voxel_indices: (n_electrodes, 3) integer (ml, dv, ap) indices.
    :param kernel: The `RecruitmentKernel` that will be used at runtime.
        Passed in at setup because the candidate neighbourhood must be
        sized from the *same* kernel that later weights it -- a kernel
        with a longer tail needs a wider gather.
    :param jacobian_atlas: Optional computed/loaded `JacobianAtlas`. If
        absent, `magnification_model` must be given, and every electrode
        is flagged `scalar_magnification`.
    :param magnification_model: Fallback/override magnification model.
    :param rng: Used only for the scatter subsample.

    The array takes its hemisphere from the atlas it is built on, so an
    array on a `MirroredAtlas` is the right LGN's and its electrodes sit
    in the left hemifield. Nothing else in this class changes between
    the two: the mirror is entirely inside the atlas and the Jacobian.

    Everything else comes from the config: `electrodes.max_current_ua`
    (the largest current the protocol will ever deliver, which sizes the
    candidate neighbourhood -- the array is then valid for any current at
    or below it), `current_spread.tail_weight` (the kernel weight at
    which the gather radius is truncated to keep the neighbourhood
    finite), `electrodes.max_candidate_voxels` (the per-electrode
    candidate budget; a voxel stride is chosen automatically to stay
    under it, so the recruitment sum samples the footprint rather than
    enumerating it -- fine for a weighted average, and scaled correctly
    by `voxels_per_candidate`) and `electrodes.n_scatter_points` (how
    many candidate voxels each electrode keeps for per-voxel scatter
    rendering).
    """

    def __init__(self, atlas, params: Mapping, voxel_indices: np.ndarray,
                 kernel: Optional[RecruitmentKernel] = None,
                 jacobian_atlas=None,
                 magnification_model: Optional[MagnificationModel] = None,
                 rng: Optional[np.random.Generator] = None):
        self.atlas = atlas
        self.params = params
        self.kernel = kernel or RecruitmentKernel.from_params(params)
        self.max_current_ua = float(
            require(params, 'electrodes.max_current_ua'))
        self.rng = np.random.default_rng() if rng is None else rng
        self.exclude_ipsi_flat_inclination = require(
            params, 'atlas.exclude_ipsi_flat_inclination')
        tail_weight = require(params, 'current_spread.tail_weight')
        max_candidate_voxels = require(params,
                                       'electrodes.max_candidate_voxels')
        n_scatter_points = require(params, 'electrodes.n_scatter_points')

        self.hemisphere = getattr(atlas, 'hemisphere', 'left')

        self.voxel_indices = np.asarray(voxel_indices, dtype=int)
        if self.voxel_indices.ndim != 2 or self.voxel_indices.shape[1] != 3:
            raise ValueError("voxel_indices must have shape "
                             "(n_electrodes, 3).")
        self.n_electrodes = len(self.voxel_indices)

        self._resolve_positions()
        self._build_candidate_neighborhood(tail_weight, max_candidate_voxels)
        self._gather_candidate_properties()
        self._resolve_magnification(jacobian_atlas, magnification_model)
        self._select_scatter_points(n_scatter_points)
        self._precompute_scatter_bandwidth()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def spread_in_visual_field(cls, atlas, params: Mapping,
                               rng: Optional[np.random.Generator] = None,
                               n_electrodes: Optional[int] = None,
                               **kwargs) -> 'LGNElectrodeArray':
        """Farthest-point-sample electrodes for even visual-field coverage.

        Spreads in retinotopic rather than physical space, and the
        shipped `electrodes.depth_selection` is ``random`` so electrodes
        do not all land in one lamina.

        :param n_electrodes: How many to place. Defaults to
            `electrodes.n_electrodes`, which is per nucleus; the builder
            passes the per-hemisphere override when the config sets one.
        """
        from dynaphos_lgn.lgn_utils import get_electrode_layout
        rng = np.random.default_rng() if rng is None else rng
        if n_electrodes is None:
            n_electrodes = require(params, 'electrodes.n_electrodes')
        _, voxels = get_electrode_layout(
            atlas, n_electrodes,
            cell_class=require(params, 'electrodes.cell_class'),
            min_eccentricity=require(params,
                                     'electrodes.min_eccentricity_deg'),
            max_eccentricity=require(params,
                                     'electrodes.max_eccentricity_deg'),
            bin_size_deg=require(params, 'electrodes.bin_size_deg'),
            exclude_ipsi_flat_inclination=require(
                params, 'atlas.exclude_ipsi_flat_inclination'),
            depth_selection=require(params, 'electrodes.depth_selection'),
            rng=rng)
        return cls(atlas, params, voxels, rng=rng, **kwargs)

    @classmethod
    def from_visual_field_targets(cls, atlas, params: Mapping,
                                  eccentricity_deg: Sequence[float],
                                  inclination_deg: Sequence[float],
                                  **kwargs) -> 'LGNElectrodeArray':
        """Put an electrode at the tissue nearest each requested percept
        location.

        Nearest-neighbour in visual-field Cartesian coordinates, so the
        realised position can differ from the requested one where the
        atlas has no tissue there; `position_error_deg` says by how much.
        Restricted to `electrodes.cell_class` when the config names one.
        """
        ecc = np.atleast_1d(np.asarray(eccentricity_deg, dtype=float))
        incl = np.atleast_1d(np.asarray(inclination_deg, dtype=float))
        cell_class = require(params, 'electrodes.cell_class')
        valid = atlas.valid.copy()
        if cell_class is not None:
            valid &= np.isin(atlas.layer,
                             resolve_class_codes(params, cell_class))
        if require(params, 'atlas.exclude_ipsi_flat_inclination'):
            valid &= ~atlas.is_ipsi_sentinel()
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

    # ------------------------------------------------------------------
    # Setup steps
    # ------------------------------------------------------------------
    def _resolve_positions(self):
        """Look up each electrode's retinotopic position, with a small
        neighbourhood fallback where it lands on a placeholder voxel."""
        atlas = self.atlas
        idx = tuple(self.voxel_indices.T)
        ecc = np.asarray(atlas.eccentricity_deg[idx], dtype=float).copy()
        incl = np.asarray(atlas.inclination_deg[idx], dtype=float).copy()
        on_tissue = np.asarray(atlas.valid[idx], dtype=bool).copy()

        self.fell_on_placeholder = ~on_tissue
        n_bad = int(self.fell_on_placeholder.sum())
        if n_bad:
            logging.info("%d of %d electrodes landed on a placeholder voxel; "
                         "falling back to a local neighbourhood average.",
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
        `electrodes.position_fallback_radius_vox` of one index.

        Averaged in Cartesian coordinates, since averaging inclination
        directly would break across the +-180 deg wrap.
        """
        atlas = self.atlas
        radius = require(self.params,
                         'electrodes.position_fallback_radius_vox')
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

        Sized from the EFFECTIVE kernel radius at `max_current_ua`, not
        the nominal Stoney radius, so the smooth kernel's tail is not
        clipped. A stride keeps the candidate count within budget;
        `voxels_per_candidate` records what each candidate stands for so
        absolute cell counts stay correct.
        """
        voxel_size = self.atlas._voxel_size_mm
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
        logging.info("Candidate neighbourhood: %d voxels per electrode "
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

        # Per-layer one-hot, for the scatter-sum that turns per-voxel
        # weights into one activation total per layer touched.
        self.layer_names_by_code = layer_names(self.params)
        self.layer_ids = np.array(sorted(self.layer_names_by_code),
                                  dtype=np.int8)
        self.candidate_layer_onehot = (
            layer[..., None] == self.layer_ids[None, None, :]).astype(
            np.float32)

        classes = np.zeros(layer.shape, dtype=np.int8)
        self.class_codes = {'magno': 1, 'parvo': 2}
        for code, cls in layer_to_class(self.params).items():
            classes[layer == code] = self.class_codes[cls]
        self.candidate_class = classes  # 0 none, 1 magno, 2 parvo
        self.touches_magno = np.any(classes == 1, axis=1)
        self.touches_parvo = np.any(classes == 2, axis=1)

    def _resolve_magnification(self, jacobian_atlas, magnification_model):
        """Local magnification at each electrode centre, plus flags."""
        self.jacobian_atlas = jacobian_atlas
        scalar_fallback = np.zeros(self.n_electrodes, dtype=bool)

        if jacobian_atlas is not None:
            self.magnification_model = JacobianMagnification(
                jacobian_atlas, self.params, self.atlas)
            local = self.magnification_model.at_voxels(self.voxel_indices)
            missing = ~local.valid | ~np.isfinite(local.major_deg_per_mm)
        elif magnification_model is not None:
            self.magnification_model = magnification_model
            local = magnification_model.at_eccentricity(
                self.eccentricity_deg, inclination_deg=self.inclination_deg)
            missing = ~local.valid
            scalar_fallback[:] = True
        else:
            raise ValueError("Provide either jacobian_atlas or "
                             "magnification_model.")

        if magnification_model is not None and jacobian_atlas is not None:
            # Fill Jacobian gaps from the scalar model rather than
            # dropping the electrode entirely.
            fallback = magnification_model.at_eccentricity(
                self.eccentricity_deg, inclination_deg=self.inclination_deg)
            for attr in ('major_deg_per_mm', 'minor_deg_per_mm',
                         'orientation_rad'):
                setattr(local, attr,
                        np.where(missing, getattr(fallback, attr),
                                 getattr(local, attr)))
            local.valid = local.valid | fallback.valid
            scalar_fallback |= missing
        elif missing.any():
            logging.warning("%d electrodes have no usable magnification and "
                            "will render with NaN size.", int(missing.sum()))

        self.local_magnification: LocalMagnification = local
        self.flags = ElectrodeFlags(
            central_isotropic=np.asarray(
                local.isotropic_by_construction, dtype=bool)
            | (self.eccentricity_deg
               < isotropic_construction_radius_deg(self.params)),
            unreliable_beyond_neighborhood=np.asarray(
                local.unreliable_beyond_neighborhood, dtype=bool),
            touches_placeholder=np.any(
                ~self.candidate_valid, axis=1),
            class_heterogeneous=self.touches_magno & self.touches_parvo,
            scalar_magnification=scalar_fallback)

    def _select_scatter_points(self, n_scatter_points: int):
        """Subsample candidate voxels for per-voxel scatter rendering.

        By distance rank with a stride, which keeps the radial profile
        representative instead of over-weighting the much more numerous
        outer shell the way a uniform random draw would.
        """
        n_points = min(n_scatter_points, self.n_candidates)
        order = np.argsort(self.candidate_distance_mm)
        step = max(1, self.n_candidates // n_points)
        chosen = order[::step][:n_points]
        self.scatter_index = np.sort(chosen)
        self.n_scatter_points = len(self.scatter_index)

    def _precompute_scatter_bandwidth(self):
        """Floor for the scatter splat kernel, in degrees.

        Sized from the point cloud's own spacing, never from a
        magnification factor, so no per-voxel-Jacobian assumption sneaks
        back in through the renderer.
        """
        pts = self.candidate_vf_xy[:, self.scatter_index, :]
        valid = self.candidate_valid[:, self.scatter_index]
        bandwidth = np.zeros(self.n_electrodes)
        for i in range(self.n_electrodes):
            # Masked-out candidates sit at the electrode centre;
            # including them would drive the median spacing to zero.
            p = pts[i][valid[i]]
            if len(p) < 2:
                continue
            sub = p if len(p) <= 256 else p[
                np.linspace(0, len(p) - 1, 256).astype(int)]
            d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            nearest = np.min(d, axis=1)
            # The atlas's quantisation makes distinct voxels share a
            # retinotopic position, so drop the zero-spacing pairs.
            nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
            nn_spacing = float(np.median(nearest)) if nearest.size else 0.0

            # Nearest-neighbour spacing alone under-smooths this cloud
            # (a voxel lattice seen through a quantised map clusters
            # along one axis), leaving lattice artefacts. So compare
            # against a uniform cloud of the same count and area.
            extent = float(np.percentile(
                np.linalg.norm(p - self.vf_xy[i], axis=-1),
                require(self.params,
                        'rendering.scatter_extent_percentile')))
            uniform_spacing = (np.sqrt(np.pi * extent ** 2 / len(p))
                               if extent > 0 else 0.0)
            bandwidth[i] = max(nn_spacing, uniform_spacing)
        self.scatter_bandwidth_floor_deg = bandwidth

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------
    @property
    def render_mode(self) -> np.ndarray:
        """Per-electrode renderer choice, as an array of strings.

        'scatter' wherever the single-point Jacobian should not be
        extrapolated across the footprint, 'ellipse' otherwise -- with
        the deliberate exception that electrodes confined to the central
        isotropic-by-construction patch stay 'ellipse', where scatter
        would cost more for the same circle.
        """
        mode = np.where(self.flags.unreliable_beyond_neighborhood,
                        'scatter', 'ellipse')
        mode = np.where(self.flags.central_isotropic, 'ellipse', mode)
        return mode

    def phosphene_sigma_deg(self, current_ua: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Rendered phosphene sigmas (major, minor), degrees.

        Multiplies the recruitment sigma in tissue (mm) by the two
        principal magnifications (deg/mm). An isotropic magnification
        gives equal sigmas, i.e. a circle.
        """
        sigma_mm = np.asarray(self.kernel.sigma_mm(current_ua), dtype=float)
        return (sigma_mm * self.local_magnification.major_deg_per_mm,
                sigma_mm * self.local_magnification.minor_deg_per_mm)

    def distance_histograms(self) -> dict:
        """Per-(electrode, layer) histograms of candidate distance, into
        `recruitment.distance_histogram_bins` bins.

        The recruitment weight depends only on distance and current, so
        binning the distances turns the per-layer activation step from
        O(n_electrodes x n_candidates) per forward pass into
        O(n_electrodes x n_layers x n_bins) -- typically a hundred-fold
        saving. The binning error stays far below the atlas's own
        accuracy floor.

        :return: dict with ``bin_centers_mm`` (n_bins,), ``counts``
            (n_electrodes, n_layers, n_bins), ``cells`` (same shape,
            weighted by cells per voxel and by `voxels_per_candidate`).
        """
        n_bins = require(self.params, 'recruitment.distance_histogram_bins')
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
        electrode -- shared across electrodes, since the offset sphere
        is."""
        return self.candidate_distance_mm[self.scatter_index]

    @property
    def scatter_valid(self) -> np.ndarray:
        """(n_electrodes, n_scatter_points) tissue mask for scatter points."""
        return self.candidate_valid[:, self.scatter_index]

    def scatter_patch_radius_deg(self) -> np.ndarray:
        """Half-width of each electrode's scatter render patch, degrees.

        From the actual spread of the points about the centre plus a
        margin, not from a magnification estimate.
        """
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
                require(self.params,
                        'rendering.scatter_patch_radius_percentile')))
        return radius * require(self.params,
                                'rendering.scatter_patch_radius_margin')

    def per_layer_cell_counts(self) -> np.ndarray:
        """(n_electrodes, n_layers) reachable cells, at maximum current.

        Scaled by `voxels_per_candidate`, so the stride used to keep the
        candidate set affordable does not deflate the count.
        """
        cells = (self.candidate_cells_per_voxel[..., None]
                 * self.candidate_layer_onehot)
        return cells.sum(axis=1) * self.voxels_per_candidate

    def describe(self) -> str:
        hemifield = 'right' if self.hemisphere == 'left' else 'left'
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
