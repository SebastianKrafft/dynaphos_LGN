"""Merged phosphenes, drawn from the recruited tissue itself.

Under field superposition, electrodes whose summed field recruits one
connected volume of tissue are seen as one phosphene. `TissueMerging`
sums every electrode's drive on a lattice of atlas voxels, finds the
connected recruited volumes, and draws each volume that holds more than
one electrode from its own voxels' retinotopic positions. The shape of
a merged phosphene, including tissue recruited only between its
electrodes, therefore comes from the atlas, not from its members'
blobs.

Brightness and detection are calibrated per electrode only, so a merged
phosphene takes its brightest member's brightness x detection
probability. A volume holding one electrode renders exactly as that
electrode's own phosphene.
"""
import logging
import math
from dataclasses import dataclass
from typing import List, Mapping, Optional

import numpy as np
import torch
from scipy import ndimage

from dynaphos.utils import to_tensor
from dynaphos_lgn.aggregation import drive_radius, effective_mode
from dynaphos_lgn.electrodes import LGNElectrodeArray, cloud_bandwidth_deg
from dynaphos_lgn.rendering import gaussian_blur

MERGING = ('merged', 'grown')


def build_merging(array: LGNElectrodeArray, params: Mapping,
                  data_kwargs: dict) -> Optional['TissueMerging']:
    """`TissueMerging` when merged phosphenes are asked for and field
    superposition is what runs, else None."""
    merging = params['aggregation']['phosphene_merging']
    if merging not in MERGING:
        raise ValueError(f"Unknown aggregation.phosphene_merging "
                         f"{merging!r}; expected one of "
                         f"{', '.join(MERGING)}.")
    if (merging == 'grown'
            or effective_mode(params) != 'field_superposition'):
        return None
    return TissueMerging(array, params, data_kwargs)


@dataclass
class RecruitedVolumes:
    """Which electrodes and lattice sites share a recruited volume.

    :ivar labels: (n,) volume per electrode, one array per batch item.
    :ivar site_volume: (n_sites,) volume each lattice site is drawn in,
        or -1, one array per batch item.
    :ivar weight: (batch, n_sites) recruitment at each lattice site.
    """

    labels: List[np.ndarray]
    site_volume: List[np.ndarray]
    weight: torch.Tensor


def _is_tissue(array: LGNElectrodeArray, voxels: np.ndarray) -> np.ndarray:
    """Which voxels can be recruited, by the candidate gather's rules."""
    atlas = array.atlas
    idx = tuple(voxels.T)
    tissue = (np.asarray(atlas.valid[idx], dtype=bool)
              & np.isin(atlas.layer[idx], array.layer_ids))
    if array.exclude_ipsi_flat_inclination:
        tissue &= ~atlas.is_ipsi_sentinel(atlas.inclination[idx])
    return tissue


class TissueMerging:
    """Recruited volumes on a lattice of atlas voxels, and their percept.

    Lattice sites are every `aggregation.merge_lattice_mm` along each
    atlas axis, within an electrode's gather radius. Each site sums
    every electrode's signed drive ``(R / d)^2`` from its exact
    distance. Sites at or above threshold (drive >= 1) are recruited,
    and connected recruited sites (26-neighbor) form one volume. An
    electrode belongs to the volume of its nearest site; an
    unstimulated electrode, or one whose nearest site is not recruited,
    stands alone.

    Connectivity is judged on the field, not on the tissue: the
    interlaminar gaps the atlas leaves empty do not split a volume, just
    as they do not split one electrode's phosphene. Only tissue sites
    are drawn.

    Which volume each electrode is in is a discrete decision and carries
    no gradient. The merged blobs do: every tissue site is drawn in the
    volume of the electrode driving it hardest, weighted by its
    recruitment under the summed field.
    """

    def __init__(self, array: LGNElectrodeArray, params: Mapping,
                 data_kwargs: dict):
        atlas = array.atlas
        voxel_size = atlas.voxel_size_mm
        self.kernel = array.kernel
        self.n_electrodes = array.n_electrodes
        stride = max(1, round(params['aggregation']['merge_lattice_mm']
                              / voxel_size))
        reach = array.max_radius_mm

        # -- sites within reach of each electrode ---------------------
        r = math.ceil(reach / (stride * voxel_size))
        axis = np.arange(-r, r + 1)
        offsets = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'),
                           axis=-1).reshape(-1, 3)
        lattice_shape = (np.asarray(atlas.atlas_shape) - 1) // stride + 1
        nearest = np.all(offsets == 0, axis=1)
        keys, sources, distances = [], [], []
        for e, voxel in enumerate(array.voxel_indices):
            # The nearest site is kept even beyond reach, so every
            # electrode has one; unrecruited, it leaves the electrode alone.
            own = np.clip(np.round(voxel / stride).astype(int), 0,
                          lattice_shape - 1)
            site = own + offsets
            d = np.linalg.norm(site * stride - voxel, axis=1) * voxel_size
            keep = ((d <= reach) | nearest) & np.all(
                (site >= 0) & (site < lattice_shape), axis=1)
            keys.append(np.ravel_multi_index(tuple(site[keep].T),
                                             lattice_shape))
            sources.append(np.full(keep.sum(), e))
            distances.append(d[keep])
        key, source, distance = map(np.concatenate,
                                    (keys, sources, distances))
        unique, entry_site = np.unique(key, return_inverse=True)
        lattice = np.stack(np.unravel_index(unique, lattice_shape), axis=1)
        self.tissue = _is_tissue(array, lattice * stride)
        self.n_sites, self.n_entries = len(lattice), len(entry_site)

        # Nearest site per electrode.
        order = np.lexsort((distance, source))
        first = np.unique(source[order], return_index=True)[1]
        self.own_site = entry_site[order[first]]

        # -- connected components grid ---------------------------------
        low = lattice.min(axis=0)
        self.grid_shape = tuple(lattice.max(axis=0) - low + 1)
        self.grid_index = tuple((lattice - low).T)

        # -- where each site lands in the image (fixed) ----------------
        voxels = tuple((lattice * stride).T)
        ecc = np.asarray(atlas.eccentricity_deg[voxels], dtype=float)
        incl = np.deg2rad(np.asarray(atlas.inclination_deg[voxels],
                                     dtype=float))
        xy = np.stack([ecc * np.cos(incl), ecc * np.sin(incl)], axis=1)
        self._build_splat(xy, params)

        # -- blur bandwidth per electrode (fixed) ----------------------
        # As for scatter rendering: sized to hide the lattice of the
        # tissue it smooths over, never below a pixel floor.
        bounds = np.searchsorted(source, np.arange(self.n_electrodes + 1))
        percentile = params['rendering']['scatter_extent_percentile']
        spacing = np.zeros(self.n_electrodes)
        for e, (a, b) in enumerate(zip(bounds, bounds[1:])):
            sites = entry_site[a:b]
            spacing[e] = cloud_bandwidth_deg(xy[sites[self.tissue[sites]]],
                                             array.vf_xy[e], percentile)
        floor_px = params['rendering']['scatter_bandwidth_floor_pixels']
        self.sigma_px = np.maximum(spacing / self.px_deg, floor_px)
        self.truncation_sigmas = params['rendering'][
            'scatter_kernel_truncation_sigmas']
        self.max_kernel_radius_px = max(1, math.ceil(
            self.truncation_sigmas * float(self.sigma_px.max())))

        long = dict(device=data_kwargs.get('device'), dtype=torch.long)
        self.entry_site = torch.as_tensor(entry_site, **long)
        self.entry_source = torch.as_tensor(source, **long)
        self.entry_distance = torch.clamp(
            to_tensor(distance, **data_kwargs), min=0.5 * voxel_size)
        self.splat_index = torch.as_tensor(self.splat_index, **long)
        self.splat_weight = to_tensor(self.splat_weight, **data_kwargs)
        logging.info("Tissue merging: %d lattice sites every %.3f mm, %d "
                     "(site, electrode) entries.", self.n_sites,
                     stride * voxel_size, self.n_entries)

    def _build_splat(self, xy: np.ndarray, params: Mapping):
        """Bilinear splat of every site into the full image. Sites
        outside it are dropped."""
        res_x, res_y = params['run']['resolution']
        x_org, y_org = params['run']['origin']
        half = params['run']['view_angle'] / 2
        per_px_x = 2 * half / max(res_x - 1, 1)
        per_px_y = 2 * half / max(res_y - 1, 1)
        self.px_deg = min(per_px_x, per_px_y)
        self.resolution = (res_y, res_x)

        px = (xy[:, 0] - (x_org - half)) / per_px_x
        py = (xy[:, 1] - (y_org - half)) / per_px_y
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        fx, fy = px - x0, py - y0
        index, weight = [], []
        for dy in (0, 1):
            for dx in (0, 1):
                xi, yi = x0 + dx, y0 + dy
                inside = (xi >= 0) & (xi < res_x) & (yi >= 0) & (yi < res_y)
                w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                index.append(np.where(inside, yi * res_x + xi, 0))
                weight.append(np.where(inside, w, 0.0))
        self.splat_index = np.stack(index, axis=1)
        self.splat_weight = np.stack(weight, axis=1)

    def volumes(self, amplitude_ua: torch.Tensor) -> RecruitedVolumes:
        """Recruited volumes under the summed field.

        :param amplitude_ua: (..., n) stimulation currents, uA.
        """
        radius = drive_radius(self.kernel, amplitude_ua)
        sign = torch.sign(amplitude_ua)
        term = (sign[..., self.entry_source]
                * (radius[..., self.entry_source] / self.entry_distance) ** 2)
        drive = torch.zeros(*radius.shape[:-1], self.n_sites,
                            dtype=term.dtype, device=term.device).index_add(
            -1, self.entry_site, term)
        weight = self.kernel.weights_from_drive(drive).reshape(
            -1, self.n_sites)

        drive = drive.detach().reshape(-1, self.n_sites)
        term = term.detach().reshape(-1, self.n_entries)
        stimulated = (radius.detach() > 0).reshape(-1, self.n_electrodes)
        labels, site_volume = [], []
        for b in range(len(drive)):
            label = self._label(drive[b], stimulated[b].cpu().numpy())
            labels.append(label)
            site_volume.append(self._site_volume(term[b], label))
        return RecruitedVolumes(labels, site_volume, weight)

    def _label(self, drive: torch.Tensor, stimulated: np.ndarray
               ) -> np.ndarray:
        """(n,) volume per electrode, numbered from 0."""
        recruited = (drive >= 1.0).cpu().numpy()
        grid = np.zeros(self.grid_shape, dtype=bool)
        grid[tuple(i[recruited] for i in self.grid_index)] = True
        component, _ = ndimage.label(grid, structure=np.ones((3, 3, 3)))
        site_component = np.append(component[self.grid_index], 0)
        own = np.where(stimulated, site_component[self.own_site], 0)
        alone = -1 - np.arange(self.n_electrodes)
        return np.unique(np.where(own > 0, own, alone),
                         return_inverse=True)[1]

    def _site_volume(self, term: torch.Tensor, labels: np.ndarray
                     ) -> np.ndarray:
        """(n_sites,) volume each tissue site is drawn in: that of the
        electrode driving it hardest, if that volume holds several
        electrodes."""
        strongest = torch.full((self.n_sites,), -math.inf, dtype=term.dtype,
                               device=term.device).scatter_reduce(
            0, self.entry_site, term, 'amax')
        wins = (term >= strongest[self.entry_site]) & (term > 0)
        driver = np.full(self.n_sites, -1)
        driver[self.entry_site[wins].cpu().numpy()] = \
            self.entry_source[wins].cpu().numpy()
        merged = np.append(np.bincount(labels) > 1, False)
        volume = np.append(labels, -1)[driver]
        return np.where(merged[volume] & self.tissue, volume, -1)

    def percept(self, intensity: torch.Tensor, maps: torch.Tensor,
                volumes: RecruitedVolumes) -> torch.Tensor:
        """Sum of phosphenes, one per recruited volume.

        :param intensity: (batch, n) brightness x detection per
            electrode.
        :param maps: (batch, n, H, W) each electrode's own activation map.
        :return: (batch, H, W).
        """
        frames = []
        for b, labels in enumerate(volumes.labels):
            merged = np.bincount(labels)[labels] > 1
            alone = torch.as_tensor(~merged, dtype=maps.dtype,
                                    device=maps.device)
            weight = intensity[b] * alone
            frame = torch.sum(weight[:, None, None] * maps[b], dim=0)
            if merged.any():
                frame = frame + self._merged_frame(
                    intensity[b], labels, volumes.site_volume[b],
                    volumes.weight[b])
            frames.append(frame)
        return torch.stack(frames)

    def _merged_frame(self, intensity: torch.Tensor, labels: np.ndarray,
                      site_volume: np.ndarray, weight: torch.Tensor
                      ) -> torch.Tensor:
        """The merged volumes' phosphenes: their sites splatted and
        blurred, peak 1, times the brightest member's intensity."""
        size = np.bincount(labels)
        ids = np.flatnonzero(size > 1)
        compact = np.full(len(size) + 1, -1)
        compact[ids] = np.arange(len(ids))
        site_volume = compact[site_volume]
        sites = np.flatnonzero(site_volume >= 0)
        members = np.flatnonzero(compact[labels] >= 0)
        member_volume = compact[labels[members]]

        kwargs = dict(dtype=weight.dtype, device=weight.device)
        long = dict(dtype=torch.long, device=weight.device)
        res_y, res_x = self.resolution
        sites_t = torch.as_tensor(sites, **long)
        index = (torch.as_tensor(site_volume[sites], **long)[:, None]
                 * res_y * res_x + self.splat_index[sites_t])
        values = weight[sites_t, None] * self.splat_weight[sites_t]
        canvas = torch.zeros(len(ids) * res_y * res_x, **kwargs).index_add(
            0, index.reshape(-1), values.reshape(-1))

        sigma = np.zeros(len(ids))
        np.maximum.at(sigma, member_volume, self.sigma_px[members])
        blob = gaussian_blur(canvas.reshape(len(ids), res_y, res_x),
                             to_tensor(sigma, **kwargs),
                             self.truncation_sigmas,
                             self.max_kernel_radius_px)
        blob = blob / torch.clamp(blob.amax(dim=(-2, -1), keepdim=True),
                                  min=1e-12)

        brightest = torch.zeros(len(ids), **kwargs).scatter_reduce(
            0, torch.as_tensor(member_volume, **long),
            intensity[torch.as_tensor(members, **long)], 'amax',
            include_self=False)
        return torch.sum(brightest[:, None, None] * blob, dim=0)
