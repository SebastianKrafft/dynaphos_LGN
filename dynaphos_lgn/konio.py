"""Koniocellular (K) layers, added to a loaded atlas.

The Erwin et al. (1999) atlas labels only the magno and parvo laminae,
but its interlaminar zones carry retinotopy. These zones are labelled
K1-K6 (Kn lies just ventral to principal layer n; Casagrande et al.
2007) and given a uniform K cell density. Everything here is
speculative: no macaque K density or receptive-field data exists.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from dynaphos_lgn.params import optional, require, resolve_class_codes

DV_AXIS = 1     # indices increase dorsally; layer 1 is ventral-most


def konio_enabled(params: Mapping) -> bool:
    return bool(optional(params, 'konio.enabled', False))


def add_konio_layers(atlas, params: Mapping):
    """Label the interlaminar runs as K layers and give them cells."""
    atlas.layer = label_konio_layers(atlas, params)
    _add_konio_cells(atlas, params)
    atlas.update_valid()
    return atlas


def label_konio_layers(atlas, params: Mapping) -> np.ndarray:
    """A copy of `atlas.layer` with the interlaminar runs labelled K1-K6.

    A run is labelled after the principal layer just dorsal to it:
    magno layer n -> Kn; magno or edge below parvo -> K3; between parvo
    layers, counted down from the dorsal surface -> K6, K5, then K4.
    Dorsal-edge runs and runs thicker than `konio.max_run_thickness_vox`
    stay unlabelled.
    """
    layer = np.moveaxis(atlas.layer, DV_AXIS, -1)
    retinotopic = np.moveaxis(_retinotopic(atlas), DV_AXIS, -1)
    magno = resolve_class_codes(params, 'magno')
    parvo = resolve_class_codes(params, 'parvo')
    k_codes = resolve_class_codes(params, 'konio')

    gap = retinotopic & (layer == atlas._missing_layer)
    # The code bounding each run below and above; -1 at the edge of the
    # nucleus.
    boundary = np.where(retinotopic, layer, -1).astype(np.int16)
    below, above, thickness = _run_neighbours(gap, boundary)

    is_parvo = np.isin(layer, parvo)
    top_of_parvo = is_parvo & (layer != _shift_down(layer))
    parvo_above = np.flip(np.cumsum(np.flip(top_of_parvo, -1), -1,
                                    dtype=np.int16), -1)

    k_number = np.zeros(layer.shape, dtype=np.int8)
    for n, code in enumerate(magno, start=1):
        k_number[above == code] = n
    ventral_of_parvo = np.isin(above, parvo)
    k_number[ventral_of_parvo & (np.isin(below, magno) | (below == -1))] = 3
    between_parvo = ventral_of_parvo & np.isin(below, parvo)
    k_number[between_parvo] = np.maximum(
        4, 7 - parvo_above[between_parvo])
    k_number[~gap | (thickness > require(
        params, 'konio.max_run_thickness_vox'))] = 0

    lut = np.array([0] + list(k_codes), dtype=layer.dtype)
    labelled = np.where(k_number > 0, lut[k_number], layer)
    return np.moveaxis(labelled, -1, DV_AXIS)


def _add_konio_cells(atlas, params: Mapping):
    """Spread `konio.fraction_of_total` of all cells uniformly over the
    K voxels."""
    k_voxels = np.isin(atlas.layer, resolve_class_codes(params, 'konio'))
    if not k_voxels.any():
        return
    principal = np.isin(atlas.layer,
                        resolve_class_codes(params, ['magno', 'parvo']))
    fraction = require(params, 'konio.fraction_of_total')
    total = fraction / (1 - fraction) * float(
        atlas.cells_per_voxel[principal].sum())
    stored = np.round(total / k_voxels.sum() * atlas._cells_scale)
    atlas.cells = np.where(k_voxels, stored, atlas.cells).astype(
        atlas.cells.dtype)
    atlas.cells_per_voxel = atlas.cells / np.float32(atlas._cells_scale)


def _retinotopic(atlas) -> np.ndarray:
    missing = atlas._missing_ecc_incl
    return (atlas.eccentricity != missing) & (atlas.inclination != missing)


def _shift_down(a: np.ndarray) -> np.ndarray:
    """a[..., i + 1] at position i; the last position gets -1."""
    out = np.full_like(a, -1)
    out[..., :-1] = a[..., 1:]
    return out


def _run_neighbours(gap: np.ndarray, boundary: np.ndarray):
    """For every voxel of a run of `gap`, the boundary value just below
    and just above the run, and the run's length (last axis)."""
    n = gap.shape[-1]
    index = np.arange(n, dtype=np.int16)     # keeps the temporaries small
    last_below = np.maximum.accumulate(np.where(gap, -1, index), axis=-1)
    first_above = np.flip(np.minimum.accumulate(
        np.flip(np.where(gap, n, index), -1), axis=-1), -1)

    def value_at(position):
        clipped = np.clip(position, 0, n - 1)
        value = np.take_along_axis(boundary, clipped, axis=-1)
        return np.where((position < 0) | (position >= n), -1, value)

    return (value_at(last_below), value_at(first_above),
            first_above - last_below - 1)


def konio_report(atlas, params: Mapping) -> str:
    """K voxels, layer thickness and cells, per K layer."""
    codes = resolve_class_codes(params, 'konio')
    names = require(params, 'atlas.layers.names')
    k_voxels = np.isin(atlas.layer, codes)
    principal = np.isin(atlas.layer,
                        resolve_class_codes(params, ['magno', 'parvo']))
    k_cells = float(atlas.cells_per_voxel[k_voxels].sum())
    all_cells = k_cells + float(atlas.cells_per_voxel[principal].sum())
    thickness_um = _median_thickness_vox(atlas, k_voxels) \
        * atlas._voxel_size_mm * 1e3
    lines = ["Koniocellular layers (speculative)",
             "-" * 68,
             f"  K voxels                          {int(k_voxels.sum()):,}",
             f"  median K-layer thickness          {thickness_um:.0f} um "
             f"(literature ~100 um)",
             f"  K cells                           {k_cells:,.0f} "
             f"({k_cells / max(all_cells, 1) * 100:.1f}% of all cells)"]
    for code in codes:
        lines.append(f"    {names[code]:<10} "
                     f"{int((atlas.layer == code).sum()):>10,} voxels")
    return '\n'.join(lines)


def _median_thickness_vox(atlas, mask: np.ndarray) -> float:
    """Median length of the runs of `mask` along DV."""
    runs = np.moveaxis(mask, DV_AXIS, -1)
    _, _, thickness = _run_neighbours(runs, np.zeros(runs.shape, np.int16))
    starts = runs & ~np.concatenate(
        [np.zeros(runs.shape[:-1] + (1,), bool), runs[..., :-1]], axis=-1)
    return float(np.median(thickness[starts])) if starts.any() else 0.0
