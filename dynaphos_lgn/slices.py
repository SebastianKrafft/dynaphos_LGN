"""Coronal, sagittal and horizontal slices through the LGN atlas.

The companion to `dynaphos_lgn.viz3d`: the same volume, the same
overlays, the same colours and the same electrodes, cut into planes
instead of rendered as a surface. Slices are how the laminae, the
eccentricity gradient and an electrode's depth actually become
legible -- a surface can only ever show you its outside.

```
python -m dynaphos_lgn.slices                          # ortho triple
python -m dynaphos_lgn.slices --view slices --plane coronal --count 12
python -m dynaphos_lgn.slices --view combined          # slice inside the nucleus
python -m dynaphos_lgn.slices --overlay layer --electrodes layout
python -m dynaphos_lgn.slices --at 11,2.0,7.5 --backend matplotlib
```

Three views:

- **ortho** (the default): one coronal, one sagittal and one horizontal
  panel through a single point, with a crosshair on each showing where
  the other two cut. The standard way to look at a volume.
- **slices**: one plane, stepped through the nucleus. matplotlib draws
  a montage; plotly gives a slider.
- **combined**: the nucleus as a see-through shell, the active slice
  flat beside it, and that same slice drawn where it actually cuts,
  inside the shell. One slider moves all three.

**A slice is one voxel thick.** 25 µm, not an average through a slab,
and never interpolated: one image pixel is one atlas voxel or it is
nothing. Three things are therefore distinguishable on every panel,
and are meant to be:

- **background** -- no atlas voxel there at all, or one the selection
  dropped;
- **grey** -- tissue the atlas has no value for, in the band reserved
  below the colour scale, exactly as in the 3-D view;
- **colour** -- a value the atlas actually holds.

Colour limits are fixed once from a sample of the whole selected
volume rather than per slice, so a colour means the same thing on
every panel and does not shift when you move the slider.

Geometry follows `viz3d`: the scene's x axis is signed ML with the left
nucleus at negative x, so a coronal panel shows both nuclei in the
neurological convention and a sagittal position picks a nucleus by its
sign. Orientation letters on each panel edge say which way is which.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dynaphos_lgn.params import require
from dynaphos_lgn.viz3d import (JACOBIAN_OVERLAYS, OVERLAY_BUILDERS,
                                AtlasContext, ElectrodeSet, Overlay,
                                add_atlas_arguments, add_electrode_arguments,
                                add_output_arguments, add_tissue_arguments,
                                atlas_banner, axis_mm, category_colour,
                                collect_electrodes, colormap, colour_spec,
                                compute_overlay, continuous_limits,
                                group_colours, hemisphere_sign,
                                jacobian_provider, list_overlays,
                                tissue_mask, _apply_overrides)


@dataclass(frozen=True)
class Plane:
    """One of the three cardinal cuts.

    Axes are given as indices into SCENE coordinates -- 0 is signed ML
    (left nucleus negative), 1 is DV, 2 is AP -- not into the atlas's
    own (ML, DV, AP) array, because a coronal panel spans both nuclei
    and the atlas's ML axis does not.

    :ivar normal: The scene axis held fixed.
    :ivar horizontal: The scene axis along the image's x.
    :ivar vertical: The scene axis along the image's y, drawn ascending
        upwards.
    :ivar letters: (left, right, bottom, top) orientation letters.
    """

    name: str
    normal: int
    horizontal: int
    vertical: int
    normal_label: str
    horizontal_label: str
    vertical_label: str
    letters: Tuple[str, str, str, str]
    description: str


#: The three cardinal planes. Not in the config, unlike every number in
#: this package: which axis a coronal cut holds fixed is anatomy's
#: definition, not a value that could be refined.
PLANES: Dict[str, Plane] = {
    'coronal': Plane(
        name='coronal', normal=2, horizontal=0, vertical=1,
        normal_label='AP', horizontal_label='ML (mm, signed)',
        vertical_label='DV (mm)', letters=('L', 'R', 'V', 'D'),
        description='fixed AP, seen from the front'),
    'sagittal': Plane(
        name='sagittal', normal=0, horizontal=2, vertical=1,
        normal_label='ML', horizontal_label='AP (mm)',
        vertical_label='DV (mm)', letters=('P', 'A', 'V', 'D'),
        description='fixed ML, seen from the side'),
    'horizontal': Plane(
        name='horizontal', normal=1, horizontal=0, vertical=2,
        normal_label='DV', horizontal_label='ML (mm, signed)',
        vertical_label='AP (mm)', letters=('L', 'R', 'P', 'A'),
        description='fixed DV, seen from above'),
}


# ======================================================================
# The scene grid
# ======================================================================
@dataclass
class SceneGrid:
    """A regular grid in scene millimetres, and how it maps back into
    each nucleus's own voxel array.

    The two nuclei share one ML ladder, reflected: the left occupies
    negative x and the right positive, with the gap between them real
    and to scale. Every scene cell therefore belongs to at most one
    nucleus, which is what lets a sagittal position pick a nucleus by
    its sign and a coronal panel hold both.

    Nothing is resampled. A scene column IS an atlas ML index; the
    grid only records which nucleus's index it is.

    :ivar x_mm: Signed ML coordinate of each column, ascending.
    :ivar y_mm: DV coordinate of each row, ascending.
    :ivar z_mm: AP coordinate, ascending.
    :ivar owner: (nx,) index into `hemispheres` per column, -1 for the
        gap between the nuclei.
    :ivar ml_index: (nx,) that nucleus's own ML index, -1 in the gap.
    :ivar dv_offset: Atlas DV index of row 0.
    :ivar ap_offset: Atlas AP index of the first AP coordinate.
    :ivar voxel_size_mm: Grid spacing, the atlas's own.
    """

    x_mm: np.ndarray
    y_mm: np.ndarray
    z_mm: np.ndarray
    owner: np.ndarray
    ml_index: np.ndarray
    dv_offset: int
    ap_offset: int
    voxel_size_mm: float
    hemispheres: List[str] = field(default_factory=list)

    def coordinate(self, axis: int) -> np.ndarray:
        """The 1-D coordinate array along one scene axis."""
        return (self.x_mm, self.y_mm, self.z_mm)[axis]

    def nearest_index(self, axis: int, position_mm: float) -> int:
        """The index whose coordinate is closest to `position_mm`."""
        coordinate = self.coordinate(axis)
        return int(np.argmin(np.abs(coordinate - float(position_mm))))

    def hemisphere_of_column(self, column: int) -> Optional[str]:
        """Which nucleus a scene column belongs to, if any."""
        owner = int(self.owner[column])
        return None if owner < 0 else self.hemispheres[owner]


def build_scene_grid(params: Mapping, atlases: Mapping,
                     masks: Mapping) -> SceneGrid:
    """Lay the nuclei out on one signed-ML grid and crop it to the
    tissue.

    :param atlases: {hemisphere: built atlas}.
    :param masks: {hemisphere: boolean volume of the tissue to show}.
    :raises ValueError: if the two nuclei would land on the same
        column, which would mean the atlas origin puts them across the
        midline from each other rather than either side of it.
    """
    names = list(atlases)
    first = atlases[names[0]]
    voxel = float(first._voxel_size_mm)

    signed = {name: axis_mm(atlases[name], 0) * hemisphere_sign(params, name)
              for name in names}
    low = min(float(values.min()) for values in signed.values())
    high = max(float(values.max()) for values in signed.values())
    n_columns = int(round((high - low) / voxel)) + 1

    owner = np.full(n_columns, -1, dtype=int)
    ml_index = np.full(n_columns, -1, dtype=int)
    for position, name in enumerate(names):
        exact = (signed[name] - low) / voxel
        column = np.round(exact).astype(int)
        if np.max(np.abs(exact - column)) > 0.01:
            raise ValueError(
                "The two nuclei do not share a regular grid: twice the "
                "atlas's ML origin is not a whole number of voxels, so a "
                "signed ML axis cannot hold both without resampling. "
                "Draw one hemisphere at a time.")
        if np.any(owner[column] >= 0):
            raise ValueError(
                "The left and right nuclei overlap on the signed ML axis. "
                "That needs atlas.horsley_clarke_origin_mm.ml to be zero, "
                "which would put both nuclei on the midline.")
        owner[column] = position
        ml_index[column] = np.arange(len(column))

    # --- crop to the tissue, so an empty half is not drawn ------------
    occupied = np.zeros(n_columns, dtype=bool)
    dv_any = np.zeros(first.atlas_shape[1], dtype=bool)
    ap_any = np.zeros(first.atlas_shape[2], dtype=bool)
    for position, name in enumerate(names):
        mask = masks[name]
        here = owner == position
        occupied[here] |= mask.any(axis=(1, 2))[ml_index[here]]
        dv_any |= mask.any(axis=(0, 2))
        ap_any |= mask.any(axis=(0, 1))

    if not occupied.any():
        raise ValueError("The selection leaves no tissue to slice.")

    columns = np.flatnonzero(occupied)
    rows = np.flatnonzero(dv_any)
    depths = np.flatnonzero(ap_any)
    column_slice = slice(columns[0], columns[-1] + 1)
    row_slice = slice(rows[0], rows[-1] + 1)
    depth_slice = slice(depths[0], depths[-1] + 1)

    x_mm = low + np.arange(n_columns) * voxel
    return SceneGrid(
        x_mm=x_mm[column_slice],
        y_mm=axis_mm(first, 1)[row_slice],
        z_mm=axis_mm(first, 2)[depth_slice],
        owner=owner[column_slice],
        ml_index=ml_index[column_slice],
        dv_offset=int(rows[0]),
        ap_offset=int(depths[0]),
        voxel_size_mm=voxel,
        hemispheres=names)


# ======================================================================
# Cutting one plane
# ======================================================================
@dataclass
class Slice:
    """One plane, with every requested overlay evaluated on it.

    :ivar plane: Which cut this is.
    :ivar position_mm: Where along the plane's normal, in scene
        millimetres.
    :ivar index: The scene index that position corresponds to.
    :ivar tissue: (ny, nx) True where the selection has a voxel. False
        is background -- no voxel at all -- which is a different thing
        from a voxel with no value.
    :ivar values: {overlay name: (ny, nx) values, NaN where the atlas
        has none}.
    :ivar extent_mm: (left, right, bottom, top) for `imshow`.
    :ivar n_voxels: How many tissue voxels the plane holds.
    """

    plane: Plane
    position_mm: float
    index: int
    tissue: np.ndarray
    values: Dict[str, np.ndarray]
    extent_mm: Tuple[float, float, float, float]
    n_voxels: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.tissue.shape


def _plane_shape(grid: SceneGrid, plane: Plane) -> Tuple[int, int]:
    return (len(grid.coordinate(plane.vertical)),
            len(grid.coordinate(plane.horizontal)))


def _extent(grid: SceneGrid, plane: Plane
            ) -> Tuple[float, float, float, float]:
    half = grid.voxel_size_mm / 2
    horizontal = grid.coordinate(plane.horizontal)
    vertical = grid.coordinate(plane.vertical)
    return (float(horizontal[0]) - half, float(horizontal[-1]) + half,
            float(vertical[0]) - half, float(vertical[-1]) + half)


def _gather(grid: SceneGrid, masks: Mapping, plane: Plane, index: int):
    """Which atlas voxels a plane cuts, and where they land in the image.

    :return: A list of ``(hemisphere, voxels, rows, columns)``, one
        entry per nucleus the plane touches. `voxels` is an (n, 3)
        array of that nucleus's own (ml, dv, ap) indices; `rows` and
        `columns` say where each one belongs in the image.
    """
    found = []
    dv0, ap0 = grid.dv_offset, grid.ap_offset
    n_rows, n_columns = _plane_shape(grid, plane)

    if plane.name == 'sagittal':
        # One column of the scene grid, so exactly one nucleus -- which
        # is how the sign of an ML position picks a side.
        name = grid.hemisphere_of_column(index)
        if name is None:
            return found
        ml = int(grid.ml_index[index])
        patch = masks[name][ml, dv0:dv0 + n_rows, ap0:ap0 + n_columns]
        rows, columns = np.nonzero(patch)
        voxels = np.stack([np.full(len(rows), ml), rows + dv0,
                           columns + ap0], axis=1)
        return [(name, voxels, rows, columns)]

    for position, name in enumerate(grid.hemispheres):
        scene_columns = np.flatnonzero(grid.owner == position)
        if not len(scene_columns):
            continue
        ml = grid.ml_index[scene_columns]

        if plane.name == 'coronal':
            ap = ap0 + index
            patch = masks[name][ml][:, dv0:dv0 + n_rows, ap]
            along, rows = np.nonzero(patch)
            voxels = np.stack([ml[along], rows + dv0,
                               np.full(len(rows), ap)], axis=1)
        else:                                          # horizontal
            dv = dv0 + index
            patch = masks[name][ml][:, dv, ap0:ap0 + n_rows]
            along, rows = np.nonzero(patch)
            voxels = np.stack([ml[along], np.full(len(rows), dv),
                               rows + ap0], axis=1)

        if len(rows):
            found.append((name, voxels, rows, scene_columns[along]))
    return found


def cut(grid: SceneGrid, masks: Mapping, contexts: Mapping,
        overlay_names: Sequence[str], plane: Plane, index: int) -> Slice:
    """Evaluate every overlay on one plane.

    The voxels are gathered once and each overlay is evaluated on that
    same list, so asking for six overlays costs one gather and six
    lookups rather than six passes over the volume.

    :param contexts: {hemisphere: `AtlasContext`}.
    :param index: Position along the plane's normal, as a scene index.
    """
    n_rows, n_columns = _plane_shape(grid, plane)
    tissue = np.zeros((n_rows, n_columns), dtype=bool)
    values = {name: np.full((n_rows, n_columns), np.nan)
              for name in overlay_names}

    for hemisphere, voxels, rows, columns in _gather(grid, masks, plane,
                                                     index):
        tissue[rows, columns] = True
        for name in overlay_names:
            overlay = compute_overlay(name, contexts[hemisphere], voxels)
            values[name][rows, columns] = overlay.values

    return Slice(plane=plane,
                 position_mm=float(grid.coordinate(plane.normal)[index]),
                 index=index, tissue=tissue, values=values,
                 extent_mm=_extent(grid, plane),
                 n_voxels=int(tissue.sum()))


def plane_voxel_counts(grid: SceneGrid, masks: Mapping, plane: Plane
                       ) -> np.ndarray:
    """How many tissue voxels each plane along a normal would hold."""
    n = len(grid.coordinate(plane.normal))
    counts = np.zeros(n, dtype=np.int64)

    for position, name in enumerate(grid.hemispheres):
        mask = masks[name]
        if plane.name == 'coronal':
            counts += mask.sum(axis=(0, 1))[
                grid.ap_offset:grid.ap_offset + n]
        elif plane.name == 'horizontal':
            counts += mask.sum(axis=(0, 2))[
                grid.dv_offset:grid.dv_offset + n]
        else:                                          # sagittal
            columns = np.flatnonzero(grid.owner == position)
            counts[columns] += mask.sum(axis=(1, 2))[grid.ml_index[columns]]
    return counts


def occupied_range(grid: SceneGrid, masks: Mapping, plane: Plane,
                   min_fraction: float = 0.0) -> Tuple[int, int]:
    """The span of planes worth spreading automatic slices over.

    The nucleus tapers to a handful of voxels at each end, and a
    montage that starts and finishes on those spends two panels on
    almost nothing. `min_fraction` trims them relative to the fullest
    plane, so the rule scales with whatever was selected instead of
    being a voxel count someone has to keep in step.

    This only moves where automatic slices are PLACED. It never
    changes what a slice shows, and an explicit range overrides it.

    :param min_fraction: Keep planes holding at least this share of the
        fullest plane's voxels. 0 keeps every plane with anything at
        all in it.
    """
    counts = plane_voxel_counts(grid, masks, plane)
    if not counts.any():
        raise ValueError(f"No tissue in any {plane.name} plane.")
    threshold = max(float(min_fraction) * float(counts.max()), 1.0)
    found = np.flatnonzero(counts >= threshold)
    if not len(found):
        found = np.flatnonzero(counts)
    return int(found[0]), int(found[-1])


def slice_indices(grid: SceneGrid, masks: Mapping, plane: Plane,
                  count: int,
                  range_mm: Optional[Sequence[float]] = None,
                  min_fraction: float = 0.0) -> List[int]:
    """Evenly spaced positions across the tissue, as scene indices.

    Spread over the range that actually holds tissue rather than over
    the grid, and deduplicated, so asking for more slices than the
    nucleus is thick gives you the nucleus rather than repeats of its
    end planes.

    :param count: How many slices to place.
    :param range_mm: ``(low, high)`` along the normal to restrict to.
    :param min_fraction: Passed to `occupied_range`, which trims the
        near-empty planes at each end of the nucleus.
    """
    low, high = occupied_range(grid, masks, plane, min_fraction)
    if range_mm is not None:
        coordinate = grid.coordinate(plane.normal)
        half = grid.voxel_size_mm / 2
        wanted = (min(range_mm), max(range_mm))
        # Checked against the grid's own span before snapping: without
        # this a range nowhere near the nucleus clamps silently to its
        # nearest edge and returns a slice, which looks like an answer.
        if wanted[1] < coordinate[0] - half or wanted[0] > coordinate[-1] + half:
            raise ValueError(
                f"The requested {plane.name} range {wanted[0]} to "
                f"{wanted[1]} mm lies outside the drawn volume, which "
                f"spans {plane.normal_label} {coordinate[0]:+.3f} to "
                f"{coordinate[-1]:+.3f} mm.")
        requested = [grid.nearest_index(plane.normal, value)
                     for value in wanted]
        low = max(low, min(requested))
        high = min(high, max(requested))
        if low > high:
            raise ValueError(
                f"The requested {plane.name} range {wanted[0]} to "
                f"{wanted[1]} mm holds no tissue.")
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}.")
    if count == 1:
        return [(low + high) // 2]
    chosen = np.unique(np.round(np.linspace(low, high, count)).astype(int))
    return [int(value) for value in chosen]


# ======================================================================
# Colour, fixed across every slice
# ======================================================================
def sample_limits(params: Mapping, masks: Mapping, contexts: Mapping,
                  overlay_names: Sequence[str], rng: np.random.Generator
                  ) -> Dict[str, Optional[Tuple[float, float]]]:
    """Colour limits from the whole selected volume, not from one plane.

    A slice viewer whose scale is set per slice makes every slice look
    the same and none of them comparable. These limits are computed
    once, from a random sample of the tissue across both nuclei, and
    then used for every plane and every position -- so a colour means
    the same number wherever you see it, and moving the slider does not
    move the scale.

    :return: {overlay name: (low, high)}, or None for a categorical
        overlay, whose categories are fixed by the config anyway.
    """
    budget = int(require(params, 'visualization.slices.colour_limit_sample'))
    sampled: Dict[str, List[np.ndarray]] = {name: [] for name in overlay_names}
    kinds: Dict[str, str] = {}
    labels: Dict[str, str] = {}

    for hemisphere, mask in masks.items():
        voxels = np.argwhere(mask)
        if not len(voxels):
            continue
        share = max(int(budget / max(len(masks), 1)), 1)
        if len(voxels) > share:
            voxels = voxels[rng.choice(len(voxels), share, replace=False)]
        for name in overlay_names:
            overlay = compute_overlay(name, contexts[hemisphere], voxels)
            kinds[name] = overlay.kind
            labels[name] = overlay.label
            sampled[name].append(overlay.values)

    limits: Dict[str, Optional[Tuple[float, float]]] = {}
    for name in overlay_names:
        if kinds.get(name) == 'categorical' or not sampled[name]:
            limits[name] = None
            continue
        pooled = Overlay(name=name, label=labels[name],
                         values=np.concatenate(sampled[name]),
                         kind='continuous')
        limits[name] = continuous_limits(pooled, params)
    return limits


def slice_colours(overlay: Overlay, image: np.ndarray, tissue: np.ndarray,
                  params: Mapping,
                  limits: Optional[Tuple[float, float]] = None):
    """RGBA for one slice, and the `ColourSpec` behind it.

    Three states, kept apart on purpose: background where there is no
    voxel, the missing-value grey where there is a voxel but no number,
    and the colour scale where there is one.
    """
    import matplotlib.colors as mcolors

    flat = Overlay(name=overlay.name, label=overlay.label,
                   values=image.ravel(), kind=overlay.kind,
                   categories=overlay.categories, note=overlay.note,
                   unit=overlay.unit)
    spec = colour_spec(flat, params, limits=limits)
    rgba = spec.rgba.reshape(image.shape + (4,)).copy()
    background = np.array(mcolors.to_rgba(
        require(params, 'visualization.slices.background')))
    rgba[~tissue] = background
    return rgba, spec


def heatmap_values(image: np.ndarray, tissue: np.ndarray,
                   spec) -> np.ndarray:
    """A slice's values prepared for a plotly heatmap.

    Tissue with no value is moved into the reserved grey band, as in
    the 3-D view; background stays NaN, which plotly draws as
    transparent so the panel's own background shows through. The two
    therefore stay distinguishable without a second trace.
    """
    prepared = np.where(tissue, spec.values.reshape(image.shape), np.nan)
    return prepared


# ======================================================================
# Electrodes on a plane
# ======================================================================
@dataclass
class ContactsOnPlane:
    """The contacts near enough to one plane to be drawn on it.

    :ivar horizontal_mm: In-plane x of each contact.
    :ivar vertical_mm: In-plane y.
    :ivar distance_mm: How far off the plane each one actually is.
    :ivar rows: Index back into the full `ElectrodeSet`.
    """

    horizontal_mm: np.ndarray
    vertical_mm: np.ndarray
    distance_mm: np.ndarray
    rows: np.ndarray

    def __len__(self) -> int:
        return len(self.rows)


def contacts_near(electrodes: Optional[ElectrodeSet], plane: Plane,
                  position_mm: float, slab_mm: float) -> ContactsOnPlane:
    """Contacts within `slab_mm` of a plane.

    A slice is 25 µm thick and a contact is a point, so without a slab
    a contact would essentially never land exactly on a drawn plane.
    The slab is a drawing tolerance, not a claim about the electrode:
    how far off each one really is travels with it, into the hover and
    into the marker's transparency.
    """
    empty = ContactsOnPlane(np.zeros(0), np.zeros(0), np.zeros(0),
                            np.zeros(0, dtype=int))
    if electrodes is None or not len(electrodes):
        return empty

    distance = np.abs(electrodes.positions_mm[:, plane.normal]
                      - float(position_mm))
    rows = np.flatnonzero(distance <= slab_mm)
    if not len(rows):
        return empty
    return ContactsOnPlane(
        horizontal_mm=electrodes.positions_mm[rows, plane.horizontal],
        vertical_mm=electrodes.positions_mm[rows, plane.vertical],
        distance_mm=distance[rows], rows=rows)


# ======================================================================
# Assembling a figure's worth of slices
# ======================================================================
@dataclass
class SliceFigure:
    """Everything a backend needs to draw a set of slices.

    :ivar slices: The planes to draw, in order.
    :ivar overlays: {name: a representative `Overlay`}, for labels,
        categories and the honesty note. The numbers live in each
        `Slice`.
    :ivar limits: {name: colour-axis ends}, shared by every slice.
    :ivar crosshair: Scene position the panels cut through, or None.
    """

    slices: List[Slice]
    overlays: Dict[str, Overlay]
    order: List[str]
    limits: Dict[str, Optional[Tuple[float, float]]]
    electrodes: Optional[ElectrodeSet]
    grid: SceneGrid
    params: Mapping
    banner: str
    selection: str
    crosshair: Optional[np.ndarray] = None
    shell: Optional[List] = None

    def describe(self) -> str:
        lines = [f"LGN slices: {self.banner}",
                 f"  tissue drawn      : {self.selection}",
                 f"  scene grid        : {len(self.grid.x_mm)} x "
                 f"{len(self.grid.y_mm)} x {len(self.grid.z_mm)} voxels of "
                 f"{self.grid.voxel_size_mm * 1e3:.0f} um "
                 f"({', '.join(self.grid.hemispheres)} LGN)"]
        if self.crosshair is not None:
            lines.append(
                f"  crosshair         : ML {self.crosshair[0]:+.2f}, "
                f"DV {self.crosshair[1]:+.2f}, AP {self.crosshair[2]:+.2f} mm")
        for one in self.slices:
            lines.append(
                f"  {one.plane.name:<10} {one.plane.normal_label} = "
                f"{one.position_mm:+.3f} mm : {one.n_voxels:,} voxels "
                f"of {one.shape[0]} x {one.shape[1]}")
        lines.append(f"  overlays          : {', '.join(self.order)}")
        for name in self.order:
            low_high = self.limits[name]
            scale = ('categorical' if low_high is None
                     else f'{low_high[0]:.3g} to {low_high[1]:.3g}')
            lines.append(f"    {name:<26} {scale}")
        if self.shell:
            faces = sum(mesh.n_faces for mesh in self.shell)
            lines.append(f"  shell             : {faces:,} faces at stride "
                         f"{self.shell[0].stride}, around the slices")
        if self.electrodes is not None:
            lines.append(f"  electrodes        : {len(self.electrodes)} "
                         f"({self.electrodes.source})")
        return '\n'.join(lines)


def resolve_position(grid: SceneGrid, masks: Mapping, params: Mapping,
                     at: Optional[str],
                     electrodes: Optional[ElectrodeSet]) -> np.ndarray:
    """Where the ortho panels should cut, in scene millimetres.

    :param at: ``ML,DV,AP`` in scene coordinates (signed ML, left
        nucleus negative); ``electrode:N`` for a contact's own
        position; or None for the centroid of the drawn tissue.
    """
    if at is None:
        return tissue_centroid(grid, masks)

    if at.startswith('electrode:'):
        if electrodes is None or not len(electrodes):
            raise ValueError(
                f"--at {at!r} names a contact, but no electrodes were "
                f"asked for. Add --electrodes, --probe, --grid or "
                f"--electrodes-file.")
        which = int(at.split(':', 1)[1])
        if not -len(electrodes) <= which < len(electrodes):
            raise ValueError(
                f"--at {at!r}: there are {len(electrodes)} contacts, "
                f"numbered 0 to {len(electrodes) - 1}.")
        return np.asarray(electrodes.positions_mm[which], dtype=float)

    parts = at.split(',')
    if len(parts) != 3:
        raise ValueError(
            f"--at wants ML,DV,AP in millimetres, 'electrode:N', or "
            f"nothing at all for the centroid; got {at!r}.")
    position = np.array([float(value) for value in parts])

    # ML is a SIGNED scene coordinate here, because that is what picks
    # the nucleus. Someone reading millimetres off the atlas will type
    # the positive lateral distance it quotes, so if the sign they gave
    # names no tissue and its mirror does, take the mirror and say so.
    if column_at(grid, position[0]) is None:
        mirrored = column_at(grid, -position[0])
        if mirrored is not None:
            logging.warning(
                "ML %+.2f mm names no drawn tissue; using %+.2f mm, in the "
                "%s nucleus. --at takes SIGNED ML, with the left nucleus at "
                "negative x.", position[0], -position[0],
                grid.hemisphere_of_column(mirrored))
            position[0] = -position[0]
    return position


def column_at(grid: SceneGrid, x_mm: float) -> Optional[int]:
    """The scene column a signed ML position falls in, or None.

    None both outside the grid and in the gap between the nuclei --
    `nearest_index` alone cannot tell those from a hit, because it
    clamps to the nearest edge whatever it is given.
    """
    coordinate = grid.x_mm
    half = grid.voxel_size_mm / 2
    if x_mm < coordinate[0] - half or x_mm > coordinate[-1] + half:
        return None
    column = grid.nearest_index(0, x_mm)
    return None if grid.owner[column] < 0 else column


def tissue_centroid(grid: SceneGrid, masks: Mapping) -> np.ndarray:
    """The centre of mass of the drawn tissue, in scene millimetres.

    A sensible default place to cut: it is inside the nucleus, which a
    bounding-box centre need not be.
    """
    weight = 0.0
    total = np.zeros(3)
    for position, name in enumerate(grid.hemispheres):
        mask = masks[name]
        columns = np.flatnonzero(grid.owner == position)
        if not len(columns):
            continue
        per_ml = mask.sum(axis=(1, 2))[grid.ml_index[columns]]
        per_dv = mask.sum(axis=(0, 2))
        per_ap = mask.sum(axis=(0, 1))
        n = float(mask.sum())
        if n == 0:
            continue
        total += n * np.array([
            float(np.average(grid.x_mm[columns], weights=per_ml))
            if per_ml.sum() else 0.0,
            float(np.average(axis_mm_range(grid, 1), weights=_crop(
                per_dv, grid.dv_offset, len(grid.y_mm)))),
            float(np.average(axis_mm_range(grid, 2), weights=_crop(
                per_ap, grid.ap_offset, len(grid.z_mm))))])
        weight += n
    if weight == 0:
        raise ValueError("The selection leaves no tissue to slice.")
    return total / weight


def _crop(counts: np.ndarray, offset: int, length: int) -> np.ndarray:
    return counts[offset:offset + length]


def axis_mm_range(grid: SceneGrid, axis: int) -> np.ndarray:
    return grid.coordinate(axis)


def build_slice_figure(params: Mapping, atlases: Mapping, masks: Mapping,
                       contexts: Mapping, overlay_names: Sequence[str],
                       planes: Sequence[Tuple[Plane, int]],
                       electrodes: Optional[ElectrodeSet],
                       banner: str, selection: str,
                       crosshair: Optional[np.ndarray] = None,
                       grid: Optional[SceneGrid] = None,
                       shell: Optional[List] = None) -> SliceFigure:
    """Cut every requested plane and fix the colour scales once.

    :param shell: Nucleus surfaces from `build_shell`, for the combined
        view. The other views do not need them and do not pay for them.
    """
    if grid is None:
        grid = build_scene_grid(params, atlases, masks)
    rng = np.random.default_rng(require(params, 'run.seed'))

    limits = sample_limits(params, masks, contexts, overlay_names, rng)
    cuts = [cut(grid, masks, contexts, overlay_names, plane, index)
            for plane, index in planes]

    # One representative overlay per name, for its label, categories and
    # note. Taken from a plane that actually has tissue, so a
    # categorical overlay's category list is populated.
    representative: Dict[str, Overlay] = {}
    for name in overlay_names:
        sample = np.argwhere(next(iter(masks.values())))[:1]
        context = contexts[next(iter(masks))]
        representative[name] = compute_overlay(name, context, sample)

    return SliceFigure(slices=cuts, overlays=representative,
                       order=list(overlay_names), limits=limits,
                       electrodes=electrodes, grid=grid, params=params,
                       banner=banner, selection=selection,
                       crosshair=crosshair, shell=shell)


# ======================================================================
# matplotlib
# ======================================================================
def _draw_panel(axes, figure: SliceFigure, one: Slice, name: str,
                params: Mapping):
    """One slice, its electrodes, its crosshair and its letters."""
    overlay = figure.overlays[name]
    rgba, _ = slice_colours(overlay, one.values[name], one.tissue, params,
                            figure.limits[name])
    axes.imshow(rgba, origin='lower', extent=one.extent_mm,
                interpolation='nearest', aspect='equal')

    _draw_crosshair(axes, figure, one, params)
    _draw_contacts(axes, figure, one, params)
    _draw_letters(axes, one.plane, params)

    axes.set_xlabel(one.plane.horizontal_label, fontsize=7)
    axes.set_ylabel(one.plane.vertical_label, fontsize=7)
    axes.tick_params(labelsize=6)
    axes.set_title(f'{one.plane.name}  {one.plane.normal_label} = '
                   f'{one.position_mm:+.2f} mm', fontsize=8)
    axes.set_facecolor(require(params, 'visualization.slices.background'))


def _draw_crosshair(axes, figure: SliceFigure, one: Slice, params: Mapping):
    if figure.crosshair is None or not require(
            params, 'visualization.slices.crosshair'):
        return
    colour = require(params, 'visualization.slices.crosshair_colour')
    axes.axvline(figure.crosshair[one.plane.horizontal], color=colour,
                 linewidth=0.6, alpha=0.7)
    axes.axhline(figure.crosshair[one.plane.vertical], color=colour,
                 linewidth=0.6, alpha=0.7)


def _draw_contacts(axes, figure: SliceFigure, one: Slice, params: Mapping):
    electrodes = figure.electrodes
    if electrodes is None or not len(electrodes):
        return
    slab = float(require(params, 'visualization.slices.electrode_slab_mm'))
    near = contacts_near(electrodes, one.plane, one.position_mm, slab)
    if not len(near):
        return

    colours = group_colours(params, len(electrodes.group_names))
    size = float(require(params, 'visualization.electrodes.marker_size')) ** 2
    # Faded with distance from the plane, so a contact drawn on a slice
    # it is merely near does not read as one sitting in it.
    alpha = 1.0 - 0.6 * (near.distance_mm / max(slab, 1e-9))
    for row, x, y, a in zip(near.rows, near.horizontal_mm, near.vertical_mm,
                            alpha):
        outside = bool(electrodes.outside[row])
        colour = colours[int(electrodes.group[row])]
        axes.scatter(x, y, s=size, marker='D',
                     facecolors='none' if outside else colour,
                     edgecolors=colour if outside else 'white',
                     linewidths=1.0 if outside else 0.5,
                     alpha=float(np.clip(a, 0.25, 1.0)), zorder=5)

    label_every = int(require(params, 'visualization.electrodes.label_every'))
    if label_every > 0:
        for row, x, y in zip(near.rows, near.horizontal_mm,
                             near.vertical_mm):
            if row % label_every == 0:
                axes.annotate(electrodes.labels[row], (x, y),
                              textcoords='offset points', xytext=(3, 3),
                              fontsize=5, color='white')


def _draw_letters(axes, plane: Plane, params: Mapping):
    if not require(params, 'visualization.slices.orientation_letters'):
        return
    left, right, bottom, top = plane.letters
    style = dict(transform=axes.transAxes, fontsize=7, color='0.6',
                 ha='center', va='center')
    axes.text(0.025, 0.5, left, **style)
    axes.text(0.975, 0.5, right, **style)
    axes.text(0.5, 0.025, bottom, **style)
    axes.text(0.5, 0.975, top, **style)


def _colourbar(figure_handle, axes, figure: SliceFigure, name: str,
               params: Mapping):
    """A colourbar for a continuous overlay, a patch legend for a
    categorical one. Shared by every panel, because the scale is."""
    import matplotlib.colors as mcolors
    from matplotlib import cm
    from matplotlib.patches import Patch

    overlay = figure.overlays[name]
    if overlay.kind == 'categorical':
        cmap = colormap(require(params, f'visualization.colormaps.{name}'))
        n = len(overlay.categories or [])
        handles = [Patch(facecolor=category_colour(cmap, i, n), label=text)
                   for i, text in enumerate(overlay.categories or [])]
        handles.append(Patch(
            facecolor=require(params, 'visualization.missing_colour'),
            label='no value'))
        figure_handle.legend(handles=handles, fontsize=6, loc='lower right',
                             framealpha=0.9)
        return

    low, high = figure.limits[name]
    mappable = cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=low, vmax=high),
        cmap=colormap(require(params, f'visualization.colormaps.{name}')))
    bar = figure_handle.colorbar(mappable, ax=axes, shrink=0.75, pad=0.02,
                                 fraction=0.035)
    bar.set_label(overlay.label, fontsize=7)
    bar.ax.tick_params(labelsize=6)


def render_ortho_matplotlib(figure: SliceFigure, out_path: Path,
                            show: bool = False) -> Path:
    """One row per overlay, one column per plane."""
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    params = figure.params
    width, height = require(params, 'visualization.slices.figsize')
    rows, columns = len(figure.order), len(figure.slices)
    # Constrained rather than tight: these panels have a fixed aspect
    # (millimetres are equal in both directions) and a colourbar per
    # row, which is exactly the case tight_layout warns it cannot do.
    handle = plt.figure(figsize=(width * columns + 1.4, height * rows),
                        layout='constrained')

    for row, name in enumerate(figure.order):
        panels = []
        for column, one in enumerate(figure.slices):
            axes = handle.add_subplot(rows, columns,
                                      row * columns + column + 1)
            _draw_panel(axes, figure, one, name, params)
            panels.append(axes)
        # The row is identified by its colourbar, so the panels keep
        # their coordinate labels rather than the overlay's name too.
        _colourbar(handle, panels, figure, name, params)

    handle.suptitle(_figure_title(figure), fontsize=10)
    return _write(handle, out_path, params, show)


def render_montage_matplotlib(figure: SliceFigure, out_path: Path,
                              show: bool = False) -> Path:
    """A grid of parallel slices, one overlay."""
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    params = figure.params
    name = figure.order[0]
    if len(figure.order) > 1:
        logging.warning(
            "A montage shows one overlay at a time; drawing %r and ignoring "
            "%s. The ortho view takes several at once, and so does the "
            "plotly backend's dropdown.", name, ', '.join(figure.order[1:]))

    columns = min(int(require(params, 'visualization.slices.montage_columns')),
                  len(figure.slices))
    rows = int(np.ceil(len(figure.slices) / columns))
    width, height = require(params, 'visualization.slices.figsize')
    handle = plt.figure(figsize=(width * columns + 1.4, height * rows),
                        layout='constrained')

    panels = []
    for position, one in enumerate(figure.slices):
        axes = handle.add_subplot(rows, columns, position + 1)
        _draw_panel(axes, figure, one, name, params)
        panels.append(axes)
    _colourbar(handle, panels, figure, name, params)

    handle.suptitle(f'{_figure_title(figure)}\n'
                    f'{figure.overlays[name].label}', fontsize=10)
    return _write(handle, out_path, params, show)


def _figure_title(figure: SliceFigure) -> str:
    plane_names = sorted({one.plane.name for one in figure.slices})
    return (f"LGN atlas, {' / '.join(plane_names)} slices\n"
            f"{figure.banner} | {figure.selection}")


def _write(handle, out_path: Path, params: Mapping, show: bool) -> Path:
    from matplotlib import pyplot as plt
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle.savefig(out_path,
                   dpi=int(require(params, 'visualization.slices.dpi')))
    if show:
        plt.show()
    else:
        plt.close(handle)
    return out_path


# ======================================================================
# plotly
# ======================================================================
def _heatmap_traces(figure: SliceFigure, one: Slice, params: Mapping,
                    visible_overlay: str):
    """One heatmap per overlay for a single slice, plus its specs."""
    import plotly.graph_objects as go

    traces, specs = [], {}
    grid = figure.grid
    horizontal = grid.coordinate(one.plane.horizontal)
    vertical = grid.coordinate(one.plane.vertical)

    for name in figure.order:
        overlay = figure.overlays[name]
        _, spec = slice_colours(overlay, one.values[name], one.tissue,
                                params, figure.limits[name])
        specs[name] = spec
        traces.append(go.Heatmap(
            x=horizontal, y=vertical,
            z=heatmap_values(one.values[name], one.tissue, spec),
            colorscale=spec.colorscale, zmin=spec.cmin, zmax=spec.cmax,
            colorbar=dict(title=dict(text=overlay.label, side='right'),
                          tickvals=spec.tickvals, ticktext=spec.ticktext,
                          len=0.75),
            name=overlay.label, visible=(name == visible_overlay),
            hovertemplate=(f'{one.plane.horizontal_label}: %{{x:.3f}}<br>'
                           f'{one.plane.vertical_label}: %{{y:.3f}}<br>'
                           f'{overlay.label}: %{{z:.4g}}<extra></extra>')))
    return traces, specs


def _contact_trace(figure: SliceFigure, one: Slice, params: Mapping):
    """The contacts near one plane, as a single scatter."""
    import plotly.graph_objects as go

    electrodes = figure.electrodes
    slab = float(require(params, 'visualization.slices.electrode_slab_mm'))
    near = contacts_near(electrodes, one.plane, one.position_mm, slab)
    colours = ([group_colours(params, len(electrodes.group_names))[
                    int(electrodes.group[row])] for row in near.rows]
               if len(near) else [])
    hover = ([f'{electrodes.hover[row]}<br>{one.plane.normal_label} offset: '
              f'{distance * 1e3:.0f} um from this plane'
              for row, distance in zip(near.rows, near.distance_mm)]
             if len(near) else [])
    return go.Scatter(
        x=near.horizontal_mm, y=near.vertical_mm, mode='markers',
        marker=dict(size=require(params,
                                 'visualization.electrodes.marker_size'),
                    symbol='diamond', color=colours,
                    line=dict(width=1, color='white')),
        customdata=hover, hovertemplate='%{customdata}<extra></extra>',
        name=f'electrodes within {slab * 1e3:.0f} um', showlegend=True)


def _plotly_height(figure: SliceFigure, n_columns: int, params: Mapping,
                   margin_x: int, margin_y: int,
                   spacing: float = 0.06) -> float:
    """A figure height that fits the panels, from the tissue's own
    proportions.

    The panels hold millimetres equal in both directions, so their
    shape is the nucleus's, not a layout choice. Deriving the height
    from it is what stops a wide ortho row from sitting in half a page
    of white space. Capped at the configured height so a tall plane
    cannot run off the screen.
    """
    width = float(require(params, 'visualization.figure.width'))
    usable = (width - margin_x) * (1.0 - spacing * max(n_columns - 1, 0))
    panel_width = usable / max(n_columns, 1)
    ratios = [(one.extent_mm[3] - one.extent_mm[2])
              / max(one.extent_mm[1] - one.extent_mm[0], 1e-9)
              for one in figure.slices[:n_columns]]
    height = panel_width * max(ratios or [1.0]) + margin_y
    return float(min(height,
                     float(require(params, 'visualization.figure.height'))))


def _axis_layout(figure: SliceFigure, one: Slice, params: Mapping,
                 anchor: str) -> dict:
    """Equal millimetres in both directions, and the plane's labels."""
    return dict(
        xaxis=dict(title=dict(text=one.plane.horizontal_label,
                              font=dict(size=10)),
                   constrain='domain'),
        yaxis=dict(title=dict(text=one.plane.vertical_label,
                              font=dict(size=10)),
                   scaleanchor=anchor, scaleratio=1, constrain='domain'))


def render_ortho_plotly(figure: SliceFigure, out_path: Path,
                        open_browser: Optional[bool] = None) -> Path:
    """Three panels side by side, with an overlay dropdown."""
    from plotly.subplots import make_subplots

    params = figure.params
    first = figure.order[0]
    handle = make_subplots(
        rows=1, cols=len(figure.slices),
        subplot_titles=[f'{one.plane.name}<br><sub>'
                        f'{one.plane.normal_label} = '
                        f'{one.position_mm:+.2f} mm</sub>'
                        for one in figure.slices],
        horizontal_spacing=0.06)

    overlay_traces: Dict[str, List[int]] = {name: [] for name in figure.order}
    for column, one in enumerate(figure.slices, start=1):
        traces, _ = _heatmap_traces(figure, one, params, first)
        for name, trace in zip(figure.order, traces):
            trace.showscale = (column == 1)
            handle.add_trace(trace, row=1, col=column)
            overlay_traces[name].append(len(handle.data) - 1)
        if figure.electrodes is not None and len(figure.electrodes):
            contacts = _contact_trace(figure, one, params)
            contacts.showlegend = (column == 1)
            handle.add_trace(contacts, row=1, col=column)
        _add_crosshair_shapes(handle, figure, one, params, row=1, col=column)

    if len(figure.order) > 1:
        handle.update_layout(updatemenus=[dict(
            buttons=_overlay_buttons(figure, overlay_traces, handle),
            direction='down', showactive=True, x=0.0, xanchor='left',
            y=1.16, yanchor='top')])

    for column, one in enumerate(figure.slices, start=1):
        anchor = 'x' if column == 1 else f'x{column}'
        handle.update_xaxes(title_text=one.plane.horizontal_label,
                            title_font=dict(size=10), row=1, col=column)
        handle.update_yaxes(title_text=one.plane.vertical_label,
                            title_font=dict(size=10), scaleanchor=anchor,
                            scaleratio=1, row=1, col=column)

    handle.update_layout(
        title=dict(text=_plotly_title(figure, first), x=0.5,
                   xanchor='center'),
        width=require(params, 'visualization.figure.width'),
        height=_plotly_height(figure, len(figure.slices), params,
                              margin_x=180, margin_y=250),
        plot_bgcolor=require(params, 'visualization.slices.background'),
        paper_bgcolor=require(params, 'visualization.scene.background'),
        # A deep top margin: the title carries the banner, the
        # selection and the overlay's note, and the per-panel titles
        # sit under all three.
        margin=dict(l=60, r=20, t=190, b=60))
    return _write_html(handle, out_path, params, open_browser)


def render_montage_plotly(figure: SliceFigure, out_path: Path,
                          open_browser: Optional[bool] = None) -> Path:
    """One plane, with a slider stepping through the nucleus.

    One heatmap per overlay, recoloured in place: the dropdown chooses
    which is visible and the slider replaces every one of their `z`
    arrays at once. That keeps the geometry and the axes fixed, and it
    is the reason the two controls do not fight over which slice is
    showing.
    """
    import plotly.graph_objects as go

    params = figure.params
    first = figure.order[0]
    handle = go.Figure()

    opening = figure.slices[len(figure.slices) // 2]
    horizontal = figure.grid.coordinate(opening.plane.horizontal)
    vertical = figure.grid.coordinate(opening.plane.vertical)
    traces, specs = _heatmap_traces(figure, opening, params, first)
    for trace in traces:
        handle.add_trace(trace)
    overlay_traces = list(range(len(traces)))

    contact_trace = None
    if figure.electrodes is not None and len(figure.electrodes):
        handle.add_trace(_contact_trace(figure, opening, params))
        contact_trace = len(handle.data) - 1

    steps = slider_steps(figure, params, overlay_traces, contact_trace,
                         horizontal, vertical)
    handle.update_layout(sliders=[dict(
        active=len(figure.slices) // 2, steps=steps, x=0.08, len=0.88,
        currentvalue=dict(prefix=f'{opening.plane.normal_label} = ',
                          suffix=' mm'), pad=dict(t=40))])

    if len(figure.order) > 1:
        handle.update_layout(updatemenus=[dict(
            buttons=_overlay_buttons(figure,
                                     {name: [i] for i, name
                                      in enumerate(figure.order)}, handle),
            direction='down', showactive=True, x=0.0, xanchor='left',
            y=1.12, yanchor='top')])

    handle.update_layout(
        title=dict(text=_plotly_title(figure, first, opening), x=0.5,
                   xanchor='center'),
        width=require(params, 'visualization.figure.width'),
        height=_plotly_height(figure, 1, params, margin_x=260,
                              margin_y=230),
        plot_bgcolor=require(params, 'visualization.slices.background'),
        paper_bgcolor=require(params, 'visualization.scene.background'),
        margin=dict(l=60, r=20, t=130, b=90),
        **_axis_layout(figure, opening, params, 'x'))
    return _write_html(handle, out_path, params, open_browser)


def slider_steps(figure: SliceFigure, params: Mapping,
                 overlay_traces: Sequence[int],
                 contact_trace: Optional[int],
                 horizontal: np.ndarray, vertical: np.ndarray) -> List[dict]:
    """One slider step per slice, carrying that slice's own numbers.

    Each step's colours are computed from the slice it belongs to, not
    from whichever one the figure happened to open on -- with the
    shared limits, so only the values move and the scale does not.
    Getting this wrong is quiet rather than loud: the outlines would
    still change with the slider while the colours stayed behind.
    """
    steps = []
    for one in figure.slices:
        update: Dict[str, list] = {'z': []}
        for name in figure.order:
            _, spec = slice_colours(figure.overlays[name], one.values[name],
                                    one.tissue, params, figure.limits[name])
            update['z'].append(_quantised_grid(
                heatmap_values(one.values[name], one.tissue, spec), spec))

        targets = list(overlay_traces)
        if contact_trace is not None:
            slab = float(require(
                params, 'visualization.slices.electrode_slab_mm'))
            near = contacts_near(figure.electrodes, one.plane,
                                 one.position_mm, slab)
            # The heatmaps' own x and y are restated rather than left
            # out: one restyle covers every trace it targets, and
            # passing None for them would clear the millimetre axes and
            # leave the panel indexed by array position.
            update['x'] = ([list(horizontal)] * len(targets)
                           + [list(near.horizontal_mm)])
            update['y'] = ([list(vertical)] * len(targets)
                           + [list(near.vertical_mm)])
            update['z'].append(None)
            targets.append(contact_trace)

        steps.append(dict(
            method='update', label=f'{one.position_mm:+.2f}',
            args=[update,
                  {'title.text': _plotly_title(figure, figure.order[0], one)},
                  targets]))
    return steps


def _quantised_grid(values: np.ndarray, spec) -> list:
    """A slider step's `z`, rounded to what the colour axis can show.

    Slider arguments are written into the HTML as plain JSON rather
    than base64, so full float precision there costs megabytes to
    express differences no screen resolves.
    """
    span = max(float(spec.cmax) - float(spec.cmin), 1e-12)
    decimals = int(max(0, 4 - np.floor(np.log10(span))))
    rounded = np.round(values.astype(float), decimals)
    return [[None if not np.isfinite(v) else float(v) for v in row]
            for row in rounded]


def _overlay_buttons(figure: SliceFigure, overlay_traces: Mapping,
                     handle) -> List[dict]:
    """Dropdown entries that show one overlay's heatmaps and hide the
    rest. Only `visible` changes, so the slider's own state survives."""
    buttons = []
    for name in figure.order:
        visible = [False] * len(handle.data)
        for index in range(len(handle.data)):
            trace = handle.data[index]
            if trace.type != 'heatmap':
                visible[index] = True          # electrodes stay put
        for index in overlay_traces[name]:
            visible[index] = True
        buttons.append(dict(
            label=figure.overlays[name].label, method='update',
            args=[{'visible': visible},
                  {'title.text': _plotly_title(figure, name)}]))
    return buttons


def _plotly_title(figure: SliceFigure, overlay_name: str,
                  one: Optional[Slice] = None) -> str:
    overlay = figure.overlays[overlay_name]
    where = ('' if one is None
             else f'{one.plane.name}, {one.plane.normal_label} = '
                  f'{one.position_mm:+.2f} mm · ')
    return (f'LGN atlas slices — {overlay.label}<br>'
            f'<sup>{where}{figure.banner} | {figure.selection}<br>'
            f'{overlay.note}</sup>')


def _add_crosshair_shapes(handle, figure: SliceFigure, one: Slice,
                          params: Mapping, row: int, col: int):
    if figure.crosshair is None or not require(
            params, 'visualization.slices.crosshair'):
        return
    colour = require(params, 'visualization.slices.crosshair_colour')
    handle.add_vline(x=float(figure.crosshair[one.plane.horizontal]),
                     line=dict(color=colour, width=1, dash='dot'),
                     row=row, col=col)
    handle.add_hline(y=float(figure.crosshair[one.plane.vertical]),
                     line=dict(color=colour, width=1, dash='dot'),
                     row=row, col=col)


def _write_html(handle, out_path: Path, params: Mapping,
                open_browser: Optional[bool]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotlyjs = require(params, 'visualization.output.plotlyjs')
    handle.write_html(str(out_path),
                      include_plotlyjs=True if plotlyjs == 'inline'
                      else plotlyjs)
    if open_browser is None:
        open_browser = require(params, 'visualization.output.open_browser')
    if open_browser:
        import webbrowser
        webbrowser.open(out_path.resolve().as_uri())
    return out_path



# ======================================================================
# Slice and surface in one figure
# ======================================================================
def build_shell(params: Mapping, atlases: Mapping, masks: Mapping) -> list:
    """The nucleus surfaces that give a slice its 3-D context.

    The 3-D viewer's own mesh builder at the 3-D viewer's own stride,
    so the shell here is the same object that viewer draws -- just
    uniformly coloured and see-through, because in this figure it is
    the frame around the data rather than the data.
    """
    from dynaphos_lgn.viz3d import build_mesh

    stride = int(require(params, 'visualization.stride_vox'))
    return [build_mesh(atlases[name], masks[name], stride,
                       hemisphere_sign(params, name))
            for name in atlases]


def scene_plane(grid: SceneGrid, plane: Plane, one: Slice,
                spec, stride: int):
    """The cut plane as it sits inside the scene.

    :return: ``(x, y, z, colour)``, four 2-D arrays in scene
        millimetres. `z` carries NaN wherever the plane is outside the
        tissue, which punches the rectangle back to the nucleus's own
        outline -- so the plane is the slice, not a card held up
        behind it.

    Subsampled by `stride`, by taking every nth voxel rather than
    averaging n of them: a coarser picture of the same atlas, not a
    smoothed one.
    """
    tissue = one.tissue[::stride, ::stride]
    colour = spec.values.reshape(one.tissue.shape)[::stride, ::stride]

    horizontal = grid.coordinate(plane.horizontal)[::stride]
    vertical = grid.coordinate(plane.vertical)[::stride]
    across, along = np.meshgrid(horizontal, vertical)

    axes = [None, None, None]
    axes[plane.horizontal] = across
    axes[plane.vertical] = along
    axes[plane.normal] = np.full(across.shape, one.position_mm)
    x, y, z = (np.asarray(axis, dtype=np.float32) for axis in axes)

    # Holes go in z whichever plane this is: a cell with a NaN
    # coordinate is dropped, and z is the only axis every plane has a
    # varying or constant version of.
    z = np.where(tissue, z, np.nan).astype(np.float32)
    colour = np.where(tissue, colour, np.nan).astype(np.float32)
    return x, y, z, colour


def _slice_specs(figure: SliceFigure, one: Slice, params: Mapping) -> dict:
    """The `ColourSpec` of every overlay on one slice.

    Built from that slice's own values, with the figure's shared
    limits -- the values move from slice to slice, the scale does not.
    """
    return {name: slice_colours(figure.overlays[name], one.values[name],
                                one.tissue, params,
                                figure.limits[name])[1]
            for name in figure.order}


def _electrode_marker(params: Mapping, electrodes: ElectrodeSet,
                      rows: np.ndarray, size: float, opacity: float
                      ) -> dict:
    colours = group_colours(params, len(electrodes.group_names))
    return dict(size=size, symbol='diamond', opacity=opacity,
                color=[colours[int(electrodes.group[row])] for row in rows],
                line=dict(width=1, color='white'))


def render_combined_plotly(figure: SliceFigure, out_path: Path,
                           open_browser: Optional[bool] = None) -> Path:
    """Write the combined view to an HTML file."""
    return _write_html(build_combined_figure(figure), out_path,
                       figure.params, open_browser)


def build_combined_figure(figure: SliceFigure):
    """The nucleus, the active slice inside it, and that slice flat.

    One slider drives all three, as a restyle -- see `combined_steps`
    for why not an animation. The overlay dropdown only toggles
    visibility, so the two controls never contend: changing the overlay
    keeps the slice you are on, and moving the slider keeps the overlay
    you chose.

    :return: The plotly figure, so a test can look at it rather than at
        the megabytes of HTML it turns into.
    """
    from plotly.subplots import make_subplots

    params = figure.params
    if figure.shell is None:
        raise ValueError(
            "The combined view needs the nucleus surface as well as the "
            "slices; build the figure with build_shell(...) and pass it as "
            "`shell`.")

    stride = int(require(params, 'visualization.combined.plane_stride_vox'))
    fraction = float(require(params,
                             'visualization.combined.scene_width_fraction'))
    plane = figure.slices[0].plane
    # The middle slice: the ends of the nucleus are its thin tips, and
    # a figure that opens on one of those looks broken. Every slider
    # step carries both its data and its title, so even if plotly.js
    # fires a different step than `active` on load -- it computes the
    # handle's position from pixels -- the figure stays coherent.
    opening = len(figure.slices) // 2
    handle = make_subplots(
        rows=1, cols=2, column_widths=[fraction, 1.0 - fraction],
        specs=[[{'type': 'scene'}, {'type': 'xy'}]],
        horizontal_spacing=0.07,
        subplot_titles=[f'{figure.grid.hemispheres[0]} LGN'
                        if len(figure.grid.hemispheres) == 1
                        else 'both nuclei',
                        f'{plane.name} slice'])

    _add_shell(handle, figure, params)
    surface_traces = _add_plane_surfaces(handle, figure, params, stride,
                                         opening)
    heatmap_traces = _add_panels(handle, figure, params, opening)
    moving, fixed = _add_combined_electrodes(handle, figure, params, opening)

    _add_combined_controls(handle, figure, params, stride, surface_traces,
                           heatmap_traces, moving, opening)
    _lay_out_combined(handle, figure, params, plane, opening,
                      heatmap_traces[0])
    return handle


def _add_shell(handle, figure: SliceFigure, params: Mapping):
    """The nucleus, once per nucleus, as a see-through shell."""
    import plotly.graph_objects as go

    lighting = require(params, 'visualization.surface.lighting')
    for mesh in figure.shell:
        if mesh.n_faces == 0:
            continue
        triangles = mesh.triangles
        handle.add_trace(go.Mesh3d(
            x=mesh.vertices_mm[:, 0], y=mesh.vertices_mm[:, 1],
            z=mesh.vertices_mm[:, 2],
            i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
            color=require(params, 'visualization.combined.shell_colour'),
            opacity=float(require(params,
                                  'visualization.combined.shell_opacity')),
            flatshading=bool(require(params,
                                     'visualization.surface.flat_shading')),
            lighting=dict(ambient=lighting['ambient'],
                          diffuse=lighting['diffuse'],
                          specular=lighting['specular'],
                          roughness=lighting['roughness'],
                          fresnel=lighting['fresnel']),
            name=f'{mesh.hemisphere} LGN', showlegend=True,
            hoverinfo='skip'), row=1, col=1)


def _add_plane_surfaces(handle, figure: SliceFigure, params: Mapping,
                        stride: int, opening: int) -> List[int]:
    """One cut plane per overlay, all but the first hidden."""
    import plotly.graph_objects as go

    one = figure.slices[opening]
    specs = _slice_specs(figure, one, params)
    traces = []
    for position, name in enumerate(figure.order):
        spec = specs[name]
        x, y, z, colour = scene_plane(figure.grid, one.plane, one, spec,
                                      stride)
        handle.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=colour,
            colorscale=spec.colorscale, cmin=spec.cmin, cmax=spec.cmax,
            showscale=False, name=f'{figure.overlays[name].label} (in situ)',
            visible=(position == 0), hoverinfo='skip',
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0)),
            row=1, col=1)
        traces.append(len(handle.data) - 1)
    return traces


def _add_panels(handle, figure: SliceFigure, params: Mapping,
                opening: int) -> List[int]:
    """One flat heatmap per overlay, all but the first hidden."""
    one = figure.slices[opening]
    traces, _ = _heatmap_traces(figure, one, params, figure.order[0])
    indices = []
    for trace in traces:
        handle.add_trace(trace, row=1, col=2)
        indices.append(len(handle.data) - 1)
    return indices


def _add_combined_electrodes(handle, figure: SliceFigure, params: Mapping,
                             opening: int) -> Tuple[Dict[str, int],
                                                    List[int]]:
    """Every contact in the scene, and the ones this slice cuts.

    :return: ({role: trace index} for the traces a frame moves, and the
        indices of the ones it does not).
    """
    import plotly.graph_objects as go

    electrodes = figure.electrodes
    moving: Dict[str, int] = {}
    fixed: List[int] = []
    if electrodes is None or not len(electrodes):
        return moving, fixed

    size = float(require(params, 'visualization.electrodes.marker_size'))
    if require(params, 'visualization.combined.show_all_electrodes'):
        rows = np.arange(len(electrodes))
        handle.add_trace(go.Scatter3d(
            x=electrodes.positions_mm[:, 0], y=electrodes.positions_mm[:, 1],
            z=electrodes.positions_mm[:, 2], mode='markers',
            marker=_electrode_marker(params, electrodes, rows, size * 0.6,
                                     0.45),
            customdata=list(electrodes.hover),
            hovertemplate='%{customdata}<extra></extra>',
            name='electrodes', showlegend=True), row=1, col=1)
        fixed.append(len(handle.data) - 1)

    one = figure.slices[opening]
    slab = float(require(params, 'visualization.slices.electrode_slab_mm'))
    near = contacts_near(electrodes, one.plane, one.position_mm, slab)

    handle.add_trace(go.Scatter3d(
        x=electrodes.positions_mm[near.rows, 0],
        y=electrodes.positions_mm[near.rows, 1],
        z=electrodes.positions_mm[near.rows, 2], mode='markers',
        marker=_electrode_marker(params, electrodes, near.rows, size * 1.3,
                                 1.0),
        customdata=[electrodes.hover[row] for row in near.rows],
        hovertemplate='%{customdata}<extra></extra>',
        name=f'in this slice (+-{slab * 1e3:.0f} um)', showlegend=True),
        row=1, col=1)
    moving['scene'] = len(handle.data) - 1

    handle.add_trace(_contact_trace(figure, one, params), row=1, col=2)
    moving['panel'] = len(handle.data) - 1
    return moving, fixed


def _coordinate_grid(values: np.ndarray) -> list:
    """A 2-D coordinate array as nested JSON, holes included.

    Four decimals is a tenth of a micrometre, two orders of magnitude
    finer than the 25 um grid these coordinates come off, so the
    rounding is invisible and the file is a third smaller.
    """
    rounded = np.round(np.asarray(values, dtype=float), 4)
    return [[None if not np.isfinite(v) else float(v) for v in row]
            for row in rounded]


def combined_steps(figure: SliceFigure, params: Mapping, stride: int,
                   surface_traces: List[int], heatmap_traces: List[int],
                   moving: Mapping) -> List[dict]:
    """One slider step per slice, as a restyle rather than a frame.

    plotly.js cannot animate a heatmap -- it is not one of the
    animatable trace types -- so a frame moves the cut plane and the
    electrodes and leaves the flat panel undrawn, which looks like a
    blank panel and says nothing about why. A restyle redraws every
    kind of trace. The price is that the numbers go into the HTML as
    JSON rather than as binary, so they are rounded to what the colour
    axis and the voxel grid can actually show.

    One restyle covers four kinds of trace at once, so every attribute
    carries a value for every trace it targets: the real one where the
    attribute applies, None where it does not. None for an attribute a
    trace type does not have is ignored; None for one it DOES have
    would clear it, which is how the millimetre axes got lost once
    already.
    """
    electrodes = figure.electrodes
    slab = float(require(params, 'visualization.slices.electrode_slab_mm'))
    horizontal = list(figure.grid.coordinate(figure.slices[0].plane.horizontal))
    vertical = list(figure.grid.coordinate(figure.slices[0].plane.vertical))

    targets = list(surface_traces) + list(heatmap_traces)
    if moving:
        targets += [moving['scene'], moving['panel']]

    steps = []
    for one in figure.slices:
        specs = _slice_specs(figure, one, params)
        x_of, y_of, z_of, colour_of, hover_of = [], [], [], [], []

        for name in figure.order:                      # the cut planes
            x, y, z, colour = scene_plane(figure.grid, one.plane, one,
                                          specs[name], stride)
            x_of.append(_coordinate_grid(x))
            y_of.append(_coordinate_grid(y))
            z_of.append(_coordinate_grid(z))
            colour_of.append(_quantised_grid(colour, specs[name]))
            hover_of.append([])

        for name in figure.order:                      # the flat panels
            x_of.append(horizontal)
            y_of.append(vertical)
            z_of.append(_quantised_grid(
                heatmap_values(one.values[name], one.tissue, specs[name]),
                specs[name]))
            colour_of.append(None)
            hover_of.append([])

        if moving:
            near = contacts_near(electrodes, one.plane, one.position_mm,
                                 slab)
            hover = [electrodes.hover[row] for row in near.rows]
            positions = electrodes.positions_mm[near.rows]
            x_of.append([float(v) for v in positions[:, 0]])
            y_of.append([float(v) for v in positions[:, 1]])
            z_of.append([float(v) for v in positions[:, 2]])
            colour_of.append(None)
            hover_of.append(hover)

            x_of.append([float(v) for v in near.horizontal_mm])
            y_of.append([float(v) for v in near.vertical_mm])
            z_of.append(None)                          # scatter has no z
            colour_of.append(None)
            hover_of.append(hover)

        steps.append(dict(
            method='update', label=f'{one.position_mm:+.2f}',
            args=[{'x': x_of, 'y': y_of, 'z': z_of,
                   'surfacecolor': colour_of, 'customdata': hover_of},
                  {'title.text': _plotly_title(figure, figure.order[0], one)},
                  targets]))
    return steps


def _add_combined_controls(handle, figure: SliceFigure, params: Mapping,
                           stride: int, surface_traces: List[int],
                           heatmap_traces: List[int], moving: Mapping,
                           opening: int):
    """The position slider, and the overlay dropdown beside it."""
    plane = figure.slices[0].plane
    handle.update_layout(sliders=[dict(
        active=opening,
        steps=combined_steps(figure, params, stride, surface_traces,
                             heatmap_traces, moving),
        x=0.06, len=0.9,
        currentvalue=dict(prefix=f'{plane.normal_label} = ', suffix=' mm'),
        pad=dict(t=50))])

    menus = []
    if len(figure.order) > 1:
        menus.append(dict(
            buttons=_combined_overlay_buttons(figure, handle,
                                              surface_traces,
                                              heatmap_traces),
            direction='down', showactive=True, x=0.0, xanchor='left',
            y=1.08, yanchor='top'))
    if menus:
        handle.update_layout(updatemenus=menus)


def _combined_overlay_buttons(figure: SliceFigure, handle,
                              surface_traces: List[int],
                              heatmap_traces: List[int]) -> List[dict]:
    """Show one overlay's plane and panel, hide the others.

    Only `visible` moves, so the slider's position survives a change of
    overlay -- and each slider step restyles every overlay's traces,
    visible or not, so whichever one you switch to is already showing
    the slice you are on.
    """
    chosen = set(surface_traces) | set(heatmap_traces)
    buttons = []
    for position, name in enumerate(figure.order):
        visible = [index not in chosen for index in range(len(handle.data))]
        visible[surface_traces[position]] = True
        visible[heatmap_traces[position]] = True
        buttons.append(dict(
            label=figure.overlays[name].label, method='update',
            args=[{'visible': visible},
                  {'title.text': _plotly_title(figure, name)}]))
    return buttons


def _lay_out_combined(handle, figure: SliceFigure, params: Mapping,
                      plane: Plane, opening: int, panel_trace: int):
    """Axes, camera and title for the two halves."""
    camera = require(params, 'visualization.scene.camera_eye')
    labels = require(params, 'visualization.scene.axis_labels')

    # Asked of the trace rather than assumed: a scene in the first cell
    # does not consume an x axis, so the panel beside it is the
    # figure's FIRST xy subplot and its axis is 'x', not 'x2'. Anchoring
    # to an axis that does not exist leaves the panel blank.
    anchor = handle.data[panel_trace].xaxis or 'x'
    handle.update_xaxes(title_text=plane.horizontal_label,
                        title_font=dict(size=10), row=1, col=2)
    handle.update_yaxes(title_text=plane.vertical_label,
                        title_font=dict(size=10), scaleanchor=anchor,
                        scaleratio=1, row=1, col=2)
    handle.update_layout(
        title=dict(text=_plotly_title(figure, figure.order[0],
                                      figure.slices[opening]),
                   x=0.5, xanchor='center'),
        scene=dict(aspectmode='data',
                   xaxis_title=labels['x'], yaxis_title=labels['y'],
                   zaxis_title=labels['z'],
                   camera=dict(eye=dict(x=camera[0], y=camera[1],
                                        z=camera[2]))),
        width=require(params, 'visualization.figure.width'),
        height=require(params, 'visualization.figure.height'),
        plot_bgcolor=require(params, 'visualization.slices.background'),
        paper_bgcolor=require(params, 'visualization.scene.background'),
        legend=dict(x=0.0, y=0.0, xanchor='left', yanchor='bottom',
                    bgcolor='rgba(255,255,255,0.75)'),
        margin=dict(l=40, r=20, t=150, b=110))


def render_combined_matplotlib(figure: SliceFigure, out_path: Path,
                               show: bool = False) -> Path:
    """The still version: the shell with the cut plane in it, beside
    the flat slice.

    No slider, so it draws the middle slice -- or whichever one `--at`
    picked. The point of the combined view is the interaction, so this
    exists to put the same picture in a document, not to replace it.
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    params = figure.params
    if figure.shell is None:
        raise ValueError("The combined view needs the nucleus surface too.")

    name = figure.order[0]
    one = figure.slices[len(figure.slices) // 2]
    width, height = require(params, 'visualization.slices.figsize')
    handle = plt.figure(figsize=(width * 2.4 + 1.4, height * 1.4),
                        layout='constrained')

    scene = handle.add_subplot(1, 2, 1, projection='3d')
    _draw_shell_matplotlib(scene, figure, one, name, params)
    panel = handle.add_subplot(1, 2, 2)
    _draw_panel(panel, figure, one, name, params)
    _colourbar(handle, [panel], figure, name, params)

    handle.suptitle(f'{_figure_title(figure)}\n'
                    f'{figure.overlays[name].label}', fontsize=10)
    return _write(handle, out_path, params, show)


def _draw_shell_matplotlib(axes, figure: SliceFigure, one: Slice, name: str,
                           params: Mapping):
    """The nucleus outline with the cut plane inside it."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    stride = int(require(params, 'visualization.combined.plane_stride_vox'))
    limits = []
    for mesh in figure.shell:
        if mesh.n_faces == 0:
            continue
        quads = mesh.quad_vertices_mm
        axes.add_collection3d(Poly3DCollection(
            quads,
            facecolors=require(params,
                               'visualization.combined.shell_colour'),
            alpha=float(require(params,
                                'visualization.combined.shell_opacity')),
            edgecolors='none'))
        limits.append(mesh.vertices_mm)

    # The same RGBA the flat panel uses, subsampled the same way, not a
    # second colour calculation: normalising the colormap linearly would
    # give a categorical overlay four colours that are not its four
    # categories, and the halves of the figure would then disagree about
    # what a lamina looks like.
    rgba, spec = slice_colours(figure.overlays[name], one.values[name],
                               one.tissue, params, figure.limits[name])
    x, y, z, _ = scene_plane(figure.grid, one.plane, one, spec, stride)
    axes.plot_surface(x, y, z, facecolors=rgba[::stride, ::stride],
                      shade=False, rstride=1, cstride=1, linewidth=0,
                      antialiased=False)

    if figure.electrodes is not None and len(figure.electrodes):
        positions = figure.electrodes.positions_mm
        axes.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                     s=float(require(
                         params, 'visualization.electrodes.marker_size')) ** 2,
                     c=require(params, 'visualization.electrodes.colour'),
                     marker='D', depthshade=False, zorder=5)

    labels = require(params, 'visualization.scene.axis_labels')
    camera = require(params, 'visualization.scene.camera_eye')
    _set_scene_box(axes, limits, labels, camera)


def _set_scene_box(axes, limits: list, labels: Mapping,
                   camera: Sequence[float]):
    if not limits:
        return
    stacked = np.concatenate(limits)
    low, high = stacked.min(axis=0), stacked.max(axis=0)
    for setter, lo, hi in zip((axes.set_xlim, axes.set_ylim, axes.set_zlim),
                              low, high):
        setter(lo, hi)
    axes.set_box_aspect(tuple(np.maximum(high - low, 1e-6)))
    axes.set_xlabel(labels['x'], fontsize=7)
    axes.set_ylabel(labels['y'], fontsize=7)
    axes.set_zlabel(labels['z'], fontsize=7)
    axes.tick_params(labelsize=6)
    eye = np.asarray(camera, dtype=float)
    radius = float(np.linalg.norm(eye))
    axes.view_init(elev=float(np.degrees(np.arcsin(eye[2] / radius))),
                   azim=float(np.degrees(np.arctan2(eye[1], eye[0]))))

# ======================================================================
# Command line
# ======================================================================
def build_parser() -> argparse.ArgumentParser:
    """The slice viewer's command line.

    Every option it shares with the 3-D viewer comes from the same
    builder, so `--overlay`, `--clip`, `--probe` and the rest mean
    exactly the same thing in both.
    """
    parser = argparse.ArgumentParser(
        prog='python -m dynaphos_lgn.slices',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_atlas_arguments(parser)
    add_tissue_arguments(parser, surface=False)
    # The combined view draws the nucleus as well, so it is the one
    # slice view that has a surface to coarsen.
    parser.add_argument('--stride', type=int, default=None,
                        help='--view combined only: atlas voxels per voxel '
                             'of the surrounding shell, per axis (default: '
                             'visualization.stride_vox).')

    cuts = parser.add_argument_group('slices')
    cuts.add_argument('--view', choices=['ortho', 'slices', 'combined'],
                      default='ortho',
                      help="'ortho' cuts all three planes through one "
                           "point; 'slices' steps one plane through the "
                           "nucleus; 'combined' does that too and draws the "
                           "active slice inside the 3-D nucleus beside it "
                           "(default: ortho).")
    cuts.add_argument('--plane', choices=sorted(PLANES), default=None,
                      help='Which plane --view slices and --view combined '
                           'step through (default: '
                           'visualization.slices.plane, or '
                           'visualization.combined.plane).')
    cuts.add_argument('--at', default=None, metavar='ML,DV,AP',
                      help='Where to cut, in SCENE millimetres (signed ML, '
                           'left nucleus negative). Also accepts '
                           "'electrode:N'. Default: the centroid of the "
                           'drawn tissue.')
    cuts.add_argument('--count', type=int, default=None,
                      help='How many slices --view slices draws '
                           '(default: visualization.slices.count).')
    cuts.add_argument('--range', dest='range_mm', default=None,
                      metavar='LOW:HIGH',
                      help='Restrict --view slices to this span along the '
                           "plane's normal, in millimetres.")
    cuts.add_argument('--slab', type=float, default=None,
                      help='Draw contacts within this distance of a plane, '
                           'in mm (default: '
                           'visualization.slices.electrode_slab_mm).')
    cuts.add_argument('--no-crosshair', action='store_true',
                      help='Leave off the lines showing where the other '
                           'planes cut.')

    add_electrode_arguments(parser)
    add_output_arguments(parser)
    return parser


def _apply_slice_overrides(params: dict, options) -> dict:
    """Fold the slice-specific flags into the config, as viz3d does for
    its own."""
    slices = params['visualization']['slices']
    if options.plane is not None:
        slices['plane'] = options.plane
        # The combined view keeps its own default plane, so that the
        # two views can sit on different ones in a config -- but a flag
        # means the plane the user is looking at, whichever view it is.
        params['visualization']['combined']['plane'] = options.plane
    if options.count is not None:
        slices['count'] = int(options.count)
    if options.slab is not None:
        slices['electrode_slab_mm'] = float(options.slab)
    if options.no_crosshair:
        slices['crosshair'] = False
    if options.range_mm is not None:
        parts = str(options.range_mm).split(':')
        if len(parts) != 2:
            raise ValueError(
                f"--range wants LOW:HIGH in millimetres, got "
                f"{options.range_mm!r}.")
        slices['range_mm'] = [float(parts[0]), float(parts[1])]
    return params


def main(argv: Optional[Sequence[str]] = None) -> int:
    from dynaphos.utils import load_params
    from dynaphos_lgn.build import (build_atlas, build_hemisphere_atlas,
                                    configured_hemispheres)

    options = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if options.verbose
                        else logging.WARNING,
                        format='%(levelname)s: %(message)s')

    if options.list_overlays:
        print(list_overlays())
        return 0

    repository = Path(__file__).resolve().parents[1]
    params_path = Path(options.params
                       or repository / 'config' / 'params_lgn.yaml')
    params = load_params(str(params_path))
    params = _apply_slice_overrides(_apply_overrides(params, options),
                                    options)

    # The slice viewer shows one overlay at a time per panel, so it
    # defaults to the first configured one rather than to all of them:
    # every extra overlay is another full set of images in the file.
    overlays = list(require(params, 'visualization.overlays'))
    if not options.overlay:
        overlays = overlays[:1]
    unknown = [name for name in overlays if name not in OVERLAY_BUILDERS]
    if unknown:
        print(f"Unknown overlay(s) {', '.join(unknown)}.\n{list_overlays()}",
              file=sys.stderr)
        return 2

    atlas = build_atlas(params, options.atlas_dir, options.synthetic)
    atlases = {name: build_hemisphere_atlas(params, atlas, name)
               for name in configured_hemispheres(params)}

    needs_jacobian = (bool(set(overlays) & JACOBIAN_OVERLAYS)
                      or options.electrodes == 'array')
    provide = (jacobian_provider(params, atlas, options.atlas_dir,
                                 options.recompute_jacobian)
               if needs_jacobian else None)

    rng = np.random.default_rng(require(params, 'run.seed'))
    electrodes = collect_electrodes(params, atlases, options, rng, provide)

    mask_options = dict(
        cell_class=options.cell_class, layers=options.layers,
        min_eccentricity_deg=options.ecc_min,
        max_eccentricity_deg=options.ecc_max, clips=options.clip,
        include_ipsi_placeholder=True if options.include_ipsi else None)
    masks = {name: tissue_mask(atlases[name], params, **mask_options)
             for name in atlases}
    contexts = {name: AtlasContext(
        atlases[name], params,
        None if provide is None else (lambda h=name: provide(h)))
        for name in atlases}

    try:
        grid = build_scene_grid(params, atlases, masks)
        position = resolve_position(grid, masks, params, options.at,
                                    electrodes)
        planes = _planes_to_cut(params, grid, masks, options, position)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    from dynaphos_lgn.viz3d import _describe_selection
    shell = (build_shell(params, atlases, masks)
             if options.view == 'combined' else None)
    figure = build_slice_figure(
        params, atlases, masks, contexts, overlays, planes, electrodes,
        banner=atlas_banner(atlas),
        selection=_describe_selection(mask_options, params),
        crosshair=position if options.view == 'ortho' else None, grid=grid,
        shell=shell)
    print(figure.describe())

    return _render(figure, params, options, repository)


def _planes_to_cut(params: Mapping, grid: SceneGrid, masks: Mapping,
                   options, position: np.ndarray
                   ) -> List[Tuple[Plane, int]]:
    """Which planes at which indices, for the chosen view."""
    if options.view == 'ortho':
        return [(plane, grid.nearest_index(plane.normal,
                                           position[plane.normal]))
                for plane in (PLANES['coronal'], PLANES['sagittal'],
                              PLANES['horizontal'])]

    combined = options.view == 'combined'
    section = 'combined' if combined else 'slices'
    plane = PLANES[require(params, f'visualization.{section}.plane')]
    indices = slice_indices(
        grid, masks, plane,
        int(require(params, 'visualization.slices.count')),
        require(params, 'visualization.slices.range_mm'),
        float(require(params,
                      f'visualization.{section}.min_slice_fraction')))
    return [(plane, index) for index in indices]


def _warn_about_size(megabytes: float, params: Mapping, n_slices: int,
                     n_overlays: int):
    """Say what makes the file big, and which knob shrinks it.

    Every slice carries a full-resolution panel for every overlay, so
    the size is the product of the two -- which is not obvious from
    either flag on its own.
    """
    limit = float(require(params, 'visualization.output.size_warning_mb'))
    if megabytes <= limit:
        return
    print(f"      That is {n_slices} slices x {n_overlays} overlay(s) of "
          f"data, and it will be slow to open."
          f"\n      Fewer of either shrinks it in proportion; "
          f"visualization.output.plotlyjs: cdn saves a further ~4 MB.")


def _render(figure: SliceFigure, params: Mapping, options,
            repository: Path) -> int:
    backend = require(params, 'visualization.backend')
    directory = Path(require(params, 'visualization.output.directory'))
    if not directory.is_absolute():
        directory = repository / directory
    kind = options.view if options.view in ('ortho', 'combined') else 'slices'
    plotly_renderers = {'ortho': render_ortho_plotly,
                        'slices': render_montage_plotly,
                        'combined': render_combined_plotly}
    matplotlib_renderers = {'ortho': render_ortho_matplotlib,
                            'slices': render_montage_matplotlib,
                            'combined': render_combined_matplotlib}

    if backend == 'plotly':
        from importlib.util import find_spec
        if find_spec('plotly') is None:
            print("The plotly backend needs plotly: `pip install plotly`, "
                  "or use --backend matplotlib, which needs nothing "
                  "further.", file=sys.stderr)
            return 3
        out = Path(options.out or directory / require(
            params, f'visualization.output.{kind}_html_name'))
        plotly_renderers[kind](figure, out)
        megabytes = out.stat().st_size / 1e6
        print(f"\nWrote {out} ({megabytes:.1f} MB).")
        _warn_about_size(megabytes, params, len(figure.slices),
                         len(figure.order))
    elif backend == 'matplotlib':
        out = Path(options.out or directory / require(
            params, f'visualization.output.{kind}_png_name'))
        matplotlib_renderers[kind](figure, out, show=options.show)
        print(f"\nWrote {out}.")
    else:
        print(f"Unknown backend {backend!r}; expected plotly or matplotlib.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':       # pragma: no cover
    raise SystemExit(main())
