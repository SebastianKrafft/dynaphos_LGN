"""Live exploration of the LGN atlas in a notebook.

`viz3d` and `slices` write **figures**: every state a control can
reach is computed up front and frozen into one HTML file. That is what
makes such a file shareable, and it is also what caps how interactive
it can be -- a slider there is a lookup table, so an arbitrary slice
position, a different spacing between slices, or a switch from coronal
to sagittal is not something the file can do. It would have to contain
every one of them.

This module is the other half. Same volume, same overlays, same
colours, same electrodes, driven from ipywidgets with the kernel still
attached, so nothing is precomputed and the controls are free:

- **position** steps a plane through the nucleus, one 25 um voxel at a
  time, anywhere in the volume;
- **count** and **spacing** turn that single plane into a stack of
  parallel cuts whose separation you set in millimetres;
- **plane** switches between coronal, sagittal and horizontal;
- **overlay**, the electrode slab, the shell's opacity and the in-scene
  plane's stride are all live;
- the tissue selection -- cell class, laminae, eccentricity, clips --
  rebuilds the masks, the grid and the shell on demand.

The price is a running kernel. This writes no file and nothing here can
be emailed, which is what `Explorer.export` is for: the explorer keeps
its state in the parameter dictionary rather than in the widgets, so at
any moment it can hand back the YAML that would reproduce what you are
looking at, and the `slices.py` command line that renders it as a
static figure.

Usage, from a notebook::

    from dynaphos_lgn import explore
    ex = explore.Explorer.open()          # config, atlas, first draw
    ex.widget                             # the viewer

    print(ex.export().text)               # what to paste back
    ex.export().save()                    # ... or write it out

`examples/demo_lgn_explorer.ipynb` does exactly that, with prose.

Needs `ipywidgets`, `plotly` and `anywidget` (which is what plotly's
`FigureWidget` is built on). ``python -m dynaphos_lgn.explore`` checks
for them and says what is missing.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, \
    Tuple

import numpy as np

from dynaphos_lgn.params import require
from dynaphos_lgn.slices import (PLANES, Plane, SceneGrid, Slice,
                                 _electrode_marker, build_scene_grid,
                                 build_shell, column_at, contacts_near, cut,
                                 heatmap_values, occupied_range,
                                 plane_voxel_counts, sample_limits,
                                 scene_plane, slice_colours)
from dynaphos_lgn.viz3d import (JACOBIAN_OVERLAYS, OVERLAY_BUILDERS,
                                AtlasContext, ElectrodeSet, Overlay,
                                _describe_selection, atlas_banner,
                                collect_electrodes, compute_overlay,
                                group_colours, jacobian_provider,
                                list_overlays, tissue_mask)


# ======================================================================
# The volume: everything a tissue selection fixes
# ======================================================================
@dataclass
class Volume:
    """The atlas and everything that follows from WHICH tissue is shown.

    The masks, the scene grid, the shell mesh and the colour limits all
    depend on the selection and none of them depend on where the cut
    is. Holding them still is what lets the position, spacing and plane
    controls be free: moving any of those costs one plane's worth of
    work, not a rebuild.

    :ivar overlays: {name: a representative `Overlay`}, for its label,
        its categories and its honesty note. Its numbers are whichever
        voxel came first and are not used.
    :ivar limits: {name: colour-axis ends}, from a sample of the whole
        volume, so a colour means the same number at every position.
    """

    params: Mapping
    atlases: Dict[str, Any]
    masks: Dict[str, np.ndarray]
    contexts: Dict[str, AtlasContext]
    grid: SceneGrid
    shell: List[Any]
    electrodes: Optional[ElectrodeSet]
    overlays: Dict[str, Overlay]
    limits: Dict[str, Optional[Tuple[float, float]]]
    banner: str
    selection: str
    eccentricity_deg_max: float

    @property
    def order(self) -> List[str]:
        return list(self.overlays)

    def describe(self) -> str:
        """What was loaded, in the style of the package's other
        `describe` methods."""
        lines = [f"LGN explorer volume: {self.banner}",
                 f"  tissue shown      : {self.selection}",
                 f"  scene grid        : {len(self.grid.x_mm)} x "
                 f"{len(self.grid.y_mm)} x {len(self.grid.z_mm)} voxels of "
                 f"{self.grid.voxel_size_mm * 1e3:.0f} um "
                 f"({', '.join(self.grid.hemispheres)} LGN)",
                 f"  overlays offered  : {', '.join(self.order)}"]
        faces = sum(mesh.n_faces for mesh in self.shell)
        lines.append(f"  shell             : {faces:,} faces at stride "
                     f"{self.shell[0].stride}")
        if self.electrodes is not None:
            lines.append(f"  electrodes        : {len(self.electrodes)} "
                         f"({self.electrodes.source})")
        return '\n'.join(lines)


def representative_overlay(name: str, masks: Mapping,
                           contexts: Mapping) -> Overlay:
    """One evaluated `Overlay` per name, for its label and categories.

    Its values come from whichever voxel happened to be first and are
    thrown away. That is enough because a categorical overlay's
    category list comes from the config, not from the sample -- see
    `viz3d._categorical`, which walks the configured code-to-name map
    in sorted order whatever values it is handed.
    """
    hemisphere = next(iter(masks))
    voxels = np.argwhere(masks[hemisphere])[:1]
    if not len(voxels):
        raise ValueError("The selection leaves no tissue to explore.")
    return compute_overlay(name, contexts[hemisphere], voxels)


def build_volume(params: Mapping, atlases: Mapping,
                 jacobian_for: Optional[Callable[[str], object]],
                 overlay_names: Sequence[str],
                 mask_options: Optional[Mapping] = None,
                 electrodes: Optional[ElectrodeSet] = None,
                 banner: str = '') -> Volume:
    """Select the tissue and derive everything a live draw needs.

    This is the expensive call -- the shell mesh and the volume-wide
    colour sample are both in it -- and the one the explorer only makes
    when the selection changes.

    :param jacobian_for: Callable returning one hemisphere's
        `JacobianAtlas`, or None where none is available. Loaded at
        most once, and only if a Jacobian-derived overlay is asked for.
    :param mask_options: Keyword arguments for `viz3d.tissue_mask`.
    """
    mask_options = dict(mask_options or {})
    masks = {name: tissue_mask(atlases[name], params, **mask_options)
             for name in atlases}
    contexts = {name: AtlasContext(
        atlases[name], params,
        None if jacobian_for is None else (lambda h=name: jacobian_for(h)))
        for name in atlases}

    grid = build_scene_grid(params, atlases, masks)
    shell = build_shell(params, atlases, masks)
    rng = np.random.default_rng(require(params, 'run.seed'))
    limits = sample_limits(params, masks, contexts, overlay_names, rng)
    overlays = {name: representative_overlay(name, masks, contexts)
                for name in overlay_names}

    return Volume(params=params, atlases=dict(atlases), masks=masks,
                  contexts=contexts, grid=grid, shell=shell,
                  electrodes=electrodes, overlays=overlays, limits=limits,
                  banner=banner,
                  selection=_describe_selection(mask_options, params),
                  eccentricity_deg_max=eccentricity_ceiling(atlases))


def eccentricity_ceiling(atlases: Mapping) -> float:
    """The largest eccentricity the atlas holds anywhere.

    The top of the eccentricity filter's range. Two things about it
    are deliberate. It comes from the data rather than from
    `atlas.max_eccentricity_deg`, which is a documented extrapolation
    limit per cell class and would leave the upper half of the control
    doing nothing on an atlas -- the synthetic one, say -- that never
    reaches it. And it comes from the whole atlas rather than from the
    current selection, because a range that shrank to fit what the
    filter had already removed could never be widened again.
    """
    highest = 0.0
    for atlas in atlases.values():
        values = atlas.eccentricity_deg[atlas.valid]
        finite = values[np.isfinite(values)]
        if len(finite):
            highest = max(highest, float(finite.max()))
    return highest


def fullest_plane_mm(volume: Volume, plane: Plane) -> float:
    """Where along a normal the tissue is thickest, in scene mm.

    The explorer's default position for a plane. Preferred to the
    tissue's centroid, which for two nuclei averages their signed ML
    coordinates and so lands in the 17 mm gap between them -- a
    sagittal cut through nothing at all.
    """
    counts = plane_voxel_counts(volume.grid, volume.masks, plane)
    if not counts.any():
        raise ValueError(f"No tissue in any {plane.name} plane.")
    return float(volume.grid.coordinate(plane.normal)[int(np.argmax(counts))])


# ======================================================================
# Where the parallel cuts sit
# ======================================================================
@dataclass
class Stack:
    """A set of parallel planes along one normal.

    :ivar indices: Scene indices along `plane.normal`, ascending.
    :ivar active: Position in `indices` of the plane the flat panel is
        showing.
    :ivar spacing_mm: The spacing actually achieved, which is a whole
        number of voxels and so need not be the one asked for.
    :ivar requested: How many planes were asked for, which is more than
        were placed when the stack runs off the end of the volume.
    """

    plane: Plane
    indices: List[int]
    active: int
    spacing_mm: float
    requested: int

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def active_index(self) -> int:
        return self.indices[self.active]

    @property
    def dropped(self) -> int:
        """Planes that fell outside the volume and were not drawn."""
        return self.requested - len(self.indices)


def stack_indices(grid: SceneGrid, plane: Plane, centre_mm: float,
                  count: int, spacing_mm: float) -> Stack:
    """`count` parallel planes `spacing_mm` apart, one exactly at
    `centre_mm`.

    Built outwards from the active plane in whole voxels, so every
    neighbouring pair is separated by exactly the same distance and the
    active plane sits exactly where the position control says. Asking
    for a spacing the 25 um grid cannot express therefore moves the
    OTHER planes, never the one being read; the spacing that resulted
    travels on the `Stack` instead of being assumed to be the one asked
    for.

    Planes that fall off the end of the volume are dropped, not shifted
    inwards: a stack that slid to stay whole would no longer be centred
    where it says it is. How many were lost is on `Stack.dropped`.

    :param centre_mm: Along `plane.normal`, in scene millimetres.
    :raises ValueError: for a non-positive count or spacing, either of
        which would otherwise silently produce an empty stack.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}.")
    if spacing_mm <= 0:
        raise ValueError(f"spacing_mm must be positive, got {spacing_mm}.")

    coordinate = grid.coordinate(plane.normal)
    step = max(1, int(round(float(spacing_mm) / grid.voxel_size_mm)))
    centre = grid.nearest_index(plane.normal, centre_mm)

    # The centre member is at offset 0, so it always survives the range
    # check below and `active` is always well defined. For an even
    # count that puts it at the lower of the two middle positions.
    offsets = (np.arange(count) - (count - 1) // 2) * step
    raw = centre + offsets
    kept = raw[(raw >= 0) & (raw < len(coordinate))]

    return Stack(plane=plane, indices=[int(value) for value in kept],
                 active=int(np.flatnonzero(kept == centre)[0]),
                 spacing_mm=step * grid.voxel_size_mm, requested=int(count))


# ======================================================================
# One drawable state
# ======================================================================
@dataclass
class View:
    """What the live controls currently say to draw.

    Everything here is cheap to change: none of it touches the masks,
    the grid or the shell.
    """

    overlay: str
    stack: Stack
    plane_stride_vox: int
    electrode_slab_mm: float
    shell_opacity: float


@dataclass
class Drawn:
    """What the last draw actually put on screen.

    Reported rather than assumed, because three of these differ from
    what was asked for: the stack loses planes at the ends of the
    volume, the spacing snaps to whole voxels, and a position can land
    on a plane with no tissue in it at all.
    """

    active: Slice
    stack: Stack
    n_contacts: int
    seconds: float
    off_tissue: bool


# ======================================================================
# The figure
# ======================================================================
@dataclass
class LiveFigure:
    """A `FigureWidget` and the trace indices that address it.

    The trace list is fixed at construction and never grows: a live
    figure that added and removed traces would renumber them under
    every handler, so the stack's planes are `max_planes` `Surface`
    traces that are shown or hidden. `plane_traces[0]` is always the
    active plane, whichever member of the stack that is.
    """

    handle: Any
    shell_traces: List[int]
    plane_traces: List[int]
    panel_trace: int
    scene_all_trace: Optional[int]
    scene_near_trace: Optional[int]
    panel_near_trace: Optional[int]


def _blank_surface():
    """Placeholder geometry for a plane trace that is not in use.

    Real coordinates with a NaN height, rather than NaN throughout: a
    surface whose x and y are also NaN contributes nothing an autorange
    can use, and the scene's extent would then depend on how many
    planes happened to be showing.
    """
    flat = np.zeros((2, 2), dtype=np.float32)
    return flat, flat.copy(), np.full((2, 2), np.nan, dtype=np.float32)


def build_live_figure(volume: Volume, params: Mapping) -> LiveFigure:
    """The two halves, with every trace the explorer will ever need.

    Left: the nucleus as a see-through shell with the cut planes inside
    it, exactly as `slices --view combined` draws them. Right: the
    active plane flat, one image pixel per atlas voxel.

    Nothing is filled in here. `draw` puts the first `View` in, so that
    the opening picture and every later one come from one code path.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fraction = float(require(params,
                             'visualization.combined.scene_width_fraction'))
    max_planes = int(require(params, 'visualization.explorer.max_planes'))
    lighting = require(params, 'visualization.surface.lighting')

    handle = make_subplots(rows=1, cols=2,
                           specs=[[{'type': 'scene'}, {'type': 'xy'}]],
                           column_widths=[fraction, 1.0 - fraction],
                           horizontal_spacing=0.06)

    shell_traces: List[int] = []
    for mesh in volume.shell:
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
        shell_traces.append(len(handle.data) - 1)

    x, y, z = _blank_surface()
    plane_traces: List[int] = []
    for position in range(max_planes):
        handle.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=z, showscale=False, visible=False,
            hoverinfo='skip', name='cut plane',
            showlegend=position == 0,
            # Flat, unlit: the plane is a picture of the atlas, and
            # shading it would vary its colours with the camera.
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0)),
            row=1, col=1)
        plane_traces.append(len(handle.data) - 1)

    scene_all = scene_near = panel_near = None
    if volume.electrodes is not None and len(volume.electrodes):
        electrodes = volume.electrodes
        size = float(require(params, 'visualization.electrodes.marker_size'))
        rows = np.arange(len(electrodes))
        if require(params, 'visualization.combined.show_all_electrodes'):
            handle.add_trace(go.Scatter3d(
                x=electrodes.positions_mm[:, 0],
                y=electrodes.positions_mm[:, 1],
                z=electrodes.positions_mm[:, 2], mode='markers',
                marker=_electrode_marker(params, electrodes, rows,
                                         size * 0.6, 0.45),
                customdata=list(electrodes.hover),
                hovertemplate='%{customdata}<extra></extra>',
                name='electrodes', showlegend=True), row=1, col=1)
            scene_all = len(handle.data) - 1

        handle.add_trace(go.Scatter3d(
            x=[], y=[], z=[], mode='markers',
            marker=_electrode_marker(params, electrodes, rows[:0], size * 1.3,
                                     1.0),
            customdata=[], hovertemplate='%{customdata}<extra></extra>',
            name='in this slice', showlegend=True), row=1, col=1)
        scene_near = len(handle.data) - 1

    handle.add_trace(go.Heatmap(x=[], y=[], z=[[]], showscale=True,
                                colorbar=dict(len=0.75)), row=1, col=2)
    panel_trace = len(handle.data) - 1

    if volume.electrodes is not None and len(volume.electrodes):
        handle.add_trace(go.Scatter(
            x=[], y=[], mode='markers',
            marker=dict(size=require(
                params, 'visualization.electrodes.marker_size'),
                symbol='diamond', line=dict(width=1, color='white')),
            customdata=[], hovertemplate='%{customdata}<extra></extra>',
            name='in this slice', showlegend=False), row=1, col=2)
        panel_near = len(handle.data) - 1

    _lay_out(handle, params)
    return LiveFigure(handle=go.FigureWidget(handle),
                      shell_traces=shell_traces, plane_traces=plane_traces,
                      panel_trace=panel_trace, scene_all_trace=scene_all,
                      scene_near_trace=scene_near, panel_near_trace=panel_near)


def _lay_out(handle, params: Mapping):
    """Axes, camera and the revisions that decide what survives a
    redraw.

    `uirevision` is how a plotly figure keeps the user's own camera and
    zoom across an update. The scene's is a constant, so rotating the
    nucleus survives every control; each panel axis carries the plane's
    name instead, so switching from coronal to sagittal -- which
    changes what the axes even measure -- correctly resets the panel
    rather than holding a zoom onto coordinates that no longer exist.
    """
    camera = require(params, 'visualization.scene.camera_eye')
    labels = require(params, 'visualization.scene.axis_labels')
    handle.update_layout(
        uirevision='lgn-explorer',
        scene=dict(aspectmode='data', uirevision='lgn-explorer-scene',
                   xaxis_title=labels['x'], yaxis_title=labels['y'],
                   zaxis_title=labels['z'],
                   camera=dict(eye=dict(x=camera[0], y=camera[1],
                                        z=camera[2]))),
        width=require(params, 'visualization.explorer.figure_width'),
        height=require(params, 'visualization.explorer.figure_height'),
        plot_bgcolor=require(params, 'visualization.slices.background'),
        paper_bgcolor=require(params, 'visualization.scene.background'),
        legend=dict(x=0.0, y=0.0, xanchor='left', yanchor='bottom',
                    bgcolor='rgba(255,255,255,0.75)'),
        margin=dict(l=40, r=20, t=70, b=60))


def rounded(values: np.ndarray, decimals: int) -> np.ndarray:
    """Values at the precision the picture can actually show.

    plotly's widget serialises only ONE-dimensional numeric arrays as
    binary buffers -- see `plotly.serializers._py_to_js` -- and every
    array a cut plane or a flat panel is made of has two dimensions, so
    they all reach the browser as JSON text. There
    `-10.774999999999999` costs twenty characters to say what `-10.775`
    says in seven, and a two-nucleus coronal redraw is a million
    numbers.

    Kept in float64 deliberately. Rounding in float32 is worse than not
    rounding at all: the nearest float32 to 10.775 prints as
    10.774999618530273, so the saving is undone the moment it is
    written out.
    """
    return np.round(np.asarray(values, dtype=np.float64), int(decimals))


def colour_decimals(spec, params: Mapping) -> int:
    """How finely a colour value is worth stating.

    Relative to the colour axis's own span, as `viz3d._quantised` does
    it for the same reason: the significant digits are what matter, and
    an eccentricity in degrees and a cell count per voxel do not share
    a scale.
    """
    digits = int(require(params,
                         'visualization.explorer.colour_significant_digits'))
    span = max(float(spec.cmax) - float(spec.cmin), 1e-12)
    return int(max(0, digits - np.floor(np.log10(span))))


def draw(live: LiveFigure, volume: Volume, view: View) -> Drawn:
    """Put one `View` on screen, in a single update.

    Every plane in the stack is cut and coloured before anything is
    sent, and the whole figure then changes at once: a `batch_update`
    is one message to the browser rather than one per trace, which is
    the difference between a slider that tracks and a slider that
    stutters.

    Only the active overlay is evaluated. The others stay one control
    change away, and not computing them is what keeps a plane's cost to
    one gather and one lookup.
    """
    started = time.perf_counter()
    params = volume.params
    name = view.overlay
    overlay = volume.overlays[name]
    plane = view.stack.plane

    cuts = [cut(volume.grid, volume.masks, volume.contexts, [name], plane,
                index) for index in view.stack.indices]
    specs = [slice_colours(overlay, one.values[name], one.tissue, params,
                           volume.limits[name])[1] for one in cuts]
    active = cuts[view.stack.active]
    active_spec = specs[view.stack.active]

    # The active plane first, so it is always trace 0 and always the
    # opaque one however the stack is arranged around it.
    order = [view.stack.active] + [position for position in range(len(cuts))
                                   if position != view.stack.active]
    opaque = float(require(params,
                           'visualization.explorer.active_plane_opacity'))
    faint = float(require(params,
                          'visualization.explorer.companion_plane_opacity'))
    coarser = int(require(params,
                          'visualization.explorer.companion_stride_multiple'))
    near = contacts_near(volume.electrodes, plane, active.position_mm,
                         view.electrode_slab_mm)
    places = int(require(params, 'visualization.explorer.coordinate_decimals'))

    with live.handle.batch_update():
        for trace in live.shell_traces:
            live.handle.data[trace].opacity = view.shell_opacity

        for position, trace in enumerate(live.plane_traces):
            if position >= len(order):
                live.handle.data[trace].visible = False
                continue
            which = order[position]
            spec = specs[which]
            # The companions are context, and on two nuclei they are
            # most of what a redraw ships, so they are drawn coarser
            # than the one being read.
            x, y, z, colour = scene_plane(
                volume.grid, plane, cuts[which], spec,
                view.plane_stride_vox * (1 if position == 0 else coarser))
            live.handle.data[trace].update(dict(
                x=rounded(x, places), y=rounded(y, places),
                z=rounded(z, places),
                surfacecolor=rounded(colour, colour_decimals(spec, params)),
                colorscale=spec.colorscale, cmin=spec.cmin, cmax=spec.cmax,
                opacity=opaque if position == 0 else faint, visible=True,
                name=('this slice' if position == 0
                      else f'{cuts[which].position_mm:+.2f} mm')))

        live.handle.data[live.panel_trace].update(dict(
            x=volume.grid.coordinate(plane.horizontal),
            y=volume.grid.coordinate(plane.vertical),
            z=rounded(heatmap_values(active.values[name], active.tissue,
                                     active_spec),
                      colour_decimals(active_spec, params)),
            colorscale=active_spec.colorscale, zmin=active_spec.cmin,
            zmax=active_spec.cmax,
            colorbar=dict(title=dict(text=overlay.label, side='right'),
                          tickvals=active_spec.tickvals,
                          ticktext=active_spec.ticktext, len=0.75),
            name=overlay.label,
            hovertemplate=(f'{plane.horizontal_label}: %{{x:.3f}}<br>'
                           f'{plane.vertical_label}: %{{y:.3f}}<br>'
                           f'{overlay.label}: %{{z:.4g}}<extra></extra>')))

        _update_electrodes(live, volume, view, near)
        _update_axes(live, plane)
        live.handle.layout.title = dict(
            text=_title(volume, view, active), x=0.5, xanchor='center')

    return Drawn(active=active, stack=view.stack, n_contacts=len(near),
                 seconds=time.perf_counter() - started,
                 off_tissue=active.n_voxels == 0)


def _update_electrodes(live: LiveFigure, volume: Volume, view: View,
                       near) -> None:
    """The contacts this slice cuts, in the scene and on the panel."""
    electrodes = volume.electrodes
    if electrodes is None or live.scene_near_trace is None:
        return
    params = volume.params
    size = float(require(params, 'visualization.electrodes.marker_size'))
    slab = view.electrode_slab_mm
    hover = [f'{electrodes.hover[row]}<br>'
             f'{view.stack.plane.normal_label} offset: '
             f'{distance * 1e3:.0f} um from this plane'
             for row, distance in zip(near.rows, near.distance_mm)]

    live.handle.data[live.scene_near_trace].update(dict(
        x=electrodes.positions_mm[near.rows, 0],
        y=electrodes.positions_mm[near.rows, 1],
        z=electrodes.positions_mm[near.rows, 2],
        marker=_electrode_marker(params, electrodes, near.rows, size * 1.3,
                                 1.0),
        customdata=hover, name=f'in this slice (±{slab * 1e3:.0f} um)'))

    if live.panel_near_trace is not None:
        colours = group_colours(params, len(electrodes.group_names))
        live.handle.data[live.panel_near_trace].update(dict(
            x=near.horizontal_mm, y=near.vertical_mm,
            marker=dict(size=size, symbol='diamond',
                        color=[colours[int(electrodes.group[row])]
                               for row in near.rows],
                        line=dict(width=1, color='white')),
            customdata=hover))


def _update_axes(live: LiveFigure, plane: Plane) -> None:
    """The panel's labels, its equal-millimetre aspect and its
    revision.

    The aspect anchor is asked of the trace rather than assumed: a
    scene in the first subplot cell consumes no x axis, so the panel
    beside it is the figure's FIRST xy subplot and its axis is 'x', not
    'x2'. Anchoring to an axis that does not exist leaves the panel
    blank -- which it did, once.
    """
    anchor = live.handle.data[live.panel_trace].xaxis or 'x'
    live.handle.layout.xaxis.update(
        title=dict(text=plane.horizontal_label, font=dict(size=10)),
        constrain='domain', uirevision=plane.name)
    live.handle.layout.yaxis.update(
        title=dict(text=plane.vertical_label, font=dict(size=10)),
        scaleanchor=anchor, scaleratio=1, constrain='domain',
        uirevision=plane.name)


def _title(volume: Volume, view: View, active: Slice) -> str:
    """Plane, position and what is painted on it."""
    plane = view.stack.plane
    overlay = volume.overlays[view.overlay]
    stack = ('' if len(view.stack) < 2
             else f' · {len(view.stack)} planes, '
                  f'{view.stack.spacing_mm:.3f} mm apart')
    return (f'{plane.name}, {plane.normal_label} = '
            f'{active.position_mm:+.3f} mm{stack}<br>'
            f'<sub>{overlay.label} — {volume.banner}</sub>')


def status_text(volume: Volume, view: View, drawn: Drawn) -> str:
    """One line saying what is actually on screen.

    Reports the achieved spacing and the placed plane count rather than
    the requested ones, and says when a position has landed somewhere
    with no tissue -- all three are states the controls cannot show.
    """
    plane = view.stack.plane
    parts = [f'{drawn.active.n_voxels:,} voxels in this {plane.name} plane']
    if len(view.stack) > 1:
        parts.append(f'{len(view.stack)} planes at '
                     f'{view.stack.spacing_mm:.3f} mm')
    if view.stack.dropped:
        parts.append(f'{view.stack.dropped} outside the volume')
    if volume.electrodes is not None and len(volume.electrodes):
        parts.append(f'{drawn.n_contacts} contact(s) within '
                     f'{view.electrode_slab_mm * 1e3:.0f} um')
    parts.append(f'redrawn in {drawn.seconds * 1e3:.0f} ms')

    note = ''
    if drawn.off_tissue:
        note = (' &mdash; <b>no tissue at this position</b>'
                if plane.name != 'sagittal' or
                column_at(volume.grid, drawn.active.position_mm) is not None
                else ' &mdash; <b>this ML is between the two nuclei</b>')
    return ' &middot; '.join(parts) + note


# ======================================================================
# Exporting the settings
# ======================================================================
#: Distinguishes "absent from the baseline" from "present and None".
#: Not a number about the LGN, so not a config key.
_MISSING = object()


def _same(was: Any, now: Any) -> bool:
    """Whether a config leaf is unchanged.

    Compares across the types a round trip through YAML and a widget
    introduces: an int where the file had a float, a numpy scalar where
    it had a plain one, a list where it had a tuple. Without this every
    slider reports a change the moment it is touched, even back to its
    own starting value.
    """
    if was is _MISSING:
        return False
    if isinstance(was, np.generic):
        was = was.item()
    if isinstance(now, np.generic):
        now = now.item()
    if isinstance(was, (list, tuple)) and isinstance(now, (list, tuple)):
        return (len(was) == len(now)
                and all(_same(a, b) for a, b in zip(was, now)))
    if isinstance(was, bool) or isinstance(now, bool):
        return bool(was) is bool(now)
    if isinstance(was, (int, float)) and isinstance(now, (int, float)):
        return float(was) == float(now)
    return was == now


def config_changes(baseline: Mapping, current: Mapping,
                   prefix: str = '') -> Dict[str, Any]:
    """Every leaf of `current` that differs from `baseline`.

    :return: {dotted path: the current value}, walked in the order the
        current config holds them, so a fragment built from it reads in
        the same order as the file it came from.
    """
    found: Dict[str, Any] = {}
    known = isinstance(baseline, Mapping)
    for key, value in current.items():
        path = f'{prefix}{key}'
        under = baseline[key] if known and key in baseline else _MISSING
        if isinstance(value, Mapping):
            found.update(config_changes(
                under if isinstance(under, Mapping) else {}, value,
                f'{path}.'))
        elif not _same(under, value):
            found[path] = value
    return found


def nest(changes: Mapping) -> dict:
    """Dotted paths back into the nested shape a config file has."""
    out: dict = {}
    for path, value in changes.items():
        node = out
        keys = path.split('.')
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return out


def _plain(value: Any) -> Any:
    """Numpy scalars and arrays down to what `yaml.safe_dump` accepts.

    A numpy float reaching the dumper does not raise -- it writes a
    `!!python/object` tag, producing a file that `safe_load` then
    refuses. The failure is at read time and somewhere else, so the
    conversion happens here.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def fragment_yaml(changes: Mapping) -> str:
    """The changed settings as a pasteable YAML fragment.

    Only the branches that differ, nested as the file nests them, so it
    can be merged into `params_lgn.yaml` by hand. The file's comments
    are load-bearing documentation in this package and a full dump
    would lose all of them, which is why the fragment is what `save`
    writes by default.
    """
    import yaml
    if not changes:
        return ''
    return yaml.safe_dump(_plain(nest(changes)), sort_keys=False,
                          default_flow_style=False, allow_unicode=True)


def _flag(name: str, value: Any) -> List[str]:
    """One flag and its value, as tokens `argparse` will accept.

    A value that starts with '-' cannot travel as a separate token:
    argparse reads anything option-shaped as the next option and then
    reports the flag as missing its argument. Positions here are
    signed -- ML is negative on the left nucleus and DV starts below
    the origin -- so `--range` meets this on two of the three planes.
    The '=' form is unambiguous, and `shlex.split` keeps it in one
    piece.

    Nothing is quoted here. These are raw tokens, ready to hand
    straight to `slices.main`; quoting for a shell is
    `ExportedSettings.command`'s job, and doing it twice breaks both.
    """
    text = str(value)
    return [f'{name}={text}'] if text.startswith('-') else [name, text]


def reproduce_command(params: Mapping, baseline: Mapping,
                      mask_options: Mapping, provenance: Mapping,
                      stack: Stack, overlays: Sequence[str],
                      view: str) -> List[str]:
    """The `slices.py` command line that renders the current picture.

    Restates as flags everything that has one, so the command works on
    its own -- the YAML fragment is the durable record, not a
    prerequisite. The two never disagree: `viz3d._apply_overrides` and
    `slices._apply_slice_overrides` fold each of these flags into
    exactly the config key the explorer wrote it to.

    What has no flag and no config key is the tissue selection, which
    is why `--cell-class`, `--layers`, `--ecc-min/max` and `--clip` are
    rendered from `mask_options` rather than diffed out of the config.

    :param provenance: How the explorer was opened -- atlas source and
        electrode options -- which cannot be recovered from the config.
    :return: The command's own arguments as raw, unquoted tokens.
        `ExportedSettings.tokens` adds `--params` when one is needed;
        `ExportedSettings.command` quotes them for a shell.
    """
    argv: List[str] = ['--view', view, '--plane', stack.plane.name]

    if provenance.get('synthetic'):
        argv.append('--synthetic')
    if provenance.get('atlas_dir'):
        # Absolute: the explorer's path is relative to the notebook's
        # own directory, and the command it writes is run from the
        # project root, where that same relative path means elsewhere.
        argv += ['--atlas-dir',
                 str(Path(provenance['atlas_dir']).resolve())]
    hemispheres = require(params, 'atlas.hemispheres')
    if not _same(require(baseline, 'atlas.hemispheres'), hemispheres):
        argv += ['--hemispheres'] + list(hemispheres)

    argv += ['--overlay'] + list(overlays)
    argv += ['--count', str(len(stack))]
    argv += _flag('--range', '{:.4f}:{:.4f}'.format(
        *require(params, 'visualization.slices.range_mm')))

    stride = require(params, 'visualization.stride_vox')
    if not _same(require(baseline, 'visualization.stride_vox'), stride):
        argv += _flag('--stride', int(stride))
    slab = require(params, 'visualization.slices.electrode_slab_mm')
    if not _same(require(baseline, 'visualization.slices.electrode_slab_mm'),
                 slab):
        argv += _flag('--slab', f'{float(slab):g}')

    if mask_options.get('cell_class'):
        argv += ['--cell-class', mask_options['cell_class']]
    if mask_options.get('layers'):
        argv += ['--layers'] + [str(int(code))
                                for code in mask_options['layers']]
    if mask_options.get('min_eccentricity_deg') is not None:
        argv += _flag('--ecc-min',
                      f"{mask_options['min_eccentricity_deg']:g}")
    if mask_options.get('max_eccentricity_deg') is not None:
        argv += _flag('--ecc-max',
                      f"{mask_options['max_eccentricity_deg']:g}")
    for axis, lo, hi in (mask_options.get('clips') or ()):
        argv += _flag('--clip', f'{axis}:{lo:g}:{hi:g}')
    if mask_options.get('include_ipsi_placeholder'):
        argv.append('--include-ipsi')

    argv += _electrode_argv(params, baseline, provenance)
    return argv


def _electrode_argv(params: Mapping, baseline: Mapping,
                    provenance: Mapping) -> List[str]:
    """The electrode flags the explorer was opened with."""
    argv: List[str] = []
    source = provenance.get('electrodes', 'none')
    if source != 'none':
        argv += ['--electrodes', source]
    n_electrodes = require(params, 'electrodes.n_electrodes')
    if not _same(require(baseline, 'electrodes.n_electrodes'), n_electrodes):
        argv += _flag('--n-electrodes', int(n_electrodes))
    if provenance.get('electrodes_file'):
        argv += ['--electrodes-file', str(provenance['electrodes_file']),
                 '--electrodes-units', provenance.get('electrodes_units',
                                                      'mm')]
    for spec in (provenance.get('probe') or []):
        argv += ['--probe', str(spec)]
    for spec in (provenance.get('grid') or []):
        argv += ['--grid', str(spec)]
    return argv


@dataclass
class ExportedSettings:
    """What the explorer is showing, as configuration you can keep.

    Two forms of the same state, and they are not alternatives:

    - `fragment` is the durable record -- every config value that
      differs from the file on disk, nested so it can be merged back
      into `params_lgn.yaml` by hand.
    - `command` renders the picture as a static figure. It restates as
      flags everything that has one, so it needs nothing else in the
      ordinary case.

    :ivar needs_params_file: True when the command alone will NOT
        reproduce the picture, because the slices it asks for lie in
        the tapered ends of the nucleus that automatic placement trims
        away. Only the config can lift that trimming, so `save` has to
        write a full parameter file and the command has to name it.
    """

    changes: Dict[str, Any]
    fragment: str
    argv: List[str]
    params: Mapping
    directory: Path
    fragment_name: str
    full_name: str
    needs_params_file: bool
    saved: Optional[Path] = None

    @property
    def tokens(self) -> List[str]:
        """The arguments, as tokens to hand straight to `slices.main`.

        Prefer these over splitting `command` back up. `shlex.split`
        is POSIX by default, so it reads the backslashes in a bare
        Windows path as escapes and quietly hands back a different
        path; and only this knows when `--params` has to be appended.
        """
        tokens = list(self.argv)
        if self.saved is not None and self.saved.name == self.full_name:
            tokens += ['--params', str(self.saved)]
        elif self.needs_params_file:
            tokens += ['--params', str(self.directory / self.full_name)]
        return tokens

    @property
    def command(self) -> str:
        """The command line, as one runnable string.

        `tokens`, quoted for a shell. Quoting happens here and nowhere
        else: the tokens themselves stay raw, because a token quoted
        twice is a token that works in neither place.
        """
        import shlex
        parts = ['python', '-m', 'dynaphos_lgn.slices'] + self.tokens
        return ' '.join(shlex.quote(part) if ' ' in part else part
                        for part in parts)

    @property
    def text(self) -> str:
        """Both forms, as one valid YAML document.

        The command goes in a comment rather than beside the settings.
        This is what `save` writes, under a `.yaml` name, and a file
        that announces itself as YAML and then is not is a worse trade
        than a command line you have to uncomment.
        """
        lines = ['# LGN explorer settings.',
                 '#',
                 '# The same picture as a static figure:',
                 f'#   {self.command}']
        if self.needs_params_file:
            lines += ['#',
                      '# That position is inside the tapered end of the '
                      'nucleus, which automatic',
                      '# slice placement trims away and no flag can '
                      'un-trim, so the command needs',
                      '# these settings as a whole parameter file:',
                      '#   Explorer.export().save(full=True)']
        lines += ['#',
                  '# Merge the block below into config/params_lgn.yaml.',
                  '']
        lines.append(self.fragment.rstrip() if self.fragment
                     else '# (nothing differs from the shipped '
                          'configuration)')
        return '\n'.join(lines)

    def save(self, path: Optional[Path] = None, full: bool = False) -> Path:
        """Write the settings out.

        :param path: Where. Relative paths land under
            `visualization.output.directory`; the default name comes
            from the config too.
        :param full: Write the complete parameter dictionary rather
            than the fragment, giving a file `--params` can be pointed
            at. It loses every comment in `params_lgn.yaml`, which in
            this package carry most of the reasoning, so it is for
            feeding back to the program rather than for reading.
        """
        import yaml
        name = self.full_name if full else self.fragment_name
        out = Path(path) if path is not None else self.directory / name
        if not out.is_absolute():
            out = self.directory / out
        out.parent.mkdir(parents=True, exist_ok=True)
        if full:
            out.write_text(yaml.safe_dump(_plain(self.params),
                                          sort_keys=False,
                                          default_flow_style=False,
                                          allow_unicode=True),
                           encoding='utf-8')
        else:
            out.write_text(self.text + '\n', encoding='utf-8')
        self.saved = out
        return out

    def __str__(self) -> str:                          # pragma: no cover
        return self.text


# ======================================================================
# The explorer
# ======================================================================
class Explorer:
    """The notebook viewer: controls, figure, and the settings behind
    both.

    The widgets hold no state that matters. Every control writes its
    value into `self.params` at the dotted path the rest of the package
    reads it from, and the figure is drawn from there -- so `export` is
    a diff against the file on disk rather than a second inventory of
    what the controls happen to say, and the config keeps being the one
    place a number lives.

    The one exception is the tissue selection, which has command-line
    flags but no config keys; it lives in `self.mask_options` and is
    rendered into the exported command.

    Controls split in two, because the costs do:

    - **live** -- position, count, spacing, plane, overlay, slab,
      opacity, stride -- redraw one stack of planes, which is
      milliseconds;
    - **structural** -- cell class, laminae, eccentricity, shell stride
      -- rebuild the masks, the scene grid, the shell mesh and the
      colour sample, which is seconds. Those sit behind an Apply
      button rather than firing on every keystroke.
    """

    def __init__(self, params: dict, baseline: Mapping, atlases: Mapping,
                 jacobian_for: Optional[Callable[[str], object]],
                 overlays: Sequence[str],
                 mask_options: Optional[Mapping] = None,
                 electrodes: Optional[ElectrodeSet] = None,
                 banner: str = '',
                 provenance: Optional[Mapping] = None,
                 repository: Optional[Path] = None):
        self.params = params
        self.baseline = baseline
        self.atlases = dict(atlases)
        self.jacobian_for = jacobian_for
        self.offered = list(overlays)
        self.mask_options = dict(mask_options or {})
        self.electrodes = electrodes
        self.banner = banner
        self.provenance = dict(provenance or {})
        self.repository = Path(repository or
                               Path(__file__).resolve().parents[1])

        self.volume = build_volume(params, self.atlases, jacobian_for,
                                   self.offered, self.mask_options,
                                   electrodes, banner)
        self.live = build_live_figure(self.volume, params)
        self._drawing = False
        self._positions: Dict[str, float] = {}
        self._build_controls()
        self.refresh()

    # -- construction --------------------------------------------------
    @classmethod
    def open(cls, params_path: Optional[Path] = None,
             atlas_dir: Optional[str] = None, synthetic: bool = False,
             hemispheres: Optional[Sequence[str]] = None,
             overlays: Optional[Sequence[str]] = None,
             electrodes: str = 'none',
             n_electrodes: Optional[int] = None,
             electrodes_file: Optional[str] = None,
             electrodes_units: str = 'mm',
             probe: Optional[Sequence[str]] = None,
             grid: Optional[Sequence[str]] = None,
             recompute_jacobian: bool = False,
             **mask_options) -> 'Explorer':
        """Load the config and the atlas, and open the viewer.

        The same steps `slices.main` takes, in the same order and
        through the same builders, so what the explorer shows is what a
        run of that command would draw.

        :param overlays: Which overlays the dropdown offers. Defaults
            to `visualization.overlays`; 'all' offers every one,
            including the seven that need the Jacobian cache.
        :param electrodes: 'none', 'layout' or 'array', as
            ``--electrodes``.
        :param mask_options: Passed to `viz3d.tissue_mask` -- the
            starting tissue selection, which the controls can then
            change.
        """
        from dynaphos.utils import load_params
        from dynaphos_lgn.build import (build_atlas, build_hemisphere_atlas,
                                        configured_hemispheres)
        import copy

        repository = Path(__file__).resolve().parents[1]
        path = Path(params_path or repository / 'config' / 'params_lgn.yaml')
        params = load_params(str(path))
        baseline = copy.deepcopy(params)
        if hemispheres:
            params['atlas']['hemispheres'] = list(hemispheres)
        if n_electrodes is not None:
            params['electrodes']['n_electrodes'] = int(n_electrodes)

        names = (list(OVERLAY_BUILDERS) if overlays == 'all'
                 else list(overlays or require(params,
                                               'visualization.overlays')))
        unknown = [name for name in names if name not in OVERLAY_BUILDERS]
        if unknown:
            raise KeyError(f"Unknown overlay(s) {', '.join(unknown)}.\n"
                           f"{list_overlays()}")

        atlas = build_atlas(params, atlas_dir, synthetic)
        atlases = {name: build_hemisphere_atlas(params, atlas, name)
                   for name in configured_hemispheres(params)}

        needs_jacobian = (bool(set(names) & JACOBIAN_OVERLAYS)
                          or electrodes == 'array')
        provide = (jacobian_provider(params, atlas, atlas_dir,
                                     recompute_jacobian)
                   if needs_jacobian else None)

        options = SimpleNamespace(
            electrodes=electrodes, electrodes_file=electrodes_file,
            electrodes_units=electrodes_units, probe=list(probe or []),
            grid=list(grid or []))
        rng = np.random.default_rng(require(params, 'run.seed'))
        contacts = collect_electrodes(params, atlases, options, rng, provide)

        return cls(params=params, baseline=baseline, atlases=atlases,
                   jacobian_for=provide, overlays=names,
                   mask_options=mask_options, electrodes=contacts,
                   banner=atlas_banner(atlas),
                   provenance=dict(synthetic=synthetic, atlas_dir=atlas_dir,
                                   electrodes=electrodes,
                                   electrodes_file=electrodes_file,
                                   electrodes_units=electrodes_units,
                                   probe=list(probe or []),
                                   grid=list(grid or [])),
                   repository=repository)

    # -- the widgets ---------------------------------------------------
    def _build_controls(self) -> None:
        """Every control, and where each one writes.

        Built from the config's starting values, so a changed config
        opens on a different view without touching this code.
        """
        widgets = _ipywidgets()
        params = self.params
        plane_name = require(params, 'visualization.explorer.plane')
        continuous = bool(require(params,
                                  'visualization.explorer.continuous_update'))
        bound = require(params, 'visualization.explorer.limits')
        voxel = self.volume.grid.voxel_size_mm
        width = require(params, 'visualization.explorer.controls_width')
        wide = widgets.Layout(width=width)
        # The controls sit in a flex row beside the figure, and a flex
        # item shrinks to fit by default. Pinned, because the column
        # losing 200 px costs the sliders their track and not their
        # label: they stop being draggable while still looking present,
        # which is worse than the row overflowing and scrolling.
        column = widgets.Layout(width=width, flex='0 0 auto')
        full = dict(layout=wide, style={'description_width': require(
            params, 'visualization.explorer.description_width')})

        self.overlay_control = widgets.Dropdown(
            options=[(self.volume.overlays[name].label, name)
                     for name in self.offered],
            value=self.offered[0], description='Overlay', **full)
        self.plane_control = widgets.ToggleButtons(
            options=[(name.capitalize(), name) for name in
                     ('coronal', 'sagittal', 'horizontal')],
            value=plane_name, description='Plane', **full)

        plane = PLANES[plane_name]
        coordinate = self.volume.grid.coordinate(plane.normal)
        self.position_control = widgets.FloatSlider(
            value=fullest_plane_mm(self.volume, plane),
            min=float(coordinate[0]), max=float(coordinate[-1]),
            step=voxel, readout_format='+.3f',
            description=f'{plane.normal_label} (mm)',
            continuous_update=continuous, **full)

        self.count_control = widgets.IntSlider(
            value=int(require(params, 'visualization.explorer.count')),
            min=1, max=int(require(params,
                                   'visualization.explorer.max_planes')),
            description='Planes', continuous_update=continuous, **full)
        # One atlas voxel is the smallest spacing that still gives
        # distinct planes, and it is the grid's own number rather than
        # a config one so that it follows the atlas.
        self.spacing_control = widgets.FloatSlider(
            value=float(require(params,
                                'visualization.explorer.spacing_mm')),
            min=voxel, max=float(bound['spacing_max_mm']), step=voxel,
            readout_format='.3f', description='Spacing (mm)',
            continuous_update=continuous, **full)
        self.slab_control = widgets.FloatSlider(
            value=float(require(params,
                                'visualization.slices.electrode_slab_mm')),
            min=voxel, max=float(bound['slab_max_mm']),
            step=float(bound['slab_step_mm']),
            readout_format='.3f', description='Slab (mm)',
            continuous_update=continuous,
            disabled=not (self.electrodes is not None and len(self.electrodes)),
            **full)
        self.opacity_control = widgets.FloatSlider(
            value=float(require(params,
                                'visualization.combined.shell_opacity')),
            min=0.0, max=1.0, step=float(bound['shell_opacity_step']),
            readout_format='.2f', description='Shell',
            continuous_update=continuous, **full)
        self.plane_stride_control = widgets.IntSlider(
            value=int(require(params,
                              'visualization.explorer.plane_stride_vox')),
            min=1, max=int(bound['plane_stride_max_vox']),
            description='Plane stride', continuous_update=continuous, **full)

        live = [self.overlay_control, self.plane_control,
                self.position_control, self.count_control,
                self.spacing_control, self.slab_control,
                self.opacity_control, self.plane_stride_control]
        for control in live:
            control.observe(self._on_live_change, names='value')

        self.status = widgets.HTML(layout=widgets.Layout(width='100%'))
        self.message = widgets.HTML(layout=widgets.Layout(width='100%'))

        tissue = self._build_tissue_controls(widgets, wide, full)
        export = self._build_export_controls(widgets, wide)
        sections = widgets.Accordion(children=[tissue, export])
        sections.set_title(0, 'Tissue selection (rebuilds)')
        sections.set_title(1, 'Export these settings')
        sections.selected_index = None
        sections.layout = wide

        self.controls = widgets.VBox(live + [sections], layout=column)
        self.widget = widgets.VBox([
            widgets.HBox([self.controls, self.live.handle]),
            self.status, self.message])

    def _build_tissue_controls(self, widgets, wide, full):
        """The selection controls, behind an Apply button.

        Each of these changes which voxels exist, so applying one
        rebuilds the masks, the scene grid, the shell mesh and the
        volume-wide colour sample. That is seconds, not milliseconds,
        which is why it is a button and not an observer.
        """
        names = require(self.params, 'atlas.layers.names')
        classes = require(self.params, 'atlas.layers.classes')
        bound = require(self.params, 'visualization.explorer.limits')
        ceiling = self.volume.eccentricity_deg_max

        self.cell_class_control = widgets.Dropdown(
            options=[('all', None)] + [(name, name) for name in classes],
            value=self.mask_options.get('cell_class'),
            description='Cell class', **full)
        self.layers_control = widgets.SelectMultiple(
            options=[(f'{code}  {label}', int(code))
                     for code, label in sorted(names.items())],
            value=tuple(self.mask_options.get('layers') or ()),
            rows=int(bound['layer_rows']), description='Laminae', **full)
        self.ecc_control = widgets.FloatRangeSlider(
            value=[float(self.mask_options.get('min_eccentricity_deg') or 0.0),
                   float(self.mask_options.get('max_eccentricity_deg')
                         or ceiling)],
            min=0.0, max=ceiling,
            step=float(bound['eccentricity_step_deg']), readout_format='.1f',
            description='Ecc (deg)', **full)
        self.stride_control = widgets.IntSlider(
            value=int(require(self.params, 'visualization.stride_vox')),
            min=1, max=int(bound['shell_stride_max_vox']),
            description='Shell stride', **full)
        apply_button = widgets.Button(description='Apply selection',
                                      icon='refresh', layout=wide)
        apply_button.on_click(lambda _: self.apply_selection())
        return widgets.VBox([self.cell_class_control, self.layers_control,
                             self.ecc_control, self.stride_control,
                             apply_button], layout=wide)

    def _build_export_controls(self, widgets, wide):
        """Show the settings, and write them out."""
        self.export_box = widgets.Textarea(
            layout=widgets.Layout(width='100%', height='240px'))
        show = widgets.Button(description='Show settings', icon='eye',
                              layout=wide)
        save_fragment = widgets.Button(description='Save fragment',
                                       icon='save', layout=wide)
        save_full = widgets.Button(description='Save full params file',
                                   icon='save', layout=wide)
        show.on_click(lambda _: self._show_export())
        save_fragment.on_click(lambda _: self._save_export(full=False))
        save_full.on_click(lambda _: self._save_export(full=True))
        return widgets.VBox([show, self.export_box, save_fragment, save_full],
                            layout=wide)

    # -- behaviour -----------------------------------------------------
    def _on_live_change(self, change) -> None:
        """One handler for every live control.

        Switching plane moves the position control onto a different
        axis, which fires it in turn; the guard stops that second
        change from drawing a view assembled half from each plane.
        """
        if self._drawing:
            return
        if change['owner'] is self.plane_control:
            self._retarget_position(PLANES[change['new']])
        self.refresh()

    def _retarget_position(self, plane: Plane) -> None:
        """Point the position control at a new plane's axis.

        The three cardinal planes measure position along three
        different axes spanning three different millimetre ranges, and
        a slider cannot hold a value outside its own bounds. Widened in
        both directions first, then moved, then narrowed: setting the
        new bounds directly fails outright when the two ranges do not
        overlap -- sagittal ML and coronal AP do not -- and setting the
        value first silently clamps it to an end, opening the view on
        the edge of the nucleus.

        Where you last were on a plane is remembered, because the
        planes' positions are not comparable and carrying one over
        would mean an arbitrary jump every time you switched.
        """
        coordinate = self.volume.grid.coordinate(plane.normal)
        low, high = float(coordinate[0]), float(coordinate[-1])
        wanted = self._positions.get(plane.name)
        if wanted is None or not low <= wanted <= high:
            wanted = fullest_plane_mm(self.volume, plane)

        self._drawing = True
        try:
            self.position_control.min = min(low, self.position_control.min)
            self.position_control.max = max(high, self.position_control.max)
            self.position_control.value = wanted
            self.position_control.min = low
            self.position_control.max = high
            self.position_control.description = f'{plane.normal_label} (mm)'
        finally:
            self._drawing = False

    def view(self) -> View:
        """The `View` the controls currently describe."""
        plane = PLANES[self.plane_control.value]
        return View(
            overlay=self.overlay_control.value,
            stack=stack_indices(self.volume.grid, plane,
                                float(self.position_control.value),
                                int(self.count_control.value),
                                float(self.spacing_control.value)),
            plane_stride_vox=int(self.plane_stride_control.value),
            electrode_slab_mm=float(self.slab_control.value),
            shell_opacity=float(self.opacity_control.value))

    def refresh(self) -> Drawn:
        """Write the controls into the config, then draw from it.

        The config is written first and read back, so the picture and
        the export can never disagree about what is being shown.
        """
        view = self.view()
        self._positions[view.stack.plane.name] = float(
            self.position_control.value)
        self._record(view)
        drawn = draw(self.live, self.volume, view)
        self.status.value = status_text(self.volume, view, drawn)
        return drawn

    def _record(self, view: View) -> None:
        """Fold the live controls into the parameter dictionary.

        The same keys `slices.py` reads, so the explorer's state IS a
        configuration rather than a description of one. `range_mm` is
        the span the stack occupies and `count` its size, which is how
        `slices.slice_indices` reproduces exactly these planes.

        `min_slice_fraction` goes to zero alongside it: it exists to
        keep AUTOMATIC placement off the tapered ends of the nucleus,
        and an explicit range is the user saying where the slices go.
        """
        visualization = self.params['visualization']
        coordinate = self.volume.grid.coordinate(view.stack.plane.normal)
        visualization['slices']['plane'] = view.stack.plane.name
        visualization['combined']['plane'] = view.stack.plane.name
        visualization['slices']['count'] = len(view.stack)
        # Rounded to the same precision the figure is drawn at: the
        # grid's own coordinates carry the arithmetic that built them,
        # and `8.350000000000001` in an exported config is noise. A
        # tenth of a micrometre cannot move which voxel
        # `slices.slice_indices` then snaps to.
        places = int(require(self.params,
                             'visualization.explorer.coordinate_decimals'))
        visualization['slices']['range_mm'] = [
            round(float(coordinate[view.stack.indices[0]]), places),
            round(float(coordinate[view.stack.indices[-1]]), places)]
        visualization['slices']['min_slice_fraction'] = 0.0
        visualization['combined']['min_slice_fraction'] = 0.0
        visualization['slices']['electrode_slab_mm'] = view.electrode_slab_mm
        visualization['combined']['shell_opacity'] = view.shell_opacity
        visualization['combined']['plane_stride_vox'] = view.plane_stride_vox
        visualization['explorer']['plane'] = view.stack.plane.name
        visualization['explorer']['count'] = int(view.stack.requested)
        visualization['explorer']['spacing_mm'] = float(
            self.spacing_control.value)
        visualization['explorer']['plane_stride_vox'] = view.plane_stride_vox
        visualization['overlays'] = (
            [view.overlay] + [name for name in self.offered
                              if name != view.overlay])

    def apply_selection(self) -> None:
        """Rebuild the volume for the current tissue controls."""
        ceiling = self.volume.eccentricity_deg_max
        low, high = (float(value) for value in self.ecc_control.value)
        self.mask_options = dict(
            cell_class=self.cell_class_control.value,
            layers=list(self.layers_control.value) or None,
            min_eccentricity_deg=low if low > 0.0 else None,
            max_eccentricity_deg=high if high < ceiling else None,
            clips=self.mask_options.get('clips'),
            include_ipsi_placeholder=self.mask_options.get(
                'include_ipsi_placeholder'))
        self.params['visualization']['stride_vox'] = int(
            self.stride_control.value)

        self.message.value = '<i>rebuilding &hellip;</i>'
        try:
            self.volume = build_volume(
                self.params, self.atlases, self.jacobian_for, self.offered,
                self.mask_options, self.electrodes, self.banner)
        except ValueError as error:
            self.message.value = (f'<b style="color:#b00">{error}</b> '
                                  'The previous selection is still shown.')
            return

        # The grid has moved, so the position control's axis has too.
        # Rebuilt before the figure, for the reason `_retarget_position`
        # gives: a stale bound clamps the value it is then given.
        plane = PLANES[self.plane_control.value]
        self._retarget_position(plane)
        self.live = build_live_figure(self.volume, self.params)
        self.widget.children[0].children = (self.controls, self.live.handle)
        self.message.value = (f'<i>{self.volume.selection}</i>')
        self.refresh()

    # -- export --------------------------------------------------------
    def export(self) -> ExportedSettings:
        """The current settings, as configuration and as a command."""
        view = self.view()
        self._record(view)
        directory = Path(require(self.params,
                                 'visualization.output.directory'))
        if not directory.is_absolute():
            directory = self.repository / directory
        section = require(self.params,
                          'visualization.explorer.export.command_view')
        changes = config_changes(self.baseline, self.params)

        argv = reproduce_command(
            self.params, self.baseline, self.mask_options, self.provenance,
            view.stack, require(self.params, 'visualization.overlays'),
            section)
        return ExportedSettings(
            changes=changes, fragment=fragment_yaml(changes), argv=argv,
            params=self.params, directory=directory,
            fragment_name=require(
                self.params, 'visualization.explorer.export.fragment_name'),
            full_name=require(
                self.params, 'visualization.explorer.export.full_name'),
            needs_params_file=self._needs_params_file(view.stack, section))

    def _needs_params_file(self, stack: Stack, section: str) -> bool:
        """Whether the exported command can stand on its own.

        `slices.slice_indices` intersects an explicit range with the
        span automatic placement would have used, which trims the
        tapered ends of the nucleus. A position inside those ends is
        perfectly drawable here and would be dropped there, and no flag
        lifts the trimming -- only the config does. Reported rather
        than discovered when the command turns out to draw the wrong
        slice.
        """
        fraction = float(require(
            self.baseline, f'visualization.{section}.min_slice_fraction'))
        try:
            low, high = occupied_range(self.volume.grid, self.volume.masks,
                                       stack.plane, fraction)
        except ValueError:
            return True
        return not (low <= stack.indices[0] and stack.indices[-1] <= high)

    def _show_export(self) -> None:
        self.export_box.value = self.export().text

    def _save_export(self, full: bool) -> None:
        exported = self.export()
        path = exported.save(full=full)
        self.export_box.value = exported.text
        self.message.value = f'<i>wrote {path}</i>'

    def _repr_mimebundle_(self, **kwargs):             # pragma: no cover
        """Displaying the explorer displays its widget."""
        return self.widget._repr_mimebundle_(**kwargs)


def _ipywidgets():
    """ipywidgets, or a message saying how to get it."""
    try:
        import ipywidgets
    except ImportError as error:                       # pragma: no cover
        raise ImportError(
            "The explorer needs ipywidgets: `pip install ipywidgets`. "
            "For a static figure instead, which needs nothing further, "
            "use `python -m dynaphos_lgn.slices --view combined`."
        ) from error
    return ipywidgets


# ======================================================================
# Command line
# ======================================================================
REQUIREMENTS = ('ipywidgets', 'plotly', 'anywidget')


def check_environment() -> Tuple[List[str], List[str]]:
    """Which of the explorer's dependencies are present.

    :return: (found, missing), both as package names.
    """
    from importlib.util import find_spec
    found, missing = [], []
    for name in REQUIREMENTS:
        (found if find_spec(name) is not None else missing).append(name)
    return found, missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m dynaphos_lgn.explore',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list-overlays', action='store_true',
                        help='List the overlays the dropdown can offer.')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Check the environment and say how to open the explorer.

    There is nothing to run from a terminal: the explorer IS a
    notebook widget, and its whole point is the kernel it keeps. What
    a command line can usefully do is tell you whether the pieces are
    installed before you find out inside a cell.
    """
    options = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if options.verbose
                        else logging.WARNING,
                        format='%(levelname)s: %(message)s')
    if options.list_overlays:
        print(list_overlays())
        return 0

    found, missing = check_environment()
    print('The LGN explorer is a notebook widget, not a command.\n')
    for name in found:
        print(f'  found   {name}')
    for name in missing:
        print(f'  MISSING {name}')
    if missing:
        print(f"\nInstall them with:\n  pip install {' '.join(missing)}")
        print("\nOr write a static figure instead, which needs none of "
              "them:\n  python -m dynaphos_lgn.slices --view combined")
        return 3

    print('\nOpen it with:\n'
          '  jupyter lab examples/demo_lgn_explorer.ipynb\n'
          '\nor, in any notebook:\n'
          '  from dynaphos_lgn import explore\n'
          '  ex = explore.Explorer.open()\n'
          '  ex.widget')
    return 0


if __name__ == '__main__':                             # pragma: no cover
    sys.exit(main())
