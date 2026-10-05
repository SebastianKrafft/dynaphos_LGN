"""Tests for the live notebook explorer.

The explorer's claim is that what you are looking at and what it
exports are the same thing. Most of what is here checks that: the
controls write into the config, the config is what gets drawn, and the
command line the export hands back really does reproduce the planes on
screen.

The rest covers the three places where what you asked for and what you
got can legitimately differ -- a spacing the 25 um grid cannot express,
a stack that runs off the end of the volume, and a position with no
tissue at it -- because each of those is reported rather than assumed,
and a report that drifts from the truth is worse than none.

Everything runs on the synthetic atlas, so no download is needed.
"""
import copy

import numpy as np
import pytest

from dynaphos_lgn import explore, slices, viz3d
from dynaphos_lgn.test.conftest import requires_real_atlas

pytest.importorskip('plotly')
pytest.importorskip('ipywidgets')
pytest.importorskip('anywidget')


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture
def one_nucleus(synthetic_atlas, lgn_params):
    """A `Volume` over the left LGN, with no electrodes."""
    atlases = {'left': synthetic_atlas}
    return explore.build_volume(lgn_params, atlases, None,
                                ['layer', 'eccentricity'])


@pytest.fixture
def both_nuclei(synthetic_atlas, lgn_params):
    """A `Volume` over both nuclei, which is what makes the signed ML
    axis and its midline gap testable."""
    from dynaphos_lgn.atlas import MirroredAtlas
    atlases = {'left': synthetic_atlas,
               'right': MirroredAtlas(synthetic_atlas, lgn_params)}
    return explore.build_volume(lgn_params, atlases, None, ['layer'])


@pytest.fixture
def explorer(lgn_params):
    """A live explorer on the synthetic atlas, one nucleus.

    Built through `Explorer.open` so the loading path -- config, atlas,
    hemispheres, electrodes -- is exercised too, since that is where
    the explorer and `slices.main` have to agree.
    """
    return explore.Explorer.open(synthetic=True, hemispheres=['left'],
                                 overlays=['layer', 'eccentricity'],
                                 electrodes='layout')


# ======================================================================
# The volume
# ======================================================================
def test_volume_holds_what_a_selection_fixes(one_nucleus):
    volume = one_nucleus
    assert set(volume.masks) == {'left'}
    assert volume.masks['left'].any()
    assert volume.shell and volume.shell[0].n_faces > 0
    assert set(volume.overlays) == {'layer', 'eccentricity'}
    assert volume.limits['layer'] is None            # categorical
    assert volume.limits['eccentricity'] is not None


def test_representative_overlay_has_every_configured_category(
        one_nucleus, lgn_params):
    """One voxel is enough for the label and the categories.

    The categories come from `atlas.layers.names`, not from the sample,
    which is what lets the dropdown and the colourbar be built before
    any plane has been cut.
    """
    names = lgn_params['atlas']['layers']['names']
    assert one_nucleus.overlays['layer'].categories == [
        str(names[code]) for code in sorted(names)]


def test_eccentricity_ceiling_ignores_the_current_selection(
        synthetic_atlas, lgn_params):
    """The filter's range must not shrink to what the filter removed.

    A ceiling taken from the selected tissue would drop every time the
    eccentricity filter was tightened, and there would be no way back.
    """
    atlases = {'left': synthetic_atlas}
    wide = explore.build_volume(lgn_params, atlases, None, ['layer'])
    narrow = explore.build_volume(lgn_params, atlases, None, ['layer'],
                                  dict(max_eccentricity_deg=5.0))
    assert narrow.masks['left'].sum() < wide.masks['left'].sum()
    assert narrow.eccentricity_deg_max == wide.eccentricity_deg_max


def test_a_selection_with_no_tissue_is_refused(synthetic_atlas, lgn_params):
    with pytest.raises(ValueError):
        explore.build_volume({**lgn_params}, {'left': synthetic_atlas}, None,
                             ['layer'],
                             dict(min_eccentricity_deg=1e6))


def test_fullest_plane_is_inside_the_nucleus(both_nuclei):
    """The opening position, for every plane.

    For two nuclei the tissue's centroid lands in the 17 mm gap between
    them, so a sagittal cut there would show nothing at all. The
    fullest plane cannot.
    """
    for name in ('coronal', 'sagittal', 'horizontal'):
        plane = slices.PLANES[name]
        position = explore.fullest_plane_mm(both_nuclei, plane)
        index = both_nuclei.grid.nearest_index(plane.normal, position)
        one = slices.cut(both_nuclei.grid, both_nuclei.masks,
                         both_nuclei.contexts, ['layer'], plane, index)
        assert one.n_voxels > 0, name


# ======================================================================
# The stack
# ======================================================================
def test_one_plane_sits_exactly_where_it_was_asked_for(one_nucleus):
    grid = one_nucleus.grid
    plane = slices.PLANES['coronal']
    wanted = float(grid.z_mm[len(grid.z_mm) // 2])
    stack = explore.stack_indices(grid, plane, wanted, 1, 0.5)
    assert len(stack) == 1
    assert grid.z_mm[stack.active_index] == pytest.approx(wanted)


def test_the_stack_is_evenly_spaced_around_the_active_plane(one_nucleus):
    grid = one_nucleus.grid
    plane = slices.PLANES['coronal']
    centre = float(grid.z_mm[len(grid.z_mm) // 2])
    stack = explore.stack_indices(grid, plane, centre, 5, 0.5)

    assert len(stack) == 5
    steps = np.diff([grid.z_mm[index] for index in stack.indices])
    assert np.allclose(steps, stack.spacing_mm)
    assert grid.z_mm[stack.active_index] == pytest.approx(centre)


def test_an_unrepresentable_spacing_moves_the_others_not_the_active_one(
        one_nucleus):
    """A spacing the 25 um grid cannot express is snapped.

    The active plane stays exactly where the position control put it --
    it is the one being read -- and the achieved spacing is reported
    rather than left as the one that was asked for.
    """
    grid = one_nucleus.grid
    plane = slices.PLANES['coronal']
    centre = float(grid.z_mm[len(grid.z_mm) // 2])
    asked = 0.31                       # 12.4 voxels
    stack = explore.stack_indices(grid, plane, centre, 3, asked)

    assert stack.spacing_mm != pytest.approx(asked)
    assert stack.spacing_mm == pytest.approx(12 * grid.voxel_size_mm)
    assert grid.z_mm[stack.active_index] == pytest.approx(centre)


def test_a_stack_running_off_the_end_drops_planes_rather_than_sliding(
        one_nucleus):
    """Clipped, not shifted, and it says how many it lost.

    A stack that slid inwards to stay whole would no longer be centred
    on the position the control shows.
    """
    grid = one_nucleus.grid
    plane = slices.PLANES['coronal']
    edge = float(grid.z_mm[0])
    stack = explore.stack_indices(grid, plane, edge, 5, 0.5)

    assert grid.z_mm[stack.active_index] == pytest.approx(edge)
    assert len(stack) == 3                       # the two below fall off
    assert stack.dropped == 2
    assert stack.requested == 5


def test_a_spacing_smaller_than_a_voxel_still_gives_distinct_planes(
        one_nucleus):
    """Snapping to at least one voxel is what stops a stack of five
    from collapsing onto one plane drawn five times."""
    grid = one_nucleus.grid
    stack = explore.stack_indices(grid, slices.PLANES['coronal'],
                                  float(grid.z_mm[len(grid.z_mm) // 2]),
                                  4, 1e-6)
    assert len(set(stack.indices)) == 4
    assert stack.spacing_mm == pytest.approx(grid.voxel_size_mm)


@pytest.mark.parametrize('count, spacing', [(0, 0.5), (-1, 0.5),
                                            (3, 0.0), (3, -0.5)])
def test_a_nonsense_stack_is_refused(one_nucleus, count, spacing):
    with pytest.raises(ValueError):
        explore.stack_indices(one_nucleus.grid, slices.PLANES['coronal'],
                              0.0, count, spacing)


# ======================================================================
# Drawing
# ======================================================================
def test_the_figure_draws_the_whole_stack(one_nucleus, lgn_params):
    live = explore.build_live_figure(one_nucleus, lgn_params)
    grid = one_nucleus.grid
    stack = explore.stack_indices(grid, slices.PLANES['coronal'],
                                  float(grid.z_mm[len(grid.z_mm) // 2]),
                                  4, 0.5)
    drawn = explore.draw(live, one_nucleus, explore.View(
        overlay='layer', stack=stack, plane_stride_vox=3,
        electrode_slab_mm=0.25, shell_opacity=0.2))

    visible = [index for index in live.plane_traces
               if live.handle.data[index].visible]
    assert len(visible) == len(stack) == 4
    assert drawn.active.position_mm == pytest.approx(
        grid.z_mm[stack.active_index])


def test_only_the_active_plane_is_opaque(one_nucleus, lgn_params):
    """Otherwise a stack of identical planes says nothing about which
    one the flat panel is showing."""
    live = explore.build_live_figure(one_nucleus, lgn_params)
    grid = one_nucleus.grid
    stack = explore.stack_indices(grid, slices.PLANES['coronal'],
                                  float(grid.z_mm[len(grid.z_mm) // 2]),
                                  5, 0.4)
    explore.draw(live, one_nucleus, explore.View(
        overlay='layer', stack=stack, plane_stride_vox=3,
        electrode_slab_mm=0.25, shell_opacity=0.2))

    opacities = [live.handle.data[index].opacity
                 for index in live.plane_traces
                 if live.handle.data[index].visible]
    assert opacities[0] == 1.0
    assert all(value < 1.0 for value in opacities[1:])


def test_the_active_plane_is_always_the_first_trace(one_nucleus, lgn_params):
    """Whatever position it holds in the stack.

    The first plane trace is the one with a legend entry and the one
    the opacity rule addresses, so it has to be the one the panel
    shows -- not simply the lowest.
    """
    live = explore.build_live_figure(one_nucleus, lgn_params)
    grid = one_nucleus.grid
    centre = float(grid.z_mm[len(grid.z_mm) // 2])
    stack = explore.stack_indices(grid, slices.PLANES['coronal'], centre,
                                  5, 0.4)
    drawn = explore.draw(live, one_nucleus, explore.View(
        overlay='layer', stack=stack, plane_stride_vox=3,
        electrode_slab_mm=0.25, shell_opacity=0.2))

    assert stack.active == 2                       # genuinely in the middle
    first = live.handle.data[live.plane_traces[0]]
    finite = np.asarray(first.z, dtype=float)
    finite = finite[np.isfinite(finite)]
    assert np.allclose(finite, drawn.active.position_mm)


def test_the_panel_shows_the_active_plane_in_millimetres(one_nucleus,
                                                         lgn_params):
    """With its real coordinate arrays, not array indices.

    The montage slider lost these once by restyling them away, and a
    panel that silently falls back to indices looks almost right.
    """
    live = explore.build_live_figure(one_nucleus, lgn_params)
    grid = one_nucleus.grid
    plane = slices.PLANES['sagittal']
    stack = explore.stack_indices(
        grid, plane, explore.fullest_plane_mm(one_nucleus, plane), 1, 0.5)
    explore.draw(live, one_nucleus, explore.View(
        overlay='layer', stack=stack, plane_stride_vox=3,
        electrode_slab_mm=0.25, shell_opacity=0.2))

    panel = live.handle.data[live.panel_trace]
    assert np.allclose(panel.x, grid.coordinate(plane.horizontal))
    assert np.allclose(panel.y, grid.coordinate(plane.vertical))
    assert np.asarray(panel.z).shape == (len(panel.y), len(panel.x))


def test_the_panel_is_anchored_to_an_axis_that_exists(one_nucleus,
                                                      lgn_params):
    """A scene consumes no x axis, so the panel beside it is the
    figure's FIRST xy subplot. Anchoring to 'x2' leaves it blank."""
    live = explore.build_live_figure(one_nucleus, lgn_params)
    grid = one_nucleus.grid
    explore.draw(live, one_nucleus, explore.View(
        overlay='layer',
        stack=explore.stack_indices(grid, slices.PLANES['coronal'],
                                    float(grid.z_mm[len(grid.z_mm) // 2]),
                                    1, 0.5),
        plane_stride_vox=3, electrode_slab_mm=0.25, shell_opacity=0.2))

    anchor = live.handle.layout.yaxis.scaleanchor
    assert anchor == (live.handle.data[live.panel_trace].xaxis or 'x')
    assert anchor == 'x'


def test_switching_plane_lets_the_panel_reset_but_not_the_camera(
        one_nucleus, lgn_params):
    """`uirevision` is what survives a redraw.

    The scene's is constant, so a rotation the user made is kept
    through every control. Each panel axis carries the plane's name
    instead, because coronal and sagittal axes measure different things
    and holding a zoom across them would keep a range that no longer
    exists.
    """
    live = explore.build_live_figure(one_nucleus, lgn_params)
    grid = one_nucleus.grid

    def show(name):
        plane = slices.PLANES[name]
        explore.draw(live, one_nucleus, explore.View(
            overlay='layer',
            stack=explore.stack_indices(
                grid, plane, explore.fullest_plane_mm(one_nucleus, plane),
                1, 0.5),
            plane_stride_vox=3, electrode_slab_mm=0.25, shell_opacity=0.2))

    show('coronal')
    scene_revision = live.handle.layout.scene.uirevision
    assert live.handle.layout.xaxis.uirevision == 'coronal'
    show('sagittal')
    assert live.handle.layout.xaxis.uirevision == 'sagittal'
    assert live.handle.layout.scene.uirevision == scene_revision


def test_only_the_active_overlay_is_evaluated(one_nucleus, lgn_params,
                                              monkeypatch):
    """A plane costs one gather and one lookup, not one per overlay.

    The others are a control change away, and computing them all on
    every slider move is what a live viewer cannot afford.
    """
    asked = []
    original = explore.compute_overlay

    def spy(name, context, voxels):
        asked.append(name)
        return original(name, context, voxels)

    monkeypatch.setattr(slices, 'compute_overlay', spy)
    live = explore.build_live_figure(one_nucleus, lgn_params)
    grid = one_nucleus.grid
    explore.draw(live, one_nucleus, explore.View(
        overlay='eccentricity',
        stack=explore.stack_indices(grid, slices.PLANES['coronal'],
                                    float(grid.z_mm[len(grid.z_mm) // 2]),
                                    3, 0.4),
        plane_stride_vox=3, electrode_slab_mm=0.25, shell_opacity=0.2))
    assert set(asked) == {'eccentricity'}


def test_colour_limits_do_not_move_with_the_position(one_nucleus,
                                                     lgn_params):
    """A colour has to mean the same number at every position.

    The limits come from a sample of the whole volume once, so a plane
    whose values happen to be narrow is not stretched across the scale.
    """
    live = explore.build_live_figure(one_nucleus, lgn_params)
    grid = one_nucleus.grid
    plane = slices.PLANES['coronal']
    seen = set()
    for fraction in (0.25, 0.5, 0.75):
        index = int(fraction * (len(grid.z_mm) - 1))
        explore.draw(live, one_nucleus, explore.View(
            overlay='eccentricity',
            stack=explore.stack_indices(grid, plane,
                                        float(grid.z_mm[index]), 1, 0.5),
            plane_stride_vox=3, electrode_slab_mm=0.25, shell_opacity=0.2))
        panel = live.handle.data[live.panel_trace]
        seen.add((round(panel.zmin, 9), round(panel.zmax, 9)))
    assert len(seen) == 1


def test_contacts_in_the_slab_are_highlighted_in_both_halves(lgn_params,
                                                             synthetic_atlas):
    volume = explore.build_volume(
        lgn_params, {'left': synthetic_atlas}, None, ['layer'],
        electrodes=viz3d.electrodes_from_layout(
            synthetic_atlas, lgn_params, 32,
            np.random.default_rng(0)))
    live = explore.build_live_figure(volume, lgn_params)
    grid = volume.grid
    plane = slices.PLANES['coronal']
    drawn = explore.draw(live, volume, explore.View(
        overlay='layer',
        stack=explore.stack_indices(
            grid, plane, explore.fullest_plane_mm(volume, plane), 1, 0.5),
        plane_stride_vox=3, electrode_slab_mm=1.5, shell_opacity=0.2))

    assert drawn.n_contacts > 0
    assert len(live.handle.data[live.scene_near_trace].x) == drawn.n_contacts
    assert len(live.handle.data[live.panel_near_trace].x) == drawn.n_contacts


def test_a_position_with_no_tissue_says_so(both_nuclei, lgn_params):
    """Sagittal positions in the 17 mm gap between the nuclei cut
    nothing, and a blank panel on its own looks like a bug."""
    live = explore.build_live_figure(both_nuclei, lgn_params)
    view = explore.View(
        overlay='layer',
        stack=explore.stack_indices(both_nuclei.grid,
                                    slices.PLANES['sagittal'], 0.0, 1, 0.5),
        plane_stride_vox=3, electrode_slab_mm=0.25, shell_opacity=0.2)
    drawn = explore.draw(live, both_nuclei, view)

    assert drawn.off_tissue
    assert slices.column_at(both_nuclei.grid, drawn.active.position_mm) is None
    assert 'between the two nuclei' in explore.status_text(both_nuclei, view,
                                                           drawn)


def test_status_reports_what_was_drawn_not_what_was_asked(one_nucleus,
                                                          lgn_params):
    live = explore.build_live_figure(one_nucleus, lgn_params)
    grid = one_nucleus.grid
    view = explore.View(
        overlay='layer',
        stack=explore.stack_indices(grid, slices.PLANES['coronal'],
                                    float(grid.z_mm[0]), 5, 0.31),
        plane_stride_vox=3, electrode_slab_mm=0.25, shell_opacity=0.2)
    text = explore.status_text(one_nucleus, view,
                               explore.draw(live, one_nucleus, view))

    assert f'{view.stack.spacing_mm:.3f} mm' in text
    assert f'{view.stack.dropped} outside the volume' in text


# ======================================================================
# Exporting
# ======================================================================
def test_config_changes_finds_only_what_moved():
    baseline = {'a': {'b': 1, 'c': [1, 2]}, 'd': True}
    current = copy.deepcopy(baseline)
    current['a']['b'] = 2
    assert explore.config_changes(baseline, current) == {'a.b': 2}


def test_config_changes_ignores_a_round_trip_through_a_widget():
    """An int back as a float, a tuple back as a list.

    Without this every slider reports a change the moment it is
    touched, even when it is put back exactly where it started.
    """
    baseline = {'n': 4, 'x': 0.5, 'pair': [1.0, 2.0], 'flag': True}
    current = {'n': 4.0, 'x': np.float64(0.5), 'pair': (1, 2),
               'flag': np.bool_(True)}
    assert explore.config_changes(baseline, current) == {}


def test_config_changes_reports_a_key_the_baseline_lacks():
    assert explore.config_changes({}, {'a': {'b': 3}}) == {'a.b': 3}


def test_nest_round_trips_dotted_paths():
    changes = {'visualization.slices.count': 3,
               'visualization.slices.plane': 'sagittal',
               'atlas.hemispheres': ['left']}
    assert explore.nest(changes) == {
        'visualization': {'slices': {'count': 3, 'plane': 'sagittal'}},
        'atlas': {'hemispheres': ['left']}}


def test_the_fragment_is_yaml_that_safe_load_accepts():
    """Numpy scalars reach the dumper as `!!python/object` tags, which
    produce a file that `safe_load` then refuses -- a failure at read
    time, somewhere else."""
    import yaml
    text = explore.fragment_yaml({'visualization.stride_vox': np.int64(6),
                                  'visualization.surface.opacity':
                                      np.float32(0.5)})
    assert yaml.safe_load(text) == {
        'visualization': {'stride_vox': 6, 'surface': {'opacity': 0.5}}}


def test_the_exported_document_is_yaml(explorer):
    """Including the command line, which is a comment: a file called
    `.yaml` that is not YAML is the worse trade."""
    import yaml
    exported = explorer.export()
    assert yaml.safe_load(exported.text) is not None
    assert f'#   {exported.command}' in exported.text


def test_the_controls_write_into_the_config(explorer):
    explorer.plane_control.value = 'sagittal'
    explorer.count_control.value = 3
    explorer.spacing_control.value = 0.4

    params = explorer.params
    assert params['visualization']['slices']['plane'] == 'sagittal'
    assert params['visualization']['combined']['plane'] == 'sagittal'
    assert params['visualization']['explorer']['count'] == 3
    assert params['visualization']['explorer']['spacing_mm'] == 0.4
    assert params['visualization']['overlays'][0] == \
        explorer.overlay_control.value


def test_the_active_overlay_leads_the_exported_list(explorer):
    explorer.overlay_control.value = 'eccentricity'
    overlays = explorer.params['visualization']['overlays']
    assert overlays[0] == 'eccentricity'
    assert set(overlays) == set(explorer.offered)


def test_an_explicit_range_lifts_the_automatic_trimming(explorer):
    """`min_slice_fraction` keeps AUTOMATIC placement off the tapered
    ends of the nucleus. An explicit range is the user saying where the
    slices go, so it has nothing left to decide."""
    exported = explorer.export()
    assert exported.changes['visualization.slices.min_slice_fraction'] == 0.0
    assert exported.changes['visualization.combined.min_slice_fraction'] == 0.0


def test_the_exported_command_reproduces_the_planes_on_screen(explorer,
                                                              tmp_path):
    """The whole claim of the export, end to end.

    Runs the command the explorer hands back and checks that the
    figure it writes holds exactly the planes the explorer was
    showing -- same plane, same count, same millimetres.
    """
    import shlex

    grid = explorer.volume.grid
    explorer.position_control.value = float(grid.z_mm[len(grid.z_mm) // 2])
    explorer.count_control.value = 3
    explorer.spacing_control.value = 0.5
    drawn = explorer.refresh()
    wanted = [float(grid.z_mm[index]) for index in drawn.stack.indices]

    exported = explorer.export()
    argv = shlex.split(exported.command)[3:]
    argv += ['--no-browser', '--out', str(tmp_path / 'out.html')]
    assert slices.main(argv) == 0

    figure = _figure_from(argv)
    assert [one.plane.name for one in figure.slices] == \
        ['coronal'] * len(wanted)
    assert [one.position_mm for one in figure.slices] == \
        pytest.approx(wanted)


@pytest.mark.parametrize('plane', ['coronal', 'sagittal', 'horizontal'])
def test_the_exported_command_runs_on_every_plane(explorer, tmp_path, plane):
    """The round trip above, but on all three planes.

    Only coronal was covered, and coronal is the one plane whose
    positions are all positive. ML is signed and DV starts below the
    atlas origin, so the other two export a `--range` that begins with
    a minus sign -- which argparse read as the next option, making
    every sagittal and horizontal export unrunnable.
    """
    explorer.plane_control.value = plane
    explorer.count_control.value = 3
    explorer.spacing_control.value = 0.5
    drawn = explorer.refresh()
    coordinate = explorer.volume.grid.coordinate(drawn.stack.plane.normal)
    wanted = [float(coordinate[index]) for index in drawn.stack.indices]

    argv = explorer.export().tokens
    argv += ['--no-browser', '--out', str(tmp_path / 'out.html')]
    assert slices.main(argv) == 0

    figure = _figure_from(argv)
    assert [one.plane.name for one in figure.slices] == [plane] * len(wanted)
    assert [one.position_mm for one in figure.slices] == pytest.approx(wanted)


def test_a_negative_range_travels_as_one_token(lgn_params):
    """The '=' form, pinned where it cannot depend on the atlas.

    `--range -10.7750:-10.2750` is two tokens and argparse rejects it;
    `--range=-10.7750:-10.2750` is one and it does not.
    """
    params = copy.deepcopy(lgn_params)
    params['visualization']['slices']['range_mm'] = [-10.775, -10.275]
    stack = explore.Stack(plane=slices.PLANES['sagittal'], indices=[4],
                          active=0, spacing_mm=0.5, requested=1)

    argv = explore.reproduce_command(
        params, copy.deepcopy(lgn_params), {}, dict(synthetic=True),
        stack, ['layer'], 'combined')

    assert '--range=-10.7750:-10.2750' in argv
    assert '--range' not in argv               # never as a bare flag
    assert slices.build_parser().parse_args(argv).range_mm ==         '-10.7750:-10.2750'


def test_the_tokens_are_raw_and_only_the_command_is_quoted(lgn_params):
    """A token quoted twice works in neither place.

    `tokens` goes to `slices.main` and must hold the real path;
    `command` goes to a shell and must hold a quoted one. The notebook
    uses `tokens` rather than re-splitting `command`, because
    `shlex.split` is POSIX and eats the backslashes in a Windows path.
    """
    from pathlib import Path as _Path
    import shlex

    stack = explore.Stack(plane=slices.PLANES['coronal'], indices=[4],
                          active=0, spacing_mm=0.5, requested=1)
    directory = r'C:\My Atlas\Erwin'
    params = copy.deepcopy(lgn_params)
    params['visualization']['slices']['range_mm'] = [3.6, 3.6]
    argv = explore.reproduce_command(
        params, copy.deepcopy(lgn_params), {},
        dict(atlas_dir=directory), stack, ['layer'], 'combined')
    assert directory in argv                   # raw in the tokens

    exported = explore.ExportedSettings(
        changes={}, fragment='', argv=argv, params=lgn_params,
        directory=_Path('.'), fragment_name='f.yaml', full_name='p.yaml',
        needs_params_file=False)
    assert exported.tokens == argv
    assert shlex.split(exported.command)[3:] == argv    # and quoted once


def test_the_tokens_carry_the_params_file_the_command_names(explorer,
                                                            tmp_path):
    """`--params` is appended by `tokens`, so the two cannot drift."""
    exported = explorer.export()
    object.__setattr__(exported, 'needs_params_file', True)
    assert exported.tokens[-2] == '--params'
    assert exported.tokens[-1] in exported.command


def _figure_from(argv):
    """Rebuild the `SliceFigure` a `slices.main` run would draw.

    The same steps `slices.main` takes, so the assertion is about the
    slices that command produces rather than about bytes in a file.
    """
    from dynaphos.utils import load_params
    from dynaphos_lgn.build import (build_atlas, build_hemisphere_atlas,
                                    configured_hemispheres)
    from pathlib import Path

    options = slices.build_parser().parse_args(argv)
    repository = Path(slices.__file__).resolve().parents[1]
    params = load_params(str(repository / 'config' / 'params_lgn.yaml'))
    params = slices._apply_slice_overrides(
        viz3d._apply_overrides(params, options), options)

    atlas = build_atlas(params, options.atlas_dir, options.synthetic)
    atlases = {name: build_hemisphere_atlas(params, atlas, name)
               for name in configured_hemispheres(params)}
    masks = {name: viz3d.tissue_mask(atlases[name], params)
             for name in atlases}
    contexts = {name: viz3d.AtlasContext(atlases[name], params)
                for name in atlases}
    grid = slices.build_scene_grid(params, atlases, masks)
    planes = slices._planes_to_cut(params, grid, masks, options,
                                   slices.tissue_centroid(grid, masks))
    return slices.build_slice_figure(
        params, atlases, masks, contexts,
        list(slices.require(params, 'visualization.overlays')), planes,
        None, banner='', selection='', grid=grid)


def test_the_command_carries_the_tissue_selection(lgn_params):
    """The one part of the state with flags but no config keys, which
    is why it is rendered from `mask_options` rather than diffed."""
    baseline = copy.deepcopy(lgn_params)
    stack = explore.Stack(plane=slices.PLANES['coronal'], indices=[4],
                          active=0, spacing_mm=0.5, requested=1)
    params = copy.deepcopy(lgn_params)
    params['visualization']['slices']['range_mm'] = [3.6, 3.6]

    argv = explore.reproduce_command(
        params, baseline,
        dict(cell_class='magno', layers=[1, 2],
             min_eccentricity_deg=2.0, max_eccentricity_deg=20.0,
             clips=[('ap', 6.0, 9.0)], include_ipsi_placeholder=True),
        dict(synthetic=True, electrodes='layout'), stack, ['layer'],
        'combined')

    assert '--cell-class' in argv and 'magno' in argv
    assert argv[argv.index('--layers') + 1:argv.index('--layers') + 3] == \
        ['1', '2']
    assert '--ecc-min' in argv and '--ecc-max' in argv
    assert argv[argv.index('--clip') + 1] == 'ap:6:9'
    assert '--include-ipsi' in argv
    assert '--synthetic' in argv
    # and it parses
    slices.build_parser().parse_args(argv)


def test_the_command_names_a_params_file_when_it_needs_one(explorer,
                                                           monkeypatch):
    """A position inside the tapered end of the nucleus is drawable
    here and would be trimmed away there, and no flag lifts the
    trimming -- only the config does."""
    monkeypatch.setattr(explorer, '_needs_params_file',
                        lambda stack, section: True)
    exported = explorer.export()
    assert exported.needs_params_file
    assert '--params' in exported.command
    assert exported.full_name in exported.command
    assert 'whole parameter file' in exported.text


def test_saving_writes_where_it_says(explorer, tmp_path):
    import yaml
    exported = explorer.export()

    fragment = exported.save(tmp_path / 'fragment.yaml')
    assert fragment.exists()
    assert yaml.safe_load(fragment.read_text(encoding='utf-8')) is not None

    full = exported.save(tmp_path / 'full.yaml', full=True)
    loaded = yaml.safe_load(full.read_text(encoding='utf-8'))
    assert loaded['visualization']['slices']['range_mm'] == \
        explorer.params['visualization']['slices']['range_mm']
    assert 'atlas' in loaded and 'run' in loaded


def test_a_saved_full_file_drives_the_slice_viewer(explorer, tmp_path):
    """The point of the full export: `--params` can be pointed at it."""
    exported = explorer.export()
    path = exported.save(tmp_path / 'params.yaml', full=True)
    assert slices.main(['--params', str(path), '--view', 'slices',
                        '--synthetic', '--backend', 'matplotlib',
                        '--out', str(tmp_path / 'out.png'),
                        '--no-browser']) == 0


# ======================================================================
# The explorer's own behaviour
# ======================================================================
def test_switching_plane_moves_the_position_control_to_its_axis(explorer):
    """Sagittal ML and coronal AP do not overlap.

    Setting the new bounds directly fails outright, and setting the
    value first clamps it silently to an end.
    """
    grid = explorer.volume.grid
    for name, axis in (('sagittal', 0), ('horizontal', 1), ('coronal', 2)):
        explorer.plane_control.value = name
        coordinate = grid.coordinate(axis)
        assert explorer.position_control.min == pytest.approx(coordinate[0])
        assert explorer.position_control.max == pytest.approx(coordinate[-1])
        assert (coordinate[0] <= explorer.position_control.value
                <= coordinate[-1])


def test_each_plane_remembers_where_you_were(explorer):
    grid = explorer.volume.grid
    explorer.plane_control.value = 'coronal'
    explorer.position_control.value = float(grid.z_mm[10])
    explorer.plane_control.value = 'sagittal'
    explorer.plane_control.value = 'coronal'
    assert explorer.position_control.value == pytest.approx(grid.z_mm[10])


def test_applying_a_selection_rebuilds_the_volume(explorer):
    before = explorer.volume.masks['left'].sum()
    explorer.cell_class_control.value = 'magno'
    explorer.apply_selection()

    assert explorer.volume.masks['left'].sum() < before
    assert 'magno' in explorer.volume.selection
    assert explorer.mask_options['cell_class'] == 'magno'
    assert explorer.widget.children[0].children[1] is explorer.live.handle


def test_an_empty_selection_keeps_the_last_good_one(explorer):
    """Rebuilding into nothing is a mistake, not a state to show."""
    volume = explorer.volume
    explorer.ecc_control.value = (explorer.volume.eccentricity_deg_max,
                                  explorer.volume.eccentricity_deg_max)
    explorer.cell_class_control.value = 'magno'
    explorer.layers_control.value = (3, 4)      # parvo: nothing left
    explorer.apply_selection()

    assert explorer.volume is volume
    assert 'color:#b00' in explorer.message.value


def test_the_live_controls_redraw(explorer):
    """One handler, and every control on it."""
    grid = explorer.volume.grid
    explorer.position_control.value = float(grid.z_mm[len(grid.z_mm) // 2])
    first = explorer.status.value

    explorer.overlay_control.value = 'eccentricity'
    panel = explorer.live.handle.data[explorer.live.panel_trace]
    assert panel.zmin == pytest.approx(
        explorer.live.handle.data[explorer.live.panel_trace].zmin)
    assert 'Eccentricity' in str(explorer.live.handle.layout.title.text)

    explorer.count_control.value = 4
    assert '4 planes' in explorer.status.value
    assert explorer.status.value != first


def test_shrinking_the_stack_hides_the_planes_it_dropped(explorer):
    """A live figure keeps its trace list fixed, so a plane that is no
    longer in the stack has to be hidden rather than removed -- and a
    stale one left showing would be a cut through nowhere."""
    grid = explorer.volume.grid
    explorer.position_control.value = float(grid.z_mm[len(grid.z_mm) // 2])
    explorer.count_control.value = 6
    showing = [index for index in explorer.live.plane_traces
               if explorer.live.handle.data[index].visible]
    assert len(showing) == 6

    explorer.count_control.value = 2
    showing = [index for index in explorer.live.plane_traces
               if explorer.live.handle.data[index].visible]
    assert len(showing) == 2


def test_opening_with_an_unknown_overlay_is_refused():
    with pytest.raises(KeyError):
        explore.Explorer.open(synthetic=True, overlays=['not_an_overlay'])


# ======================================================================
# Command line
# ======================================================================
def test_check_environment_finds_the_dependencies():
    found, missing = explore.check_environment()
    assert set(found) | set(missing) == set(explore.REQUIREMENTS)
    assert not missing            # the importorskip at the top guarantees it


def test_main_reports_readiness(capsys):
    assert explore.main([]) == 0
    printed = capsys.readouterr().out
    assert 'notebook widget, not a command' in printed
    assert 'demo_lgn_explorer.ipynb' in printed


def test_main_lists_overlays(capsys):
    assert explore.main(['--list-overlays']) == 0
    assert 'eccentricity' in capsys.readouterr().out


# ======================================================================
# The real atlas
# ======================================================================
@requires_real_atlas
def test_a_real_coronal_slice_holds_every_lamina(lgn_params, real_atlas):
    """The point of the whole viewer, on the real thing: a mid-coronal
    cut through the macaque LGN shows all four laminar codes."""
    volume = explore.build_volume(lgn_params, {'left': real_atlas}, None,
                                  ['layer'])
    plane = slices.PLANES['coronal']
    stack = explore.stack_indices(
        volume.grid, plane, explore.fullest_plane_mm(volume, plane), 1, 0.5)
    one = slices.cut(volume.grid, volume.masks, volume.contexts, ['layer'],
                     plane, stack.active_index)

    codes = np.unique(one.values['layer'][np.isfinite(one.values['layer'])])
    assert len(codes) == 4
