"""Tests for the coronal / sagittal / horizontal slice viewer.

A slice picture is easy to get subtly wrong and hard to notice: the
outline can be right while the colours come from the wrong plane, or
the axes can silently fall back to array indices. Both of those
happened during development, and both have a test here.

Everything runs on the synthetic atlas, so no download is needed.
"""
import numpy as np
import pytest

from dynaphos_lgn import slices, viz3d
from dynaphos_lgn.test.conftest import requires_real_atlas


@pytest.fixture
def one_nucleus(synthetic_atlas, lgn_params):
    """{hemisphere: atlas}, masks and contexts for the left LGN."""
    atlases = {'left': synthetic_atlas}
    masks = {'left': viz3d.tissue_mask(synthetic_atlas, lgn_params)}
    contexts = {'left': viz3d.AtlasContext(synthetic_atlas, lgn_params)}
    return atlases, masks, contexts


@pytest.fixture
def both_nuclei(synthetic_atlas, lgn_params):
    from dynaphos_lgn.atlas import MirroredAtlas
    mirrored = MirroredAtlas(synthetic_atlas, lgn_params)
    atlases = {'left': synthetic_atlas, 'right': mirrored}
    masks = {name: viz3d.tissue_mask(atlas, lgn_params)
             for name, atlas in atlases.items()}
    contexts = {name: viz3d.AtlasContext(atlas, lgn_params)
                for name, atlas in atlases.items()}
    return atlases, masks, contexts


@pytest.fixture
def grid(one_nucleus, lgn_params):
    atlases, masks, _ = one_nucleus
    return slices.build_scene_grid(lgn_params, atlases, masks)


# ======================================================================
# The planes themselves
# ======================================================================
def test_every_plane_uses_three_different_scene_axes():
    for plane in slices.PLANES.values():
        axes = {plane.normal, plane.horizontal, plane.vertical}
        assert axes == {0, 1, 2}, plane.name


def test_planes_hold_the_axis_anatomy_says_they_do():
    # Scene axes: 0 signed ML, 1 DV, 2 AP.
    assert slices.PLANES['coronal'].normal == 2       # fixed AP
    assert slices.PLANES['sagittal'].normal == 0      # fixed ML
    assert slices.PLANES['horizontal'].normal == 1    # fixed DV


def test_vertical_axes_put_dorsal_and_anterior_at_the_top():
    assert slices.PLANES['coronal'].letters[3] == 'D'
    assert slices.PLANES['sagittal'].letters[3] == 'D'
    assert slices.PLANES['horizontal'].letters[3] == 'A'
    assert slices.PLANES['sagittal'].letters[1] == 'A'


# ======================================================================
# The scene grid
# ======================================================================
def test_grid_columns_are_that_atlas_own_millimetres(grid, one_nucleus,
                                                     lgn_params):
    atlases, _, _ = one_nucleus
    atlas = atlases['left']
    sign = viz3d.hemisphere_sign(lgn_params, 'left')
    for column in (0, len(grid.x_mm) // 2, len(grid.x_mm) - 1):
        index = int(grid.ml_index[column])
        expected = sign * atlas.index_to_horsley_clarke(index, 0, 0)[0]
        assert grid.x_mm[column] == pytest.approx(expected, abs=1e-9)


def test_grid_is_regular_and_at_the_voxel_pitch(grid, synthetic_atlas):
    for coordinate in (grid.x_mm, grid.y_mm, grid.z_mm):
        steps = np.diff(coordinate)
        assert np.allclose(steps, synthetic_atlas._voxel_size_mm)


def test_grid_is_cropped_to_the_tissue(grid, one_nucleus):
    _, masks, _ = one_nucleus
    mask = masks['left']
    # Every edge of the cropped grid must carry tissue, or the crop was
    # too loose.
    assert mask.any(axis=(1, 2))[grid.ml_index[0]]
    assert mask.any(axis=(1, 2))[grid.ml_index[-1]]
    assert mask.any(axis=(0, 2))[grid.dv_offset]
    assert mask.any(axis=(0, 1))[grid.ap_offset]


def test_one_nucleus_occupies_one_side(grid):
    assert np.all(grid.x_mm < 0)                     # the left nucleus
    assert np.all(grid.owner >= 0)                   # no gap to span


def test_both_nuclei_are_mirror_images_with_a_real_gap(both_nuclei,
                                                       lgn_params):
    atlases, masks, _ = both_nuclei
    grid = slices.build_scene_grid(lgn_params, atlases, masks)

    left = grid.x_mm[grid.owner == 0]
    right = grid.x_mm[grid.owner == 1]
    assert left.max() < 0 < right.min()
    assert np.allclose(np.sort(-left), np.sort(right), atol=1e-9)
    # The midline is empty, and it is empty because there is no tissue
    # there -- not because the grid skipped it.
    gap = grid.owner < 0
    assert gap.any()
    assert np.all(np.abs(grid.x_mm[gap]) < np.abs(left).min() + 1e-9)


def test_column_hemisphere_lookup(both_nuclei, lgn_params):
    atlases, masks, _ = both_nuclei
    grid = slices.build_scene_grid(lgn_params, atlases, masks)
    assert grid.hemisphere_of_column(0) == 'left'
    assert grid.hemisphere_of_column(len(grid.x_mm) - 1) == 'right'
    midline = int(np.argmin(np.abs(grid.x_mm)))
    assert grid.hemisphere_of_column(midline) is None


def test_empty_selection_says_so(synthetic_atlas, lgn_params):
    masks = {'left': np.zeros(synthetic_atlas.atlas_shape, dtype=bool)}
    with pytest.raises(ValueError, match='no tissue'):
        slices.build_scene_grid(lgn_params, {'left': synthetic_atlas}, masks)


# ======================================================================
# Cutting
# ======================================================================
@pytest.mark.parametrize('plane_name', ['coronal', 'sagittal', 'horizontal'])
def test_cut_reads_the_atlas_at_the_right_voxels(plane_name, grid,
                                                 one_nucleus, lgn_params):
    """Every coloured pixel must hold the value of the atlas voxel that
    pixel stands for -- checked against a direct lookup, not against
    the slicing code's own idea of where it was."""
    atlases, masks, contexts = one_nucleus
    atlas = atlases['left']
    plane = slices.PLANES[plane_name]
    # The fullest plane, not the middle one: the synthetic atlas has
    # interlaminar gaps, and a horizontal cut can land in one.
    index = int(np.argmax(slices.plane_voxel_counts(grid, masks, plane)))
    one = slices.cut(grid, masks, contexts, ['eccentricity'], plane, index)

    assert one.tissue.any(), f'{plane_name} plane {index} is empty'
    values = one.values['eccentricity']
    rows, columns = np.nonzero(one.tissue)

    for row, column in list(zip(rows, columns))[::37]:
        if plane_name == 'coronal':
            voxel = (grid.ml_index[column], row + grid.dv_offset,
                     index + grid.ap_offset)
        elif plane_name == 'sagittal':
            voxel = (grid.ml_index[index], row + grid.dv_offset,
                     column + grid.ap_offset)
        else:
            voxel = (grid.ml_index[column], index + grid.dv_offset,
                     row + grid.ap_offset)
        assert atlas.valid[voxel]
        assert values[row, column] == pytest.approx(
            atlas.eccentricity_deg[voxel], rel=1e-6)


def test_cut_marks_background_apart_from_missing_values(grid, one_nucleus,
                                                        lgn_params):
    """Background and "no value" must stay distinguishable: one is the
    absence of a voxel, the other a voxel the atlas cannot describe."""
    atlases, masks, contexts = one_nucleus
    plane = slices.PLANES['coronal']
    index = len(grid.z_mm) // 2
    one = slices.cut(grid, masks, contexts, ['eccentricity'], plane, index)

    assert one.tissue.sum() < one.tissue.size      # there IS background
    assert np.all(np.isnan(one.values['eccentricity'][~one.tissue]))
    assert one.n_voxels == int(one.tissue.sum())


def test_slice_extent_matches_the_grid(grid, one_nucleus, lgn_params):
    _, masks, contexts = one_nucleus
    plane = slices.PLANES['coronal']
    one = slices.cut(grid, masks, contexts, ['eccentricity'], plane,
                     len(grid.z_mm) // 2)
    half = grid.voxel_size_mm / 2
    assert one.extent_mm[0] == pytest.approx(grid.x_mm[0] - half)
    assert one.extent_mm[1] == pytest.approx(grid.x_mm[-1] + half)
    assert one.extent_mm[2] == pytest.approx(grid.y_mm[0] - half)
    assert one.extent_mm[3] == pytest.approx(grid.y_mm[-1] + half)


def test_slice_position_is_the_planes_own_coordinate(grid, one_nucleus):
    _, masks, contexts = one_nucleus
    for name, plane in slices.PLANES.items():
        index = len(grid.coordinate(plane.normal)) // 2
        one = slices.cut(grid, masks, contexts, ['eccentricity'], plane,
                         index)
        assert one.position_mm == pytest.approx(
            grid.coordinate(plane.normal)[index]), name


def test_a_sagittal_slice_belongs_to_exactly_one_nucleus(both_nuclei,
                                                         lgn_params):
    """Which is what lets the sign of an ML position pick a side."""
    atlases, masks, contexts = both_nuclei
    grid = slices.build_scene_grid(lgn_params, atlases, masks)
    plane = slices.PLANES['sagittal']

    for column in (int(np.flatnonzero(grid.owner == 0)[len(
            np.flatnonzero(grid.owner == 0)) // 2]),
            int(np.flatnonzero(grid.owner == 1)[len(
                np.flatnonzero(grid.owner == 1)) // 2])):
        found = slices._gather(grid, masks, plane, column)
        assert len(found) == 1
        assert found[0][0] == grid.hemisphere_of_column(column)


def test_a_coronal_slice_of_both_nuclei_holds_both(both_nuclei, lgn_params):
    atlases, masks, contexts = both_nuclei
    grid = slices.build_scene_grid(lgn_params, atlases, masks)
    plane = slices.PLANES['coronal']
    index = len(grid.z_mm) // 2
    one = slices.cut(grid, masks, contexts, ['eccentricity'], plane, index)

    columns = np.flatnonzero(one.tissue.any(axis=0))
    assert (grid.x_mm[columns] < 0).any()
    assert (grid.x_mm[columns] > 0).any()


def test_several_overlays_share_one_gather(grid, one_nucleus):
    _, masks, contexts = one_nucleus
    plane = slices.PLANES['coronal']
    names = ['eccentricity', 'layer', 'cells_per_voxel']
    one = slices.cut(grid, masks, contexts, names, plane, len(grid.z_mm) // 2)
    assert set(one.values) == set(names)
    for name in names:
        assert one.values[name].shape == one.tissue.shape
        assert np.isfinite(one.values[name][one.tissue]).any()


# ======================================================================
# Placing slices
# ======================================================================
def test_slice_indices_span_the_tissue(grid, one_nucleus):
    _, masks, _ = one_nucleus
    plane = slices.PLANES['coronal']
    indices = slices.slice_indices(grid, masks, plane, 7)
    assert len(indices) == 7
    assert indices == sorted(indices)
    low, high = slices.occupied_range(grid, masks, plane)
    assert indices[0] >= low and indices[-1] <= high


def test_slice_indices_do_not_repeat_when_asked_for_too_many(grid,
                                                             one_nucleus):
    _, masks, _ = one_nucleus
    plane = slices.PLANES['coronal']
    thickness = len(grid.z_mm)
    indices = slices.slice_indices(grid, masks, plane, thickness * 3)
    assert len(indices) == len(set(indices))
    assert len(indices) <= thickness


def test_min_fraction_trims_the_tapered_ends(grid, one_nucleus):
    _, masks, _ = one_nucleus
    plane = slices.PLANES['coronal']
    wide = slices.occupied_range(grid, masks, plane, 0.0)
    trimmed = slices.occupied_range(grid, masks, plane, 0.5)
    assert trimmed[0] >= wide[0]
    assert trimmed[1] <= wide[1]

    counts = slices.plane_voxel_counts(grid, masks, plane)
    assert counts[trimmed[0]] >= 0.5 * counts.max()
    assert counts[trimmed[1]] >= 0.5 * counts.max()


def test_explicit_range_overrides_the_automatic_span(grid, one_nucleus):
    _, masks, _ = one_nucleus
    plane = slices.PLANES['coronal']
    low, high = slices.occupied_range(grid, masks, plane)
    middle = (grid.z_mm[low] + grid.z_mm[high]) / 2
    indices = slices.slice_indices(grid, masks, plane, 3,
                                   range_mm=(middle, grid.z_mm[high]))
    assert grid.z_mm[indices[0]] >= middle - grid.voxel_size_mm


def test_a_range_outside_the_volume_says_so(grid, one_nucleus):
    """Rather than clamping to the nearest edge and returning a slice,
    which would look like an answer."""
    _, masks, _ = one_nucleus
    with pytest.raises(ValueError, match='outside the drawn volume'):
        slices.slice_indices(grid, masks, slices.PLANES['coronal'], 3,
                             range_mm=(1e6, 1e6 + 1))


# ======================================================================
# Colour, shared across slices
# ======================================================================
def test_limits_come_from_the_volume_not_from_one_plane(one_nucleus,
                                                        lgn_params):
    """A scale set per slice would make every slice look alike and none
    of them comparable."""
    _, masks, contexts = one_nucleus
    rng = np.random.default_rng(0)
    limits = slices.sample_limits(lgn_params, masks, contexts,
                                  ['eccentricity', 'layer'], rng)
    assert limits['layer'] is None                   # categorical
    low, high = limits['eccentricity']
    assert low < high

    whole = viz3d.compute_overlay(
        'eccentricity', contexts['left'], np.argwhere(masks['left'])[::13])
    assert low <= np.nanpercentile(whole.values, 50) <= high


def test_the_same_colours_are_used_on_every_slice(grid, one_nucleus,
                                                  lgn_params):
    _, masks, contexts = one_nucleus
    plane = slices.PLANES['coronal']
    rng = np.random.default_rng(0)
    limits = slices.sample_limits(lgn_params, masks, contexts,
                                  ['eccentricity'], rng)['eccentricity']
    overlay = viz3d.compute_overlay('eccentricity', contexts['left'],
                                    np.argwhere(masks['left'])[:1])

    specs = []
    for index in slices.slice_indices(grid, masks, plane, 3):
        one = slices.cut(grid, masks, contexts, ['eccentricity'], plane,
                         index)
        _, spec = slices.slice_colours(overlay, one.values['eccentricity'],
                                       one.tissue, lgn_params, limits)
        specs.append(spec)
    assert all(spec.cmin == specs[0].cmin for spec in specs)
    assert all(spec.cmax == specs[0].cmax for spec in specs)


def test_background_is_not_the_missing_value_grey(grid, one_nucleus,
                                                  lgn_params):
    import matplotlib.colors as mcolors
    _, masks, contexts = one_nucleus
    plane = slices.PLANES['coronal']
    one = slices.cut(grid, masks, contexts, ['eccentricity'], plane,
                     len(grid.z_mm) // 2)
    overlay = viz3d.compute_overlay('eccentricity', contexts['left'],
                                    np.argwhere(masks['left'])[:1])
    rgba, _ = slices.slice_colours(overlay, one.values['eccentricity'],
                                   one.tissue, lgn_params, (0.0, 10.0))

    background = np.array(mcolors.to_rgba(
        lgn_params['visualization']['slices']['background']))
    missing = np.array(mcolors.to_rgba(
        lgn_params['visualization']['missing_colour']))
    assert not np.allclose(background, missing)
    assert np.allclose(rgba[~one.tissue], background)
    assert rgba.shape == one.tissue.shape + (4,)


def test_heatmap_values_keep_background_transparent(grid, one_nucleus,
                                                    lgn_params):
    _, masks, contexts = one_nucleus
    plane = slices.PLANES['coronal']
    one = slices.cut(grid, masks, contexts, ['eccentricity'], plane,
                     len(grid.z_mm) // 2)
    overlay = viz3d.compute_overlay('eccentricity', contexts['left'],
                                    np.argwhere(masks['left'])[:1])
    _, spec = slices.slice_colours(overlay, one.values['eccentricity'],
                                   one.tissue, lgn_params, (0.0, 40.0))
    prepared = slices.heatmap_values(one.values['eccentricity'], one.tissue,
                                     spec)
    assert np.all(np.isnan(prepared[~one.tissue]))
    assert np.all(np.isfinite(prepared[one.tissue]))


# ======================================================================
# Electrodes on a plane
# ======================================================================
def test_contacts_are_filtered_by_distance_from_the_plane(one_nucleus,
                                                          lgn_params):
    atlases, _, _ = one_nucleus
    electrodes = viz3d.electrodes_from_layout(atlases['left'], lgn_params, 8,
                                              np.random.default_rng(0))
    plane = slices.PLANES['coronal']
    position = float(np.median(electrodes.positions_mm[:, plane.normal]))

    near = slices.contacts_near(electrodes, plane, position, 0.5)
    assert np.all(near.distance_mm <= 0.5)
    expected = np.abs(electrodes.positions_mm[:, plane.normal]
                      - position) <= 0.5
    assert len(near) == int(expected.sum())
    assert np.allclose(near.horizontal_mm,
                       electrodes.positions_mm[expected, plane.horizontal])


def test_no_electrodes_is_not_an_error(one_nucleus, lgn_params):
    assert len(slices.contacts_near(None, slices.PLANES['coronal'], 0.0,
                                    1.0)) == 0


# ======================================================================
# Where to cut
# ======================================================================
def test_centroid_lands_inside_the_tissue(grid, one_nucleus):
    atlases, masks, _ = one_nucleus
    centre = slices.tissue_centroid(grid, masks)
    assert grid.x_mm[0] <= centre[0] <= grid.x_mm[-1]
    assert grid.y_mm[0] <= centre[1] <= grid.y_mm[-1]
    assert grid.z_mm[0] <= centre[2] <= grid.z_mm[-1]


def test_at_accepts_scene_millimetres(grid, one_nucleus, lgn_params):
    _, masks, _ = one_nucleus
    wanted = (float(grid.x_mm[len(grid.x_mm) // 3]),
              float(grid.y_mm[len(grid.y_mm) // 3]),
              float(grid.z_mm[len(grid.z_mm) // 3]))
    got = slices.resolve_position(
        grid, masks, lgn_params, ','.join(str(v) for v in wanted), None)
    assert np.allclose(got, wanted)


def test_at_flips_an_unsigned_ml_onto_the_drawn_nucleus(grid, one_nucleus,
                                                        lgn_params, caplog):
    """The atlas quotes ML as a positive lateral distance, so someone
    reading millimetres off it will type the positive value."""
    _, masks, _ = one_nucleus
    lateral = -float(grid.x_mm[len(grid.x_mm) // 2])       # positive
    got = slices.resolve_position(grid, masks, lgn_params,
                                  f'{lateral},0,5', None)
    assert got[0] < 0
    assert got[0] == pytest.approx(-lateral)


def test_at_electrode_picks_that_contact(grid, one_nucleus, lgn_params):
    atlases, masks, _ = one_nucleus
    electrodes = viz3d.electrodes_from_layout(atlases['left'], lgn_params, 5,
                                              np.random.default_rng(0))
    got = slices.resolve_position(grid, masks, lgn_params, 'electrode:2',
                                  electrodes)
    assert np.allclose(got, electrodes.positions_mm[2])


def test_at_errors_name_what_went_wrong(grid, one_nucleus, lgn_params):
    _, masks, _ = one_nucleus
    with pytest.raises(ValueError, match='ML,DV,AP'):
        slices.resolve_position(grid, masks, lgn_params, '1,2', None)
    with pytest.raises(ValueError, match='no electrodes'):
        slices.resolve_position(grid, masks, lgn_params, 'electrode:0', None)


# ======================================================================
# Figures
# ======================================================================
@pytest.fixture
def ortho_figure(one_nucleus, lgn_params, grid):
    atlases, masks, contexts = one_nucleus
    centre = slices.tissue_centroid(grid, masks)
    planes = [(plane, grid.nearest_index(plane.normal, centre[plane.normal]))
              for plane in (slices.PLANES['coronal'],
                            slices.PLANES['sagittal'],
                            slices.PLANES['horizontal'])]
    return slices.build_slice_figure(
        lgn_params, atlases, masks, contexts, ['eccentricity', 'layer'],
        planes, None, banner='synthetic', selection='all tissue',
        crosshair=centre, grid=grid)


@pytest.fixture
def montage_figure(one_nucleus, lgn_params, grid):
    atlases, masks, contexts = one_nucleus
    plane = slices.PLANES['coronal']
    planes = [(plane, index)
              for index in slices.slice_indices(grid, masks, plane, 4)]
    electrodes = viz3d.electrodes_from_layout(atlases['left'], lgn_params, 6,
                                              np.random.default_rng(0))
    return slices.build_slice_figure(
        lgn_params, atlases, masks, contexts, ['eccentricity', 'layer'],
        planes, electrodes, banner='synthetic', selection='all tissue',
        grid=grid)


def test_figure_describe_reports_each_plane(ortho_figure):
    text = ortho_figure.describe()
    for name in ('coronal', 'sagittal', 'horizontal'):
        assert name in text
    assert 'crosshair' in text
    assert 'eccentricity' in text


def test_ortho_matplotlib_writes_a_figure(ortho_figure, tmp_path):
    out = slices.render_ortho_matplotlib(ortho_figure, tmp_path / 'o.png')
    assert out.exists() and out.stat().st_size > 0


def test_montage_matplotlib_writes_a_figure(montage_figure, tmp_path):
    out = slices.render_montage_matplotlib(montage_figure, tmp_path / 'm.png')
    assert out.exists() and out.stat().st_size > 0


def test_ortho_plotly_writes_a_figure(ortho_figure, tmp_path):
    pytest.importorskip('plotly')
    out = slices.render_ortho_plotly(ortho_figure, tmp_path / 'o.html',
                                     open_browser=False)
    assert out.exists()
    assert 'heatmap' in out.read_text(encoding='utf-8',
                                      errors='ignore').lower()


def test_montage_plotly_writes_a_figure(montage_figure, tmp_path):
    pytest.importorskip('plotly')
    out = slices.render_montage_plotly(montage_figure, tmp_path / 'm.html',
                                       open_browser=False)
    assert out.exists()


# ----------------------------------------------------------------------
# Regressions: two ways a slider looks right and is wrong
# ----------------------------------------------------------------------
def test_each_slider_step_carries_its_own_slices_values(montage_figure,
                                                        lgn_params):
    """The bug this pins down was silent: the outlines changed with the
    slider while every slice was painted with the opening slice's
    numbers."""
    steps = slices.slider_steps(montage_figure, lgn_params,
                                overlay_traces=[0, 1], contact_trace=None,
                                horizontal=montage_figure.grid.x_mm,
                                vertical=montage_figure.grid.y_mm)
    assert len(steps) == len(montage_figure.slices)

    for step, one in zip(steps, montage_figure.slices):
        for position, name in enumerate(montage_figure.order):
            z = np.array([[np.nan if v is None else v for v in row]
                          for row in step['args'][0]['z'][position]])
            assert z.shape == one.tissue.shape
            # Background stays absent, tissue is present.
            assert np.all(np.isnan(z[~one.tissue]))
            assert np.all(np.isfinite(z[one.tissue]))
            # And the numbers are this slice's own.
            values = one.values[name]
            known = one.tissue & np.isfinite(values)
            assert np.allclose(z[known], values[known], atol=1e-3)


def test_slider_steps_restate_the_millimetre_axes(montage_figure,
                                                  lgn_params):
    """Leaving x and y out of a step that also moves the electrodes
    cleared them, and the panel silently fell back to array indices."""
    grid = montage_figure.grid
    horizontal, vertical = grid.x_mm, grid.y_mm
    steps = slices.slider_steps(montage_figure, lgn_params,
                                overlay_traces=[0, 1], contact_trace=2,
                                horizontal=horizontal, vertical=vertical)
    for step in steps:
        update = step['args'][0]
        assert len(update['x']) == 3                  # two heatmaps, contacts
        for entry in update['x'][:2]:
            assert entry is not None
            assert np.allclose(entry, horizontal)
        for entry in update['y'][:2]:
            assert np.allclose(entry, vertical)


def test_slider_step_labels_are_the_positions(montage_figure, lgn_params):
    steps = slices.slider_steps(montage_figure, lgn_params, [0, 1], None,
                               montage_figure.grid.x_mm,
                               montage_figure.grid.y_mm)
    for step, one in zip(steps, montage_figure.slices):
        assert step['label'] == f'{one.position_mm:+.2f}'


# ======================================================================
# Slice and surface in one figure
# ======================================================================
@pytest.fixture
def combined_figure(one_nucleus, lgn_params, grid):
    atlases, masks, contexts = one_nucleus
    plane = slices.PLANES['coronal']
    planes = [(plane, index)
              for index in slices.slice_indices(grid, masks, plane, 4)]
    electrodes = viz3d.electrodes_from_layout(atlases['left'], lgn_params, 6,
                                              np.random.default_rng(0))
    return slices.build_slice_figure(
        lgn_params, atlases, masks, contexts, ['eccentricity', 'layer'],
        planes, electrodes, banner='synthetic', selection='all tissue',
        grid=grid, shell=slices.build_shell(lgn_params, atlases, masks))


def test_shell_is_one_surface_per_nucleus(both_nuclei, lgn_params):
    atlases, masks, _ = both_nuclei
    shell = slices.build_shell(lgn_params, atlases, masks)
    assert len(shell) == 2
    assert {mesh.hemisphere for mesh in shell} == {'left', 'right'}
    assert all(mesh.n_faces > 0 for mesh in shell)
    # Drawn on opposite sides, like the 3-D viewer draws them.
    assert shell[0].vertices_mm[:, 0].max() < 0
    assert shell[1].vertices_mm[:, 0].min() > 0


@pytest.mark.parametrize('plane_name', ['coronal', 'sagittal', 'horizontal'])
def test_scene_plane_sits_where_the_slice_does(plane_name, grid,
                                               one_nucleus, lgn_params):
    """The cut plane has to be at the slice's own coordinate, on the
    scene axis that plane holds fixed -- otherwise the 3-D half of the
    figure is showing a different slice from the flat half."""
    _, masks, contexts = one_nucleus
    plane = slices.PLANES[plane_name]
    index = int(np.argmax(slices.plane_voxel_counts(grid, masks, plane)))
    one = slices.cut(grid, masks, contexts, ['eccentricity'], plane, index)
    overlay = viz3d.compute_overlay('eccentricity', contexts['left'],
                                    np.argwhere(masks['left'])[:1])
    _, spec = slices.slice_colours(overlay, one.values['eccentricity'],
                                   one.tissue, lgn_params, (0.0, 40.0))

    x, y, z, colour = slices.scene_plane(grid, plane, one, spec, 1)
    constant = (x, y, z)[plane.normal]
    drawn = np.isfinite(z)
    assert drawn.any()
    assert np.allclose(constant[drawn], one.position_mm)

    # The other two axes carry the grid, and the holes follow the
    # tissue exactly.
    assert np.array_equal(drawn, one.tissue)
    across = (x, y, z)[plane.horizontal]
    assert np.allclose(across[0], grid.coordinate(plane.horizontal))


def test_scene_plane_subsamples_without_averaging(grid, one_nucleus,
                                                  lgn_params):
    _, masks, contexts = one_nucleus
    plane = slices.PLANES['coronal']
    index = int(np.argmax(slices.plane_voxel_counts(grid, masks, plane)))
    one = slices.cut(grid, masks, contexts, ['eccentricity'], plane, index)
    overlay = viz3d.compute_overlay('eccentricity', contexts['left'],
                                    np.argwhere(masks['left'])[:1])
    _, spec = slices.slice_colours(overlay, one.values['eccentricity'],
                                   one.tissue, lgn_params, (0.0, 40.0))

    full = slices.scene_plane(grid, plane, one, spec, 1)
    coarse = slices.scene_plane(grid, plane, one, spec, 3)
    assert coarse[3].shape == full[3][::3, ::3].shape
    # Every coarse value is one of the full-resolution ones, in place --
    # a sample of the atlas, not a smoothing of it.
    known = np.isfinite(coarse[3])
    assert np.allclose(coarse[3][known], full[3][::3, ::3][known])


def test_combined_needs_a_shell(montage_figure, tmp_path):
    pytest.importorskip('plotly')
    assert montage_figure.shell is None
    with pytest.raises(ValueError, match='nucleus surface'):
        slices.render_combined_plotly(montage_figure, tmp_path / 'c.html')


def test_combined_panel_is_anchored_to_an_axis_that_exists(combined_figure):
    """A scene in the first cell consumes no x axis, so the panel beside
    it is the figure's first xy subplot. Anchoring its aspect ratio to
    'x2' leaves the panel blank and says nothing about why."""
    pytest.importorskip('plotly')
    handle = slices.build_combined_figure(combined_figure)
    anchor = handle.layout.yaxis.scaleanchor
    assert anchor is not None
    axis = 'xaxis' if anchor == 'x' else f"xaxis{anchor.lstrip('x')}"
    assert handle.layout[axis] is not None
    assert handle.layout.yaxis.scaleratio == 1

    panels = [t for t in handle.data if t.type == 'heatmap']
    assert panels and all(t.xaxis in (None, anchor) for t in panels)


def test_combined_slider_restyles_rather_than_animates(combined_figure):
    """plotly.js cannot animate a heatmap, so a frame would move the
    cut plane and leave the flat panel undrawn. The steps have to be
    restyles, and there must be no frames to fall back on."""
    pytest.importorskip('plotly')
    handle = slices.build_combined_figure(combined_figure)
    assert not handle.frames
    steps = handle.layout.sliders[0].steps
    assert len(steps) == len(combined_figure.slices)
    assert all(step.method == 'update' for step in steps)


def test_combined_steps_carry_their_own_slice(combined_figure, lgn_params):
    """The same trap the montage slider fell into: outlines that move
    while the numbers stay behind."""
    pytest.importorskip('plotly')
    handle = slices.build_combined_figure(combined_figure)
    kinds = {i: t.type for i, t in enumerate(handle.data)}

    for step, one in zip(handle.layout.sliders[0].steps,
                         combined_figure.slices):
        restyle, layout, targets = step.args
        panels = [restyle['z'][position]
                  for position, index in enumerate(targets)
                  if kinds[index] == 'heatmap']
        assert len(panels) == len(combined_figure.order)

        # Panels are targeted in overlay order, so the nth is the nth.
        for grid_values, name in zip(panels, combined_figure.order):
            values = one.values[name]
            z = np.array([[np.nan if v is None else v for v in row]
                          for row in grid_values])
            assert z.shape == one.tissue.shape
            assert np.all(np.isnan(z[~one.tissue]))
            known = one.tissue & np.isfinite(values)
            assert np.allclose(z[known], values[known], atol=1e-3)
        assert f'{one.position_mm:+.2f}' in layout['title.text']


def test_combined_steps_move_the_plane_to_the_slice(combined_figure):
    """The 3-D half and the flat half must be showing the same cut."""
    pytest.importorskip('plotly')
    handle = slices.build_combined_figure(combined_figure)
    kinds = {i: t.type for i, t in enumerate(handle.data)}
    normal = combined_figure.slices[0].plane.normal

    for step, one in zip(handle.layout.sliders[0].steps,
                         combined_figure.slices):
        restyle, _, targets = step.args
        axis = ('x', 'y', 'z')[normal]
        for position, index in enumerate(targets):
            if kinds[index] != 'surface':
                continue
            drawn = [v for row in restyle[axis][position] for v in row
                     if v is not None]
            assert drawn
            assert np.allclose(drawn, one.position_mm)


def test_combined_steps_never_clear_an_attribute_a_trace_uses(
        combined_figure):
    """None means 'this trace type has no such attribute, ignore it'.
    Sent for one a trace DOES have, it clears it -- which is how the
    millimetre axes went missing once."""
    pytest.importorskip('plotly')
    handle = slices.build_combined_figure(combined_figure)
    kinds = {i: t.type for i, t in enumerate(handle.data)}
    has = {'surface': {'x', 'y', 'z', 'surfacecolor'},
           'heatmap': {'x', 'y', 'z'},
           'scatter3d': {'x', 'y', 'z'},
           'scatter': {'x', 'y'}}

    for step in handle.layout.sliders[0].steps:
        restyle, _, targets = step.args
        for position, index in enumerate(targets):
            kind = kinds[index]
            for key, values in restyle.items():
                if key == 'customdata':
                    continue
                if key in has[kind]:
                    assert values[position] is not None, (kind, key)


def test_combined_slider_opens_on_the_middle_slice(combined_figure):
    """The ends of the nucleus are its thin tips; opening on one looks
    like a broken figure."""
    pytest.importorskip('plotly')
    handle = slices.build_combined_figure(combined_figure)
    middle = len(combined_figure.slices) // 2
    slider = handle.layout.sliders[0]
    assert slider.active == middle
    assert slider.steps[middle].label == (
        f'{combined_figure.slices[middle].position_mm:+.2f}')

    opening = combined_figure.slices[middle]
    panel = [t for t in handle.data if t.type == 'heatmap'][0]
    assert np.array_equal(np.isfinite(np.asarray(panel.z, dtype=float)),
                          opening.tissue)


def test_combined_dropdown_only_toggles_visibility(combined_figure):
    """So that changing the overlay does not throw away the slice you
    are on, and the slider does not throw away the overlay."""
    pytest.importorskip('plotly')
    handle = slices.build_combined_figure(combined_figure)
    menus = [m for m in handle.layout.updatemenus
             if any('Eccentricity' in (b.label or '') for b in m.buttons)]
    assert menus
    for button in menus[0].buttons:
        assert set(button.args[0]) == {'visible'}
        assert len(button.args[0]['visible']) == len(handle.data)

    # Exactly one surface and one heatmap on at a time; the shell and
    # the electrodes always on.
    for position, button in enumerate(menus[0].buttons):
        visible = button.args[0]['visible']
        shown = [i for i, flag in enumerate(handle.data) if visible[i]]
        by_kind = {}
        for i in shown:
            by_kind.setdefault(handle.data[i].type, []).append(i)
        assert len(by_kind.get('surface', [])) == 1
        assert len(by_kind.get('heatmap', [])) == 1
        assert len(by_kind.get('mesh3d', [])) == len(combined_figure.shell)


def test_combined_plotly_writes_a_figure(combined_figure, tmp_path):
    pytest.importorskip('plotly')
    out = slices.render_combined_plotly(combined_figure, tmp_path / 'c.html',
                                        open_browser=False)
    assert out.exists()
    text = out.read_text(encoding='utf-8', errors='ignore').lower()
    assert 'surface' in text and 'mesh3d' in text


def test_combined_matplotlib_writes_a_figure(combined_figure, tmp_path):
    out = slices.render_combined_matplotlib(combined_figure,
                                            tmp_path / 'c.png')
    assert out.exists() and out.stat().st_size > 0


def test_combined_describe_mentions_the_shell(combined_figure):
    assert 'shell' in combined_figure.describe()


# ======================================================================
# Command line
# ======================================================================
def test_cli_lists_overlays(capsys):
    assert slices.main(['--list-overlays']) == 0
    assert 'eccentricity' in capsys.readouterr().out


def test_cli_rejects_an_unknown_overlay(capsys):
    assert slices.main(['--synthetic', '--overlay', 'nonsense']) == 2
    assert 'nonsense' in capsys.readouterr().err


def test_cli_writes_an_ortho_figure(tmp_path, capsys):
    out = tmp_path / 'ortho.png'
    code = slices.main(['--synthetic', '--backend', 'matplotlib',
                        '--overlay', 'layer', '--out', str(out)])
    assert code == 0
    assert out.exists()
    printed = capsys.readouterr().out
    assert 'coronal' in printed and 'sagittal' in printed


def test_cli_writes_a_montage(tmp_path, capsys):
    out = tmp_path / 'montage.png'
    code = slices.main(['--synthetic', '--backend', 'matplotlib',
                        '--view', 'slices', '--plane', 'horizontal',
                        '--count', '4', '--overlay', 'eccentricity',
                        '--out', str(out)])
    assert code == 0
    assert out.exists()
    assert capsys.readouterr().out.count('horizontal') >= 4


def test_cli_writes_a_combined_view(tmp_path, capsys):
    out = tmp_path / 'combined.png'
    code = slices.main(['--synthetic', '--backend', 'matplotlib',
                        '--view', 'combined', '--plane', 'sagittal',
                        '--count', '3', '--overlay', 'layer',
                        '--electrodes', 'layout', '--n-electrodes', '4',
                        '--out', str(out)])
    assert code == 0
    assert out.exists()
    printed = capsys.readouterr().out
    assert 'shell' in printed
    assert printed.count('sagittal') >= 3


def test_cli_electrodes_reach_the_panels(tmp_path, capsys):
    out = tmp_path / 'ortho.png'
    code = slices.main(['--synthetic', '--backend', 'matplotlib',
                        '--electrodes', 'layout', '--n-electrodes', '4',
                        '--at', 'electrode:0', '--out', str(out)])
    assert code == 0
    assert 'electrodes' in capsys.readouterr().out


def test_cli_slice_overrides_reach_the_config():
    options = slices.build_parser().parse_args(
        ['--plane', 'sagittal', '--count', '5', '--slab', '0.4',
         '--no-crosshair', '--range', '3:6'])
    params = slices._apply_slice_overrides(
        {'visualization': {'slices': {}, 'combined': {}}}, options)
    block = params['visualization']['slices']
    assert block['plane'] == 'sagittal'
    # --plane means the plane you are looking at, in either view.
    assert params['visualization']['combined']['plane'] == 'sagittal'
    assert block['count'] == 5
    assert block['electrode_slab_mm'] == 0.4
    assert block['crosshair'] is False
    assert block['range_mm'] == [3.0, 6.0]


def test_cli_range_errors_are_readable():
    options = slices.build_parser().parse_args(['--range', 'nonsense'])
    with pytest.raises(ValueError, match='LOW:HIGH'):
        slices._apply_slice_overrides({'visualization': {'slices': {}}},
                                      options)


def test_the_slice_viewer_shares_the_3d_viewers_flags():
    """Both programs build their shared options from one place, so a
    flag cannot come to mean two things."""
    shared = {'--params', '--atlas-dir', '--synthetic', '--hemispheres',
              '--overlay', '--cell-class', '--layers', '--clip',
              '--electrodes', '--probe', '--grid', '--colour-by',
              '--backend', '--out'}
    volume = {action for option in viz3d.build_parser()._actions
              for action in option.option_strings}
    plane = {action for option in slices.build_parser()._actions
             for action in option.option_strings}
    assert shared <= volume
    assert shared <= plane


def test_the_slice_viewer_drops_the_surface_only_flags():
    plane = {action for option in slices.build_parser()._actions
             for action in option.option_strings}
    # A slice is one voxel thick and fully opaque, so an opacity would
    # do nothing here -- it is not offered rather than being accepted
    # and ignored.
    assert '--surface-opacity' not in plane
    # A stride IS offered, because --view combined draws the nucleus
    # around the slice and that surface has a resolution to choose.
    assert '--stride' in plane


def test_the_ortho_and_montage_views_ignore_the_stride(tmp_path):
    """It only reaches a surface, and those two views have none -- so
    passing it must not change what they draw."""
    out = tmp_path / 'ortho.png'
    for stride in ('1', '8'):
        code = slices.main(['--synthetic', '--backend', 'matplotlib',
                            '--overlay', 'layer', '--stride', stride,
                            '--out', str(out)])
        assert code == 0
        assert out.exists()


@requires_real_atlas
def test_real_atlas_coronal_slice_shows_the_laminae(real_atlas, lgn_params):
    """On the published atlas a mid-coronal cut must contain all four
    laminar codes -- that is what a slice through the LGN looks like,
    and a plane read off the wrong axis would not."""
    masks = {'left': viz3d.tissue_mask(real_atlas, lgn_params)}
    contexts = {'left': viz3d.AtlasContext(real_atlas, lgn_params)}
    grid = slices.build_scene_grid(lgn_params, {'left': real_atlas}, masks)
    plane = slices.PLANES['coronal']
    index = slices.slice_indices(grid, masks, plane, 1)[0]
    one = slices.cut(grid, masks, contexts, ['layer'], plane, index)

    present = np.unique(one.values['layer'][one.tissue])
    assert len(present) == 4
    assert one.n_voxels > 5000
