"""Tests for the 3-D atlas viewer.

The geometry is checked against shapes whose surface area is known by
hand, the coordinates against the atlas's own inverse lookup, and the
overlays against the numbers they claim to be showing -- because a
picture that is wrong still looks like a picture.

Everything runs on the synthetic atlas, so no download is needed.
"""
import numpy as np
import pytest

from dynaphos_lgn import viz3d
from dynaphos_lgn.test.conftest import requires_real_atlas


# ======================================================================
# Geometry
# ======================================================================
def test_solid_cube_has_six_faces_per_side():
    """A 2x2x2 solid has 24 exposed faces, and no interior ones."""
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    block_mask, _ = viz3d.blocks_of(mask, 1)
    corners, owners = viz3d.exposed_faces(block_mask)
    assert len(corners) == 24
    assert len(owners) == 24


def test_hollow_shell_exposes_inner_and_outer_surfaces():
    """A cavity is a surface too: the inner faces must be drawn, which
    is what makes a clipped volume show its cut face."""
    mask = np.ones((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = False
    block_mask, _ = viz3d.blocks_of(mask, 1)
    corners, _ = viz3d.exposed_faces(block_mask)
    # 6 outer faces of 25 voxels each, plus the 6 faces around the hole.
    assert len(corners) == 6 * 25 + 6


def test_blocks_keep_thin_structures():
    """A one-voxel lamina must survive a stride that steps over it."""
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[:, 3, :] = True                 # one voxel thick, at the far
                                         # edge of the first DV block
    block_mask, representative = viz3d.blocks_of(mask, 4)
    assert block_mask[:, 0, :].all()     # the lamina's own blocks: kept
    assert not block_mask[:, 1, :].any()  # the empty ones: still empty
    # Every representative must be a real tissue voxel, not just the
    # block's centre, which here holds nothing.
    assert mask[tuple(representative[block_mask].T)].all()


def test_block_representative_is_the_most_central_tissue_voxel():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = True          # a corner of the block
    mask[2, 2, 2] = True          # and its centre
    _, representative = viz3d.blocks_of(mask, 4)
    assert tuple(representative[0, 0, 0]) == (2, 2, 2)


def test_empty_mask_gives_an_empty_mesh(synthetic_atlas):
    mesh = viz3d.build_mesh(synthetic_atlas,
                            np.zeros(synthetic_atlas.atlas_shape, dtype=bool),
                            2, 1.0)
    assert mesh.n_faces == 0
    assert mesh.n_blocks == 0


def test_mesh_spans_the_same_millimetres_as_the_atlas(synthetic_atlas,
                                                      lgn_params):
    """Vertices must land within half a voxel of the tissue's own
    bounding box -- a voxel's cube reaches half a voxel past its
    centre, and no further."""
    mask = viz3d.tissue_mask(synthetic_atlas, lgn_params)
    mesh = viz3d.build_mesh(synthetic_atlas, mask, 1, 1.0)

    index = np.argwhere(mask)
    centres = np.stack(synthetic_atlas.index_to_horsley_clarke(
        index[:, 0], index[:, 1], index[:, 2]), axis=-1)
    half = synthetic_atlas._voxel_size_mm / 2

    assert np.allclose(mesh.vertices_mm.min(axis=0),
                       centres.min(axis=0) - half, atol=1e-6)
    assert np.allclose(mesh.vertices_mm.max(axis=0),
                       centres.max(axis=0) + half, atol=1e-6)


def test_mesh_x_sign_places_the_nucleus_on_one_side(synthetic_atlas,
                                                    lgn_params):
    mask = viz3d.tissue_mask(synthetic_atlas, lgn_params)
    left = viz3d.build_mesh(synthetic_atlas, mask, 4, -1.0)
    right = viz3d.build_mesh(synthetic_atlas, mask, 4, +1.0)
    assert left.vertices_mm[:, 0].max() < 0
    assert right.vertices_mm[:, 0].min() > 0
    assert np.allclose(left.vertices_mm[:, 0], -right.vertices_mm[:, 0])


def test_centres_are_real_voxel_positions(synthetic_atlas, lgn_params):
    """Each drawn voxel's hover position must be the atlas voxel it
    took its value from, not the block's geometric centre."""
    mask = viz3d.tissue_mask(synthetic_atlas, lgn_params)
    mesh = viz3d.build_mesh(synthetic_atlas, mask, 3, 1.0)
    expected = np.stack(synthetic_atlas.index_to_horsley_clarke(
        mesh.voxel_indices[:, 0], mesh.voxel_indices[:, 1],
        mesh.voxel_indices[:, 2]), axis=-1)
    assert np.allclose(mesh.centres_mm, expected, atol=1e-5)
    assert mask[tuple(mesh.voxel_indices.T)].all()


def test_triangles_index_real_vertices(synthetic_atlas, lgn_params):
    mask = viz3d.tissue_mask(synthetic_atlas, lgn_params)
    mesh = viz3d.build_mesh(synthetic_atlas, mask, 4, 1.0)
    triangles = mesh.triangles
    assert len(triangles) == 2 * mesh.n_faces
    assert triangles.max() < len(mesh.vertices_mm)
    assert len(mesh.block_of_triangle) == len(triangles)
    assert mesh.block_of_quad.max() < mesh.n_blocks


def test_axis_mm_matches_the_atlas_inverse(synthetic_atlas):
    for axis in range(3):
        coordinate = viz3d.axis_mm(synthetic_atlas, axis)
        index = [0, 0, 0]
        index[axis] = 5
        expected = synthetic_atlas.index_to_horsley_clarke(*index)[axis]
        assert coordinate[5] == pytest.approx(expected)


def test_mirrored_nucleus_is_the_reflection(synthetic_atlas, lgn_params):
    """The right nucleus must occupy the mirror-image millimetres, not
    the left one's."""
    from dynaphos_lgn.atlas import MirroredAtlas
    mirrored = MirroredAtlas(synthetic_atlas, lgn_params)

    left = viz3d.build_mesh(synthetic_atlas,
                            viz3d.tissue_mask(synthetic_atlas, lgn_params),
                            4, -1.0)
    right = viz3d.build_mesh(mirrored,
                             viz3d.tissue_mask(mirrored, lgn_params), 4, +1.0)
    assert left.n_faces == right.n_faces
    # Lateral distance from the midline is the same on both sides.
    assert np.allclose(np.abs(left.vertices_mm[:, 0]).min(),
                       np.abs(right.vertices_mm[:, 0]).min(), atol=1e-6)
    assert np.allclose(np.abs(left.vertices_mm[:, 0]).max(),
                       np.abs(right.vertices_mm[:, 0]).max(), atol=1e-6)


# ======================================================================
# Selecting tissue
# ======================================================================
def test_tissue_mask_defaults_to_what_the_simulator_uses(synthetic_atlas,
                                                          lgn_params):
    mask = viz3d.tissue_mask(synthetic_atlas, lgn_params)
    assert mask.sum() <= synthetic_atlas.valid.sum()
    if lgn_params['atlas']['exclude_ipsi_flat_inclination']:
        assert not (mask & synthetic_atlas.is_ipsi_sentinel()).any()


def test_cell_class_selection(synthetic_atlas, lgn_params):
    codes = lgn_params['atlas']['layers']['classes']['magno']
    mask = viz3d.tissue_mask(synthetic_atlas, lgn_params, cell_class='magno')
    assert mask.any()
    assert np.isin(synthetic_atlas.layer[mask], codes).all()


def test_unknown_cell_class_names_the_valid_ones(synthetic_atlas,
                                                 lgn_params):
    with pytest.raises(ValueError, match='magno'):
        viz3d.tissue_mask(synthetic_atlas, lgn_params, cell_class='nonsense')


def test_clip_cuts_at_the_requested_millimetres(synthetic_atlas,
                                                lgn_params):
    coordinate = viz3d.axis_mm(synthetic_atlas, 2)
    low = float(np.percentile(coordinate, 30))
    high = float(np.percentile(coordinate, 60))
    mask = viz3d.tissue_mask(synthetic_atlas, lgn_params,
                             clips=[('ap', low, high)])
    kept = np.argwhere(mask)
    assert len(kept)
    assert coordinate[kept[:, 2]].min() >= low
    assert coordinate[kept[:, 2]].max() <= high


def test_eccentricity_bounds(synthetic_atlas, lgn_params):
    mask = viz3d.tissue_mask(synthetic_atlas, lgn_params,
                             min_eccentricity_deg=2.0,
                             max_eccentricity_deg=8.0)
    values = synthetic_atlas.eccentricity_deg[mask]
    assert values.min() >= 2.0
    assert values.max() <= 8.0


# ======================================================================
# Overlays
# ======================================================================
@pytest.fixture
def atlas_scene(synthetic_atlas, synthetic_jacobian, lgn_params):
    """A small scene with every overlay evaluated."""
    return viz3d.build_scene(
        lgn_params, {'left': synthetic_atlas},
        jacobian_for=lambda _: synthetic_jacobian,
        overlay_names=list(viz3d.OVERLAY_BUILDERS),
        banner='synthetic')


def test_every_overlay_evaluates(atlas_scene):
    for name in viz3d.OVERLAY_BUILDERS:
        overlay = atlas_scene.overlays[name][0]
        assert len(overlay.values) == atlas_scene.meshes[0].n_blocks
        assert overlay.kind in ('continuous', 'categorical')
        assert overlay.note, f'{name} should say what it is'


def test_continuous_overlays_carry_the_atlas_numbers(atlas_scene,
                                                     synthetic_atlas):
    mesh = atlas_scene.meshes[0]
    index = tuple(mesh.voxel_indices.T)
    assert np.allclose(atlas_scene.overlays['eccentricity'][0].values,
                       synthetic_atlas.eccentricity_deg[index])
    assert np.allclose(atlas_scene.overlays['cells_per_voxel'][0].values,
                       synthetic_atlas.cells_per_voxel[index])


def test_categorical_overlays_index_their_own_categories(atlas_scene):
    for name in ('cell_class', 'layer', 'eye', 'jacobian_flags'):
        overlay = atlas_scene.overlays[name][0]
        assert overlay.kind == 'categorical'
        finite = overlay.values[np.isfinite(overlay.values)]
        assert finite.min() >= 0
        assert finite.max() < len(overlay.categories)
        assert np.all(finite == np.round(finite))


def test_layer_overlay_matches_the_configured_names(atlas_scene,
                                                    synthetic_atlas,
                                                    lgn_params):
    mesh = atlas_scene.meshes[0]
    overlay = atlas_scene.overlays['layer'][0]
    names = lgn_params['atlas']['layers']['names']
    raw = synthetic_atlas.layer[tuple(mesh.voxel_indices.T)]
    for i in range(0, mesh.n_blocks, max(mesh.n_blocks // 50, 1)):
        if np.isfinite(overlay.values[i]):
            assert (overlay.categories[int(overlay.values[i])]
                    == names[int(raw[i])])


def test_magnification_overlays_agree_with_the_magnification_model(
        atlas_scene, synthetic_jacobian, lgn_params):
    """The picture and the simulator must read the same Jacobian."""
    from dynaphos_lgn.magnification import JacobianMagnification
    mesh = atlas_scene.meshes[0]
    model = JacobianMagnification(synthetic_jacobian, lgn_params)
    local = model.at_voxels(mesh.voxel_indices)

    for name, expected in (('magnification_major', local.major_deg_per_mm),
                           ('magnification_minor', local.minor_deg_per_mm),
                           ('anisotropy', local.anisotropy)):
        values = atlas_scene.overlays[name][0].values
        both = np.isfinite(values) & np.isfinite(expected)
        assert both.any()
        assert np.allclose(values[both], np.asarray(expected)[both],
                           rtol=1e-5)


def test_inclination_hides_the_ipsilateral_placeholder(synthetic_atlas,
                                                       lgn_params):
    """The coarse +-135 deg marker is not an angle, and must not be
    painted as one."""
    import copy
    params = copy.deepcopy(lgn_params)
    params['atlas']['exclude_ipsi_flat_inclination'] = False
    scene = viz3d.build_scene(params, {'left': synthetic_atlas}, None,
                              ['inclination'],
                              mask_options=dict(include_ipsi_placeholder=True))
    mesh = scene.meshes[0]
    placeholder = synthetic_atlas.is_ipsi_sentinel(
        synthetic_atlas.inclination[tuple(mesh.voxel_indices.T)])
    if placeholder.any():
        values = scene.overlays['inclination'][0].values
        assert not np.isfinite(values[placeholder]).any()


def test_jacobian_overlays_are_empty_without_a_jacobian(synthetic_atlas,
                                                        lgn_params):
    scene = viz3d.build_scene(lgn_params, {'left': synthetic_atlas}, None,
                              ['magnification_major', 'jacobian_residual'])
    for name in ('magnification_major', 'jacobian_residual'):
        assert not np.isfinite(scene.overlays[name][0].values).any()


def test_unknown_overlay_lists_the_known_ones(synthetic_atlas, lgn_params):
    context = viz3d.AtlasContext(synthetic_atlas, lgn_params)
    with pytest.raises(KeyError, match='eccentricity'):
        viz3d.compute_overlay('lamina_thickness', context,
                              np.zeros((1, 3), dtype=int))


# ======================================================================
# Colour
# ======================================================================
def test_missing_values_get_their_own_band(atlas_scene, lgn_params):
    """Grey must sit below the real range, never inside it, so "no
    value" cannot be read as a small value."""
    overlay = atlas_scene.overlays['jacobian_residual'][0]
    spec = viz3d.colour_spec(overlay, lgn_params)
    low, high = viz3d.continuous_limits(overlay, lgn_params)
    assert spec.cmin < low
    assert spec.cmax == pytest.approx(high)
    missing = ~np.isfinite(overlay.values)
    if missing.any():
        assert np.all(spec.values[missing] < low)


def test_categorical_colours_are_all_distinct(atlas_scene, lgn_params):
    """The last entry of a qualitative colormap is a neutral grey, and
    must not collide with the missing-value grey."""
    spec = viz3d.colour_spec(atlas_scene.overlays['layer'][0], lgn_params)
    colours = {tuple(np.round(row, 4)) for row in spec.rgba}
    categories = set(atlas_scene.overlays['layer'][0].values[
        np.isfinite(atlas_scene.overlays['layer'][0].values)].astype(int))
    assert len(colours) >= len(categories)

    import matplotlib.colors as mcolors
    missing = tuple(np.round(
        mcolors.to_rgba(lgn_params['visualization']['missing_colour']), 4))
    present = atlas_scene.overlays['layer'][0].values
    assert missing not in {tuple(np.round(row, 4))
                           for row in spec.rgba[np.isfinite(present)]}


def test_colour_limits_clip_the_configured_overlays(atlas_scene,
                                                    lgn_params):
    overlay = atlas_scene.overlays['anisotropy'][0]
    low, high = viz3d.continuous_limits(overlay, lgn_params)
    finite = overlay.finite
    if len(finite) > 50:
        assert low >= finite.min()
        assert high <= finite.max()


def test_rgba_has_one_colour_per_drawn_voxel(atlas_scene, lgn_params):
    for name in atlas_scene.order:
        spec = viz3d.colour_spec(atlas_scene.overlays[name][0], lgn_params)
        assert spec.rgba.shape == (atlas_scene.meshes[0].n_blocks, 4)
        assert len(spec.values) == atlas_scene.meshes[0].n_blocks


# ======================================================================
# Electrodes
# ======================================================================
def test_layout_electrodes_sit_on_tissue(synthetic_atlas, lgn_params):
    rng = np.random.default_rng(0)
    electrodes = viz3d.electrodes_from_layout(synthetic_atlas, lgn_params, 6,
                                              rng)
    assert len(electrodes) == 6
    assert synthetic_atlas.valid[tuple(electrodes.voxel_indices.T)].all()
    assert not electrodes.outside.any()
    assert len(electrodes.hover) == 6


def test_built_array_electrodes_carry_their_flags(built, lgn_params):
    _, array = built
    electrodes = viz3d.electrodes_from_array(array, lgn_params)
    assert len(electrodes) == array.n_electrodes
    assert all('renderer' in text for text in electrodes.hover)
    assert all('flags' in text for text in electrodes.hover)


def test_probe_spreads_contacts_along_the_shank(synthetic_atlas,
                                                lgn_params):
    electrodes = viz3d.probe_contacts(
        'entry=9.0,0.5,4.0;target=9.0,1.5,4.0;n=5',
        {'left': synthetic_atlas}, lgn_params, 'left')
    assert len(electrodes) == 5
    spacing = np.diff(electrodes.positions_mm, axis=0)
    assert np.allclose(spacing, spacing[0])
    assert np.allclose(np.linalg.norm(spacing, axis=1), 1.0 / 4)


def test_probe_spacing_overrides_the_target_distance(synthetic_atlas,
                                                     lgn_params):
    electrodes = viz3d.probe_contacts(
        'entry=9.0,0.5,4.0;target=9.0,5.0,4.0;n=4;spacing=0.2',
        {'left': synthetic_atlas}, lgn_params, 'left')
    step = np.linalg.norm(np.diff(electrodes.positions_mm, axis=0), axis=1)
    assert np.allclose(step, 0.2)


def test_probe_spec_errors_name_what_is_missing(synthetic_atlas,
                                                lgn_params):
    with pytest.raises(ValueError, match='target'):
        viz3d.probe_contacts('entry=9,0,4;n=4', {'left': synthetic_atlas},
                             lgn_params, 'left')
    with pytest.raises(ValueError, match='key=value'):
        viz3d.probe_contacts('entry 9,0,4', {'left': synthetic_atlas},
                             lgn_params, 'left')


def test_grid_lays_contacts_out_in_a_plane(synthetic_atlas, lgn_params):
    electrodes = viz3d.grid_contacts(
        'centre=9.0,1.0,4.0;plane=dv,ap;n=3,4;pitch=0.2,0.3',
        {'left': synthetic_atlas}, lgn_params, 'left')
    assert len(electrodes) == 12
    # The insertion axis is untouched; the other two span n * pitch.
    assert np.allclose(np.abs(electrodes.positions_mm[:, 0]), 9.0)
    assert electrodes.positions_mm[:, 1].ptp() == pytest.approx(0.4)
    assert electrodes.positions_mm[:, 2].ptp() == pytest.approx(0.9)


def test_grid_rejects_a_repeated_axis(synthetic_atlas, lgn_params):
    with pytest.raises(ValueError, match='two different axes'):
        viz3d.grid_contacts('centre=9,1,4;plane=dv,dv;n=2,2;pitch=0.1,0.1',
                            {'left': synthetic_atlas}, lgn_params, 'left')


def test_contacts_outside_the_atlas_are_flagged(synthetic_atlas,
                                                lgn_params):
    electrodes = viz3d.probe_contacts(
        'entry=9.0,1.0,4.0;target=90.0,1.0,4.0;n=3',
        {'left': synthetic_atlas}, lgn_params, 'left')
    assert electrodes.outside.any()
    assert any('OUTSIDE' in text for text in electrodes.hover)


def test_electrodes_from_csv(tmp_path, synthetic_atlas, lgn_params):
    path = tmp_path / 'contacts.csv'
    path.write_text('label,ml_mm,dv_mm,ap_mm\n'
                    'a,9.0,0.6,4.0\n'
                    'b,9.2,0.8,4.2\n')
    electrodes = viz3d.electrodes_from_file(path, {'left': synthetic_atlas},
                                            lgn_params, 'mm', 'left')
    assert electrodes.labels == ['a', 'b']
    assert np.allclose(np.abs(electrodes.positions_mm[0]), [9.0, 0.6, 4.0])


def test_electrodes_from_csv_in_voxel_units(tmp_path, synthetic_atlas,
                                            lgn_params):
    path = tmp_path / 'contacts.csv'
    path.write_text('ml,dv,ap\n20,20,40\n')
    electrodes = viz3d.electrodes_from_file(path, {'left': synthetic_atlas},
                                            lgn_params, 'vox', 'left')
    assert np.array_equal(electrodes.voxel_indices[0], [20, 20, 40])


def test_electrodes_from_csv_names_the_missing_column(tmp_path,
                                                      synthetic_atlas,
                                                      lgn_params):
    path = tmp_path / 'contacts.csv'
    path.write_text('x,y,z\n1,2,3\n')
    with pytest.raises(ValueError, match='ml'):
        viz3d.electrodes_from_file(path, {'left': synthetic_atlas},
                                   lgn_params, 'mm', 'left')


def test_contacts_are_described_by_their_own_nucleus(synthetic_atlas,
                                                     lgn_params):
    """A voxel index means a different voxel in the two atlases, so a
    right-nucleus contact must be looked up in the right nucleus."""
    from dynaphos_lgn.atlas import MirroredAtlas
    mirrored = MirroredAtlas(synthetic_atlas, lgn_params)
    atlases = {'left': synthetic_atlas, 'right': mirrored}

    left = viz3d.probe_contacts(
        'entry=9.0,0.5,4.0;target=9.4,1.2,4.6;n=3;hemisphere=left',
        atlases, lgn_params, 'left')
    right = viz3d.probe_contacts(
        'entry=9.0,0.5,4.0;target=9.4,1.2,4.6;n=3;hemisphere=right',
        atlases, lgn_params, 'left')

    # Same stereotaxic coordinates, mirror-image nuclei: the two must
    # sit on opposite sides of the scene but at the same lateral
    # distance, and each must read its own atlas.
    assert np.allclose(left.positions_mm[:, 0], -right.positions_mm[:, 0])
    index = tuple(right.voxel_indices.T)
    for i, text in enumerate(right.hover):
        if mirrored.valid[index][i]:
            assert (f'{mirrored.inclination_deg[index][i]:.1f} deg' in text)


def test_mixed_hemisphere_file_reads_each_side_correctly(tmp_path,
                                                         synthetic_atlas,
                                                         lgn_params):
    from dynaphos_lgn.atlas import MirroredAtlas
    atlases = {'left': synthetic_atlas,
               'right': MirroredAtlas(synthetic_atlas, lgn_params)}
    path = tmp_path / 'both.csv'
    path.write_text('label,hemisphere,ml_mm,dv_mm,ap_mm\n'
                    'L0,left,9.0,0.6,4.0\n'
                    'R0,right,9.0,0.6,4.0\n')
    electrodes = viz3d.electrodes_from_file(path, atlases, lgn_params, 'mm',
                                            'left')
    assert electrodes.positions_mm[0, 0] == -electrodes.positions_mm[1, 0]
    assert 'left LGN' in electrodes.hover[0]
    assert 'right LGN' in electrodes.hover[1]


def test_a_contact_in_an_undrawn_nucleus_says_so(synthetic_atlas,
                                                 lgn_params):
    with pytest.raises(ValueError, match='--hemispheres'):
        viz3d.probe_contacts(
            'entry=9,0.5,4;target=9,1,4;n=2;hemisphere=right',
            {'left': synthetic_atlas}, lgn_params, 'left')


def test_merge_keeps_one_colour_per_named_group(synthetic_atlas,
                                                lgn_params):
    """Two layouts coloured by lamina must share the lamina's colour,
    not get one legend entry each."""
    import copy
    params = copy.deepcopy(lgn_params)
    params['visualization']['electrodes']['colour_by'] = 'layer'
    one = viz3d.probe_contacts('entry=9.0,0.5,4.0;target=9.0,1.5,4.0;n=4',
                               {'left': synthetic_atlas}, params, 'left')
    two = viz3d.grid_contacts(
        'centre=9.0,1.0,4.5;plane=dv,ap;n=2,2;pitch=0.2,0.2',
        {'left': synthetic_atlas}, params, 'left')
    merged = viz3d.merge_electrodes([one, two])

    assert len(merged) == len(one) + len(two)
    assert len(merged.group_names) == len(set(merged.group_names))
    assert merged.group.max() < len(merged.group_names)
    for original, offset in ((one, 0), (two, len(one))):
        for i in range(len(original)):
            assert (merged.group_names[merged.group[offset + i]]
                    == original.group_names[original.group[i]])


def test_recruitment_radius_comes_from_the_kernel(lgn_params):
    from dynaphos_lgn.current_spread import RecruitmentKernel
    radius, label = viz3d.recruitment_radius_mm(lgn_params, 3)
    kernel = RecruitmentKernel.from_params(lgn_params)
    current = lgn_params['visualization']['electrodes']['spread_current_ua']
    assert np.allclose(radius, float(kernel.radius_mm(current)))
    assert 'uA' in label


def test_unit_sphere_is_a_closed_surface():
    vertices, triangles = viz3d.unit_sphere(8)
    assert np.allclose(np.linalg.norm(vertices, axis=1), 1.0)
    assert triangles.max() < len(vertices)


# ======================================================================
# Scene and backends
# ======================================================================
def test_scene_describe_reports_what_was_drawn(atlas_scene):
    text = atlas_scene.describe()
    assert 'left LGN' in text
    assert 'eccentricity' in text
    assert 'surface voxels' in text


def test_hover_text_lists_every_overlay(atlas_scene):
    blocks = np.arange(min(5, atlas_scene.meshes[0].n_blocks))
    lines = viz3d.hover_text(atlas_scene, 0, blocks)
    assert len(lines) == len(blocks)
    for name in atlas_scene.order:
        assert atlas_scene.overlays[name][0].label in lines[0]


def test_matplotlib_backend_writes_a_figure(atlas_scene, tmp_path):
    out = viz3d.render_matplotlib(atlas_scene, tmp_path / 'lgn.png')
    assert out.exists()
    assert out.stat().st_size > 0


def test_plotly_backend_writes_a_figure(atlas_scene, tmp_path):
    pytest.importorskip('plotly')
    out = viz3d.render_plotly(atlas_scene, tmp_path / 'lgn.html',
                              open_browser=False)
    assert out.exists()
    text = out.read_text(encoding='utf-8', errors='ignore')
    assert 'mesh3d' in text.lower()


def test_plotly_dropdown_recolours_one_mesh_rather_than_adding_copies(
        atlas_scene, lgn_params):
    """The geometry must be stored once however many overlays there
    are; otherwise the file grows by a full mesh per overlay."""
    pytest.importorskip('plotly')
    specs = {name: [viz3d.colour_spec(o, lgn_params)
                    for o in atlas_scene.overlays[name]]
             for name in atlas_scene.order}
    buttons = viz3d._overlay_buttons(atlas_scene, specs, [0])
    assert len(buttons) == len(atlas_scene.order)
    for button in buttons:
        restyle, layout, traces = button['args']
        assert traces == [0]
        assert 'intensity' in restyle
        assert len(restyle['intensity'][0]) == 2 * atlas_scene.meshes[0].n_faces


# ======================================================================
# Command line
# ======================================================================
def test_cli_lists_overlays(capsys):
    assert viz3d.main(['--list-overlays']) == 0
    printed = capsys.readouterr().out
    for name in viz3d.OVERLAY_BUILDERS:
        assert name in printed


def test_cli_rejects_an_unknown_overlay(capsys):
    assert viz3d.main(['--synthetic', '--overlay', 'nonsense']) == 2
    assert 'nonsense' in capsys.readouterr().err


def test_cli_writes_a_figure(tmp_path, capsys):
    out = tmp_path / 'lgn.png'
    code = viz3d.main(['--synthetic', '--backend', 'matplotlib',
                       '--stride', '6', '--overlay', 'eccentricity',
                       '--electrodes', 'layout', '--n-electrodes', '4',
                       '--out', str(out)])
    assert code == 0
    assert out.exists()
    assert 'SYNTHETIC' in capsys.readouterr().out


def test_cli_clip_parsing():
    assert viz3d._parse_clip('ap:6:9') == ('ap', 6.0, 9.0)
    with pytest.raises(Exception, match='AXIS:LOW:HIGH'):
        viz3d._parse_clip('ap:6')
    with pytest.raises(Exception, match='ml, dv, ap'):
        viz3d._parse_clip('xx:6:9')


def test_cli_overrides_reach_the_config(tmp_path):
    options = viz3d.build_parser().parse_args(
        ['--stride', '7', '--surface-opacity', '0.3', '--overlay', 'layer',
         '--n-electrodes', '11', '--hemispheres', 'left', 'right'])
    import copy
    from dynaphos.utils import load_params
    from pathlib import Path
    params = load_params(str(Path(viz3d.__file__).resolve().parents[1]
                             / 'config' / 'params_lgn.yaml'))
    params = viz3d._apply_overrides(copy.deepcopy(params), options)
    assert params['visualization']['stride_vox'] == 7
    assert params['visualization']['surface']['opacity'] == 0.3
    assert params['visualization']['overlays'] == ['layer']
    assert params['electrodes']['n_electrodes'] == 11
    assert params['atlas']['hemispheres'] == ['left', 'right']


def test_cli_overlay_all_selects_every_overlay():
    options = viz3d.build_parser().parse_args(['--overlay', 'all'])
    params = viz3d._apply_overrides({'visualization': {'electrodes': {},
                                                       'output': {},
                                                       'surface': {}},
                                     'electrodes': {}, 'atlas': {}}, options)
    assert (params['visualization']['overlays']
            == list(viz3d.OVERLAY_BUILDERS))


@requires_real_atlas
def test_real_atlas_surface_is_a_plausible_nucleus(real_atlas, lgn_params):
    """On the published atlas the drawn surface must span the volume
    Erwin et al. document, not some fraction or multiple of it."""
    mask = viz3d.tissue_mask(real_atlas, lgn_params)
    mesh = viz3d.build_mesh(real_atlas, mask, 4, -1.0)
    extent = (mesh.vertices_mm.max(axis=0) - mesh.vertices_mm.min(axis=0))
    # A macaque LGN is a few millimetres across in every direction.
    assert np.all(extent > 3.0)
    assert np.all(extent < 12.0)
    assert mesh.n_faces > 1000
