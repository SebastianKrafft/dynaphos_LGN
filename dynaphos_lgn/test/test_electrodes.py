"""Setup-time electrode geometry: candidate gather, layer grouping, flags."""
import copy

import numpy as np
import pytest

from dynaphos_lgn.current_spread import RecruitmentKernel
from dynaphos_lgn.electrodes import LGNElectrodeArray, layer_to_class


@pytest.fixture(scope='module')
def array_params(lgn_params):
    """The shared config with this module's electrode knobs set.

    The array reads every one of them from the config now, so a test
    that wants a different current ceiling or field extent says so here
    rather than passing it in. Module-scoped, hence the deep copy: the
    `lgn_params` fixture is shared with every other test file.
    """
    params = copy.deepcopy(lgn_params)
    params['electrodes']['max_current_ua'] = 80.0
    params['electrodes']['max_eccentricity_deg'] = 12.0
    return params


@pytest.fixture
def target_params(params_override):
    """Config for the placed-target tests: a lower current ceiling and a
    small scatter subsample, both only to keep them cheap."""
    return params_override(electrodes__max_current_ua=60.0,
                           electrodes__n_scatter_points=64)


@pytest.fixture(scope='module')
def array(synthetic_atlas, synthetic_jacobian, array_params):
    return LGNElectrodeArray.spread_in_visual_field(
        synthetic_atlas, array_params,
        kernel=RecruitmentKernel.from_params(array_params),
        jacobian_atlas=synthetic_jacobian, rng=np.random.default_rng(1))


class TestCandidateNeighborhood:
    def test_every_candidate_lies_inside_the_gather_radius(self, array):
        assert np.all(array.candidate_distance_mm <= array.max_radius_mm
                      + 1e-12)

    def test_gather_radius_exceeds_the_nominal_spread_radius(self, array):
        """A smooth kernel still has weight past its nominal edge, so
        gathering only to R(I_max) would clip the tail the smoothness
        exists to provide."""
        assert array.max_radius_mm > array.nominal_radius_mm

    def test_stride_keeps_the_candidate_set_within_budget(self, array,
                                                          array_params):
        budget = array_params['electrodes']['max_candidate_voxels']
        assert array.n_candidates <= budget
        assert array.voxels_per_candidate == array.candidate_stride_vox ** 3

    def test_offsets_are_shared_across_electrodes(self, array):
        """The neighbourhood shape depends only on the radius, so it can
        be computed once instead of per electrode."""
        assert array.candidate_offsets_mm.shape == (array.n_candidates, 3)
        assert array.candidate_voxels.shape == (array.n_electrodes,
                                                array.n_candidates, 3)

    def test_candidate_indices_stay_inside_the_volume(self, array,
                                                      synthetic_atlas):
        shape = np.asarray(synthetic_atlas.atlas_shape)
        assert np.all(array.candidate_voxels >= 0)
        assert np.all(array.candidate_voxels < shape)

    def test_invalid_candidates_are_masked_out_of_the_layer_map(self, array):
        assert np.all(array.candidate_layer[~array.candidate_valid] == 0)


class TestLayerGrouping:
    def test_one_hot_sums_to_one_on_tissue_and_zero_off_it(self, array):
        totals = array.candidate_layer_onehot.sum(axis=-1)
        assert np.all(totals[array.candidate_valid] == 1)
        assert np.all(totals[~array.candidate_valid] == 0)

    def test_class_labels_agree_with_layer_codes(self, array, lgn_params):
        for code, cls in layer_to_class(lgn_params).items():
            sel = array.candidate_layer == code
            assert np.all(array.candidate_class[sel]
                          == array.class_codes[cls])

    def test_histogram_counts_match_a_direct_count(self, array):
        """The histogram is a performance shortcut, so it has to be
        exactly equivalent to the thing it replaces."""
        hist = array.distance_histograms()
        for li, code in enumerate(array.layer_ids):
            direct = ((array.candidate_layer == code)
                      & array.candidate_valid).sum(axis=1)
            binned = hist['counts'][:, li].sum(axis=1)
            assert np.allclose(binned,
                               direct * array.voxels_per_candidate)

    def test_histogram_cells_match_a_direct_sum(self, array):
        hist = array.distance_histograms()
        for li, code in enumerate(array.layer_ids):
            mask = (array.candidate_layer == code) & array.candidate_valid
            direct = (array.candidate_cells_per_voxel * mask).sum(axis=1)
            assert np.allclose(hist['cells'][:, li].sum(axis=1),
                               direct * array.voxels_per_candidate,
                               rtol=1e-5)


class TestPositionsAndFlags:
    def test_visual_field_positions_are_consistent(self, array):
        r, phi = array.visual_field.polar
        assert np.allclose(r, array.eccentricity_deg, equal_nan=True)
        assert np.allclose(np.rad2deg(phi), array.inclination_deg,
                           equal_nan=True)
        assert np.allclose(array.vf_xy[:, 0], r * np.cos(phi),
                           equal_nan=True)

    def test_central_electrodes_are_flagged_isotropic(self, synthetic_atlas,
                                                      synthetic_jacobian,
                                                      target_params):
        arr = LGNElectrodeArray.from_visual_field_targets(
            synthetic_atlas, target_params, [0.2], [0.0],
            kernel=RecruitmentKernel.from_params(target_params),
            jacobian_atlas=synthetic_jacobian)
        assert arr.flags.central_isotropic[0]

    def test_central_electrodes_render_as_ellipses_not_scatter(
            self, synthetic_atlas, synthetic_jacobian, target_params):
        """Deliberate exception: inside the central 1 deg every voxel's
        position already traces back to Malpeli's isotropic formula, so
        per-voxel scatter would cost more for the same circle."""
        arr = LGNElectrodeArray.from_visual_field_targets(
            synthetic_atlas, target_params, [0.2], [0.0],
            kernel=RecruitmentKernel.from_params(target_params),
            jacobian_atlas=synthetic_jacobian)
        assert arr.render_mode[0] == 'ellipse'

    def test_visual_field_targets_land_close_to_the_request(
            self, synthetic_atlas, synthetic_jacobian, target_params):
        ecc = [3.0, 8.0, 15.0]
        arr = LGNElectrodeArray.from_visual_field_targets(
            synthetic_atlas, target_params, ecc, [10.0, -20.0, 30.0],
            kernel=RecruitmentKernel.from_params(target_params),
            jacobian_atlas=synthetic_jacobian)
        assert np.all(arr.position_error_deg < 1.0)

    def test_flags_summary_counts_every_electrode(self, array):
        text = array.flags.summary()
        assert f"/ {array.n_electrodes}" in text
        assert len(array.flags.any_flag()) == array.n_electrodes

    def test_describe_mentions_the_stride_and_the_radius(self, array):
        text = array.describe()
        assert 'stride' in text and 'spread radius' in text


class TestSizeAndScatter:
    def test_phosphene_size_grows_with_current(self, array):
        small, _ = array.phosphene_sigma_deg(10.0)
        large, _ = array.phosphene_sigma_deg(80.0)
        assert np.all(large[np.isfinite(large)]
                      > small[np.isfinite(small)])

    def test_major_axis_is_never_smaller_than_the_minor(self, array):
        major, minor = array.phosphene_sigma_deg(40.0)
        ok = np.isfinite(major) & np.isfinite(minor)
        assert np.all(major[ok] >= minor[ok] - 1e-9)

    def test_scatter_points_are_a_subset_of_the_candidates(self, array):
        assert array.n_scatter_points <= array.n_candidates
        assert np.all(array.scatter_index < array.n_candidates)
        assert len(np.unique(array.scatter_index)) == array.n_scatter_points

    def test_scatter_bandwidth_floor_is_positive_and_small(self, array):
        floor = array.scatter_bandwidth_floor_deg
        radius = array.scatter_patch_radius_deg()
        ok = radius > 0
        assert np.all(floor[ok] > 0)
        assert np.all(floor[ok] < radius[ok])

    def test_recruitment_is_larger_where_more_tissue_is_reachable(self,
                                                                 array):
        counts = array.per_layer_cell_counts()
        assert counts.shape == (array.n_electrodes, len(array.layer_ids))
        assert np.all(counts >= 0)


def test_rejects_badly_shaped_input(synthetic_atlas, synthetic_jacobian,
                                    lgn_params):
    with pytest.raises(ValueError):
        LGNElectrodeArray(synthetic_atlas, lgn_params,
                          np.zeros((4, 2), dtype=int),
                          jacobian_atlas=synthetic_jacobian)


def test_requires_some_magnification_source(synthetic_atlas, lgn_params):
    with pytest.raises(ValueError):
        LGNElectrodeArray(synthetic_atlas, lgn_params,
                          np.array([[20, 20, 30]]))
