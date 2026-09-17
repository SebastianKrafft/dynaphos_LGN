"""The mirrored right LGN, and the bilateral simulator built on it.

Two things are worth checking here and they are different in kind.

The mirror itself is exact arithmetic: a reflection, so it has an
inverse, it preserves everything orthogonal to it, and it must leave the
999 markers and the ipsilateral placeholder meaning what they meant.
Those tests compare against the source atlas rather than against
constants, so they would catch a reflection applied to the wrong axis or
an inclination convention that drifts.

The bilateral simulator is composition: the claim is that summing two
nuclei gives what one array holding all the same electrodes would, and
that nothing leaks between two state machines that should be
independent.
"""
import copy

import numpy as np
import pytest
import torch

from dynaphos_lgn.atlas import (JacobianAtlas, MirroredAtlas,
                                MirroredJacobianAtlas, mirror_inclination)
from dynaphos_lgn.bilateral import (BilateralLGNSimulator, hemifield_of,
                                    split_targets_by_hemifield)
from dynaphos_lgn.build import (build_simulator, configured_hemispheres,
                                n_electrodes_for)
from dynaphos_lgn.electrodes import LGNElectrodeArray
from dynaphos_lgn.simulator import LGNPhospheneSimulator


def visual_field_xy(atlas):
    """Cartesian visual-field position of every voxel, degrees."""
    inclination = np.deg2rad(atlas.inclination_deg)
    return (atlas.eccentricity_deg * np.cos(inclination),
            atlas.eccentricity_deg * np.sin(inclination))


@pytest.fixture(scope='module')
def mirrored_atlas(synthetic_atlas):
    return MirroredAtlas(synthetic_atlas)


@pytest.fixture(scope='module')
def reverse_ml(synthetic_atlas):
    """Index tuple reversing the ML axis, for comparing the two atlases."""
    return (slice(None, None, -1), slice(None), slice(None))


# ======================================================================
# The reflection
# ======================================================================
class TestMirrorInclination:
    @pytest.mark.parametrize('inclination, expected',
                             [(0.0, 180.0), (90.0, 90.0), (-90.0, -90.0),
                              (45.0, 135.0), (135.0, 45.0), (-135.0, -45.0),
                              (180.0, 0.0)])
    def test_known_angles(self, inclination, expected):
        assert float(mirror_inclination(inclination)) == pytest.approx(
            expected)

    def test_is_its_own_inverse(self):
        """Over (-180, 180], the range the atlas stores. The one
        alias, -180, comes back as +180: the same direction."""
        angles = np.arange(-179, 181, dtype=np.int16)
        there_and_back = mirror_inclination(mirror_inclination(angles))
        assert np.array_equal(there_and_back, angles)
        assert int(mirror_inclination(mirror_inclination(-180))) == 180

    def test_negates_x_and_leaves_y_alone(self):
        """The whole point of the transform, stated in the coordinates
        the simulator actually renders in."""
        angles = np.linspace(-180, 180, 73)
        radians, mirrored = (np.deg2rad(angles),
                             np.deg2rad(mirror_inclination(angles)))
        assert np.allclose(np.cos(mirrored), -np.cos(radians), atol=1e-12)
        assert np.allclose(np.sin(mirrored), np.sin(radians), atol=1e-12)

    def test_passes_the_sentinel_through_untouched(self):
        """999 is "not applicable", not an angle. Reflecting it would
        turn a placeholder voxel into a valid-looking one."""
        values = np.array([999, 30, 999, -60], dtype=np.int16)
        mirrored = mirror_inclination(values, sentinel=999)
        assert list(mirrored) == [999, 150, 999, -120]

    def test_respects_the_stored_scale(self):
        """Stored integers are degrees x scale, so the wrap has to
        happen in those units too."""
        assert float(mirror_inclination(450.0, scale=10.0)) == pytest.approx(
            1350.0)


class TestMirroredAtlas:
    def test_is_the_right_nucleus(self, mirrored_atlas, synthetic_atlas):
        assert synthetic_atlas.hemisphere == 'left'
        assert mirrored_atlas.hemisphere == 'right'
        assert hemifield_of('left') == 'right'
        assert hemifield_of('right') == 'left'

    def test_reflects_the_visual_field(self, mirrored_atlas, synthetic_atlas,
                                       reverse_ml):
        x, y = visual_field_xy(synthetic_atlas)
        mirrored_x, mirrored_y = visual_field_xy(mirrored_atlas)
        valid = synthetic_atlas.valid[reverse_ml]
        assert np.allclose(mirrored_x[valid], -x[reverse_ml][valid],
                           atol=1e-4)
        assert np.allclose(mirrored_y[valid], y[reverse_ml][valid], atol=1e-4)

    def test_covers_the_opposite_hemifield(self, mirrored_atlas,
                                           synthetic_atlas):
        """The reason the class exists: between them the two nuclei
        reach both halves of the field."""
        x, _ = visual_field_xy(synthetic_atlas)
        mirrored_x, _ = visual_field_xy(mirrored_atlas)
        assert x[synthetic_atlas.valid].max() > 0
        assert mirrored_x[mirrored_atlas.valid].min() < 0
        assert mirrored_x[mirrored_atlas.valid].max() <= 0

    def test_preserves_everything_a_reflection_should(self, mirrored_atlas,
                                                      synthetic_atlas,
                                                      reverse_ml):
        assert np.array_equal(mirrored_atlas.valid,
                              synthetic_atlas.valid[reverse_ml])
        assert np.array_equal(mirrored_atlas.eccentricity_deg,
                              synthetic_atlas.eccentricity_deg[reverse_ml])
        assert np.array_equal(mirrored_atlas.layer,
                              synthetic_atlas.layer[reverse_ml])
        assert (mirrored_atlas.cells_per_voxel[mirrored_atlas.valid].sum()
                == pytest.approx(synthetic_atlas.cells_per_voxel[
                                     synthetic_atlas.valid].sum()))

    def test_shares_the_source_arrays_rather_than_copying(
            self, mirrored_atlas, synthetic_atlas):
        """Eccentricity, layer and cells are reflections of the source
        atlas's own memory. A copy would double the atlas's footprint
        for no new information."""
        for attribute in ('eccentricity', 'layer', 'cells',
                          'eccentricity_deg', 'cells_per_voxel'):
            assert (getattr(mirrored_atlas, attribute).base
                    is getattr(synthetic_atlas, attribute))

    def test_ipsi_placeholder_moves_with_the_field(self, mirrored_atlas):
        """+-135 deg marks tissue coarsely representing the OTHER
        hemifield. Reflected, that is +-45 deg -- still the other
        hemifield, still excluded."""
        assert sorted(mirrored_atlas.ipsi_sentinel_values) == [-45.0, 45.0]

    def test_inherits_whether_the_source_was_synthetic(self, mirrored_atlas):
        assert mirrored_atlas.is_synthetic is True

    def test_refuses_to_mirror_a_mirror(self, mirrored_atlas):
        with pytest.raises(ValueError, match="Mirroring a mirror"):
            MirroredAtlas(mirrored_atlas)

    def test_lateral_distance_picks_out_mirror_image_tissue(
            self, mirrored_atlas, synthetic_atlas, lgn_params):
        """`ml_mm` is distance from the midline in both nuclei, so the
        same coordinate must land on reflected voxels, not the same
        index."""
        n_ml = synthetic_atlas.atlas_shape[0]
        coordinates = (9.0, -1.0, 4.0)
        left = synthetic_atlas.horsley_clarke_to_index(*coordinates)
        right = mirrored_atlas.horsley_clarke_to_index(*coordinates)
        assert right[0] == n_ml - 1 - left[0]
        assert right[1:] == left[1:]

    def test_index_to_mm_inverts_the_lookup_in_both_nuclei(
            self, mirrored_atlas, synthetic_atlas):
        """Multiplying an index by the voxel size and adding the origin
        is the obvious inverse and it is wrong for the right nucleus --
        it mirrors it back onto the left one. The atlas's own inverse
        has to round-trip in both."""
        voxels = np.argwhere(synthetic_atlas.valid)[::1701]
        for atlas in (synthetic_atlas, mirrored_atlas):
            mm = atlas.index_to_horsley_clarke(*voxels.T)
            back = np.stack(atlas.horsley_clarke_to_index(*mm), axis=-1)
            assert np.array_equal(back, voxels)

    def test_index_to_mm_puts_the_two_nuclei_in_different_places(
            self, mirrored_atlas, synthetic_atlas):
        """The naive inverse would give both nuclei identical
        coordinates, which is exactly the bug this method exists to stop.
        """
        voxel = np.argwhere(synthetic_atlas.valid)[0]
        left = synthetic_atlas.index_to_horsley_clarke(*voxel)
        right = mirrored_atlas.index_to_horsley_clarke(*voxel)
        assert left[0] != right[0]
        assert left[1:] == right[1:]

    def test_a_coordinate_meant_for_the_other_nucleus_fails_loudly(
            self, mirrored_atlas, synthetic_atlas):
        with pytest.raises(ValueError, match='right LGN'):
            mirrored_atlas.horsley_clarke_to_index(9.0, -1.0, 4.0,
                                                   hemisphere='left')
        with pytest.raises(ValueError, match='left LGN'):
            synthetic_atlas.horsley_clarke_to_index(9.0, -1.0, 4.0,
                                                    hemisphere='right')
        # Naming the nucleus you are actually addressing is fine.
        synthetic_atlas.horsley_clarke_to_index(9.0, -1.0, 4.0,
                                                hemisphere='left')


class TestMirroredJacobianAtlas:
    @pytest.fixture(scope='class')
    def mirrored_jacobian(self, synthetic_jacobian, lgn_params):
        return MirroredJacobianAtlas(synthetic_jacobian, lgn_params)

    @pytest.fixture(scope='class')
    def voxel_pairs(self, mirrored_jacobian, synthetic_atlas):
        """Matching (mirrored, source) index arrays over fit-ok voxels."""
        mirrored = np.argwhere(mirrored_jacobian.jacobian_valid)[::97]
        source = mirrored.copy()
        source[:, 0] = synthetic_atlas.atlas_shape[0] - 1 - source[:, 0]
        return mirrored, source

    def test_sign_pattern(self, mirrored_jacobian, synthetic_jacobian,
                          voxel_pairs):
        """x and ML are the reflected axes, so exactly the terms
        coupling one of them to an unreflected axis change sign."""
        mirrored, source = voxel_pairs
        expected = np.array([[1, -1, -1], [-1, 1, 1]], dtype=float)
        assert np.allclose(mirrored_jacobian.jacobian_at(mirrored),
                           expected * synthetic_jacobian.jacobian_at(source),
                           equal_nan=True)

    def test_phosphene_sizes_are_untouched(self, mirrored_jacobian,
                                           synthetic_jacobian, voxel_pairs):
        """A reflection is orthogonal on both sides, so it cannot change
        either principal magnification -- and so cannot change any
        phosphene's size."""
        from dynaphos_lgn.magnification import JacobianMagnification
        mirrored, source = voxel_pairs
        got = JacobianMagnification.decompose(
            mirrored_jacobian.jacobian_at(mirrored))
        want = JacobianMagnification.decompose(
            synthetic_jacobian.jacobian_at(source))
        assert np.allclose(got[0], want[0], equal_nan=True)
        assert np.allclose(got[1], want[1], equal_nan=True)

    def test_major_axis_is_reflected(self, mirrored_jacobian,
                                     synthetic_jacobian, voxel_pairs):
        """theta -> pi - theta: the same axis, seen in a mirror.

        An ellipse's major axis is a direction only up to pi -- the
        SVD is free to return either singular vector -- so the
        comparison is made in the doubled angle, where that freedom
        cancels and the reflection shows up as a sign change on sin
        alone.
        """
        from dynaphos_lgn.magnification import JacobianMagnification
        mirrored, source = voxel_pairs
        got = JacobianMagnification.decompose(
            mirrored_jacobian.jacobian_at(mirrored))[2]
        want = JacobianMagnification.decompose(
            synthetic_jacobian.jacobian_at(source))[2]
        finite = np.isfinite(got) & np.isfinite(want)
        assert np.allclose(np.cos(2 * got[finite]),
                           np.cos(2 * want[finite]), atol=1e-6)
        assert np.allclose(np.sin(2 * got[finite]),
                           -np.sin(2 * want[finite]), atol=1e-6)

    def test_flags_travel_with_their_voxels(self, mirrored_jacobian,
                                            synthetic_jacobian, reverse_ml):
        assert np.array_equal(mirrored_jacobian.jacobian_valid,
                              synthetic_jacobian.jacobian_valid[reverse_ml])
        assert np.array_equal(mirrored_jacobian.isotropic_flag,
                              synthetic_jacobian.isotropic_flag[reverse_ml])
        assert np.array_equal(
            mirrored_jacobian.unreliable_beyond_neighborhood_flag,
            synthetic_jacobian.unreliable_beyond_neighborhood_flag[reverse_ml])

    def test_decompose_valid_avoids_materialising_the_field(
            self, mirrored_jacobian, voxel_pairs):
        """The whole-grid decomposition is derived from the source
        atlas's own SVD; asking for it must not build a reflected copy
        of the Jacobian."""
        from dynaphos_lgn.magnification import JacobianMagnification
        major, minor, _ = mirrored_jacobian.decompose_valid()
        assert mirrored_jacobian._materialised is None
        mirrored, _ = voxel_pairs
        index = tuple(mirrored.T)
        expected = JacobianMagnification.decompose(
            mirrored_jacobian.jacobian_at(mirrored))
        assert np.allclose(major[index], expected[0], equal_nan=True)
        assert np.allclose(minor[index], expected[1], equal_nan=True)

    def test_refuses_to_be_refitted_or_cached(self, mirrored_jacobian,
                                              synthetic_atlas, tmp_path):
        with pytest.raises(NotImplementedError):
            mirrored_jacobian.compute(synthetic_atlas)
        with pytest.raises(NotImplementedError):
            mirrored_jacobian.save(tmp_path / 'mirror.npz')

    def test_refuses_to_be_desynchronised_from_its_source(
            self, mirrored_jacobian):
        with pytest.raises(AttributeError, match='reflects'):
            mirrored_jacobian.jacobian = np.zeros((2, 3))


# ======================================================================
# Electrodes in the mirrored nucleus
# ======================================================================
class TestElectrodesInTheRightNucleus:
    @pytest.fixture(scope='class')
    def right_array(self, mirrored_atlas, lgn_params, synthetic_jacobian):
        from dynaphos_lgn.build import build_kernel
        jacobian = MirroredJacobianAtlas(synthetic_jacobian, lgn_params)
        return LGNElectrodeArray.spread_in_visual_field(
            mirrored_atlas, lgn_params, rng=np.random.default_rng(0),
            kernel=build_kernel(lgn_params), jacobian_atlas=jacobian)

    def test_every_electrode_lands_in_the_left_hemifield(self, right_array):
        assert right_array.hemisphere == 'right'
        assert np.all(right_array.vf_xy[:, 0] <= 0)

    def test_candidate_footprint_stays_in_the_left_hemifield(self,
                                                             right_array):
        x = right_array.candidate_vf_xy[..., 0]
        assert np.all(x[right_array.candidate_valid] <= 1e-6)

    def test_describe_names_the_nucleus_and_the_hemifield(self, right_array):
        text = right_array.describe()
        assert 'right LGN -> left hemifield' in text
        assert 'mirrored from the published left atlas' in text


# ======================================================================
# Building
# ======================================================================
class TestHemisphereConfig:
    def test_orders_left_first(self):
        assert configured_hemispheres(
            {'atlas': {'hemispheres': ['right', 'left']}}) == ['left', 'right']

    def test_accepts_a_bare_string(self):
        assert configured_hemispheres(
            {'atlas': {'hemispheres': 'right'}}) == ['right']

    @pytest.mark.parametrize('value, message', [
        ([], 'at least one'),
        (['middle'], 'Unknown hemisphere'),
        (['left', 'left'], 'twice')])
    def test_rejects_nonsense(self, value, message):
        with pytest.raises(ValueError, match=message):
            configured_hemispheres({'atlas': {'hemispheres': value}})

    def test_electrode_count_defaults_to_the_shared_value(self, lgn_params):
        for hemisphere in ('left', 'right'):
            assert (n_electrodes_for(lgn_params, hemisphere)
                    == lgn_params['electrodes']['n_electrodes'])

    def test_electrode_count_can_differ_between_nuclei(self, params_override):
        params = params_override(
            electrodes__n_electrodes_by_hemisphere={'left': 6, 'right': 3})
        assert n_electrodes_for(params, 'left') == 6
        assert n_electrodes_for(params, 'right') == 3


@pytest.fixture(scope='module')
def bilateral_params(lgn_params):
    """The shared config, both nuclei, and a frame wide enough to show
    them."""
    params = copy.deepcopy(lgn_params)
    params['atlas']['hemispheres'] = ['left', 'right']
    params['electrodes']['n_electrodes_by_hemisphere'] = {'left': 6,
                                                          'right': 4}
    params['run']['view_angle'] = 40
    return params


@pytest.fixture(scope='module')
def bilateral(bilateral_params):
    return build_simulator(bilateral_params, synthetic=True)


class TestBuildDispatch:
    def test_one_hemisphere_still_gives_a_plain_simulator(self, lgn_params):
        """The default config is unchanged, and so is what it builds."""
        assert configured_hemispheres(lgn_params) == ['left']
        simulator, parts = build_simulator(lgn_params, synthetic=True)
        assert isinstance(simulator, LGNPhospheneSimulator)
        assert simulator.hemisphere == 'left'
        assert set(parts) == {'atlas', 'jacobian_atlas', 'kernel',
                              'magnification', 'electrode_array'}

    def test_right_alone_is_buildable(self, params_override):
        params = params_override(atlas__hemispheres=['right'])
        simulator, parts = build_simulator(params, synthetic=True)
        assert isinstance(simulator, LGNPhospheneSimulator)
        assert simulator.hemisphere == 'right'
        assert np.all(simulator.array.vf_xy[:, 0] <= 0)

    def test_both_gives_a_bilateral_simulator(self, bilateral):
        simulator, parts = bilateral
        assert isinstance(simulator, BilateralLGNSimulator)
        assert simulator.hemispheres == ['left', 'right']
        assert set(parts['hemispheres']) == {'left', 'right'}

    def test_the_hemisphere_argument_overrides_the_config(self, lgn_params):
        simulator, _ = build_simulator(lgn_params, synthetic=True,
                                       hemispheres=['left', 'right'])
        assert isinstance(simulator, BilateralLGNSimulator)

    def test_both_nuclei_share_one_atlas_load_and_one_fit(self, bilateral):
        """Reflecting is re-indexing. Loading the atlas twice, or
        refitting the Jacobian, would be minutes and gigabytes spent on
        numbers that follow from the ones already in hand."""
        _, parts = bilateral
        left = parts['hemispheres']['left']
        right = parts['hemispheres']['right']
        assert right['atlas'].base is left['atlas']
        assert right['jacobian_atlas'].base is left['jacobian_atlas']
        assert right['kernel'] is left['kernel']

    def test_per_hemisphere_electrode_counts_are_honoured(self, bilateral):
        simulator, _ = bilateral
        assert simulator.counts == [6, 4]
        assert simulator.num_phosphenes == 10

    def test_the_two_arrays_are_not_reflections_of_each_other(
            self, params_override):
        """Same count both sides: the placements should still differ,
        because a perfectly mirror-symmetric implant is an assumption
        nobody asked for."""
        params = params_override(atlas__hemispheres=['left', 'right'])
        simulator, _ = build_simulator(params, synthetic=True)
        left = simulator.vf_xy[simulator.electrode_slices['left']]
        right = simulator.vf_xy[simulator.electrode_slices['right']]
        assert not np.allclose(left * [-1, 1], right)

    def test_rebuilding_from_the_same_seed_reproduces_the_arrays(
            self, bilateral_params):
        first, _ = build_simulator(bilateral_params, synthetic=True)
        second, _ = build_simulator(bilateral_params, synthetic=True)
        assert np.allclose(first.vf_xy, second.vf_xy)

    def test_voxel_indices_for_two_nuclei_must_say_which_is_which(
            self, bilateral_params):
        with pytest.raises(ValueError, match='ambiguous'):
            build_simulator(bilateral_params, synthetic=True,
                            voxel_indices=np.array([[10, 10, 10]]))

    def test_voxel_indices_can_be_given_per_nucleus(self, bilateral_params,
                                                    synthetic_atlas):
        voxels = np.argwhere(synthetic_atlas.valid)[::3000][:3]
        simulator, _ = build_simulator(
            bilateral_params, synthetic=True,
            voxel_indices={'left': voxels, 'right': voxels[:2]})
        assert simulator.counts == [3, 2]

    def test_visual_field_targets_are_routed_by_hemifield(
            self, bilateral_params):
        simulator, _ = build_simulator(
            bilateral_params, synthetic=True,
            visual_field_targets=([3.0, 4.0, 5.0], [10.0, 170.0, -175.0]))
        assert simulator.counts == [1, 2]
        assert np.all(simulator.vf_xy[simulator.electrode_slices['right'],
                                      0] <= 0)


class TestTargetRouting:
    def test_splits_on_the_sign_of_x(self):
        split = split_targets_by_hemifield([1.0, 2.0], [0.0, 180.0],
                                           ['left', 'right'])
        assert split['left'][0] == pytest.approx([1.0])
        assert split['right'][0] == pytest.approx([2.0])

    def test_vertical_meridian_goes_to_the_left_nucleus(self):
        """Both nuclei represent that strip; the tie has to break
        somewhere, and it breaks the same way every time."""
        split = split_targets_by_hemifield([5.0, 5.0], [90.0, 180.0],
                                           ['left', 'right'])
        assert split['left'][0] == pytest.approx([5.0])

    def test_a_nucleus_with_nothing_to_do_is_an_error_not_an_empty_array(
            self):
        with pytest.raises(ValueError, match='left hemifield'):
            split_targets_by_hemifield([3.0], [0.0], ['left', 'right'])


# ======================================================================
# The bilateral forward pass
# ======================================================================
class TestBilateralForwardPass:
    @staticmethod
    def _amplitude(simulator, value=80e-6):
        return torch.full((simulator.num_phosphenes,), value)

    def test_renders_one_frame_of_the_whole_field(self, bilateral,
                                                  bilateral_params):
        simulator, _ = bilateral
        simulator.reset()
        image = simulator(self._amplitude(simulator))
        res_x, res_y = bilateral_params['run']['resolution']
        assert image.shape == (res_y, res_x)
        assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0

    def test_lights_up_both_halves(self, bilateral):
        """The point of the exercise: a one-nucleus simulator can only
        ever fill half the frame."""
        simulator, _ = bilateral
        simulator.reset()
        image = simulator(self._amplitude(simulator))
        half = image.shape[-1] // 2
        assert float(image[..., :half].sum()) > 0
        assert float(image[..., half:].sum()) > 0

    def test_electrodes_are_numbered_left_nucleus_first(self, bilateral):
        simulator, _ = bilateral
        assert list(simulator.hemisphere_of_electrode) == ['left'] * 6 + [
            'right'] * 4
        assert np.all(simulator.vf_xy[simulator.electrode_slices['left'],
                                      0] >= 0)
        assert np.all(simulator.vf_xy[simulator.electrode_slices['right'],
                                      0] <= 0)

    def test_stimulating_one_nucleus_leaves_the_other_silent(self,
                                                             bilateral):
        """Two nuclei, two independent state machines: current in one
        must not charge, light or recruit anything in the other.

        Stated over the right nucleus's own state rather than over the
        left half of the frame, because a phosphene sitting on the
        vertical meridian spreads across it -- that is the retinotopy
        doing its job, not one nucleus leaking into the other.
        """
        simulator, _ = bilateral
        simulator.reset()
        amplitude = torch.zeros(simulator.num_phosphenes)
        amplitude[simulator.electrode_slices['left']] = 80e-6
        image = simulator(amplitude)

        right = simulator.simulators['right']
        assert float(right.activation.get().abs().max()) == 0.0
        assert float(right.trace.get().abs().max()) == 0.0
        assert float(right.layer_activation.abs().max()) == 0.0
        assert float(right.detection_probability().abs().max()) == 0.0
        state = simulator.get_state()
        assert float(state['activation'][
                         simulator.electrode_slices['right']].abs().max()
                     ) == 0.0

        # And so the percept is exactly what the left nucleus alone
        # renders -- meridian spill included.
        left = simulator.simulators['left']
        alone = torch.sum(
            left.brightness.get() * left.detection_probability()
            * left.spatial_activation(), dim=left.electrode_dimension)
        assert torch.allclose(image, alone.clamp(0, 1))
        assert float(image.sum()) > 0

    def test_agrees_with_summing_the_two_nuclei_by_hand(self, bilateral):
        """The composition claim, stated exactly: the clamp is applied
        once over both nuclei, not once per nucleus."""
        simulator, _ = bilateral
        simulator.reset()
        image = simulator(self._amplitude(simulator))

        total = None
        for sub in simulator.simulators.values():
            intensity = sub.brightness.get() * sub.detection_probability()
            summed = torch.sum(intensity * sub.spatial_activation(),
                               dim=sub.electrode_dimension)
            total = summed if total is None else total + summed
        assert torch.allclose(image, total.clamp(0, 1))

    def test_zero_current_gives_an_empty_percept(self, bilateral):
        simulator, _ = bilateral
        simulator.reset()
        image = simulator(torch.zeros(simulator.num_phosphenes))
        assert float(image.max()) == 0.0

    def test_gradients_reach_both_nuclei(self, bilateral):
        simulator, _ = bilateral
        simulator.reset()
        amplitude = self._amplitude(simulator).requires_grad_(True)
        simulator(amplitude).sum().backward()
        assert amplitude.grad is not None
        assert torch.all(torch.isfinite(amplitude.grad))
        for name, span in simulator.electrode_slices.items():
            assert float(amplitude.grad[span].abs().sum()) > 0, name

    def test_reset_clears_both_nuclei(self, bilateral):
        simulator, _ = bilateral
        simulator(self._amplitude(simulator))
        simulator.reset()
        for sub in simulator.simulators.values():
            assert float(sub.activation.get().abs().max()) == 0.0

    def test_state_is_reported_over_all_electrodes(self, bilateral):
        simulator, _ = bilateral
        simulator.reset()
        simulator(self._amplitude(simulator))
        state = simulator.get_state()
        assert state['brightness'].shape[0] == simulator.num_phosphenes
        table = simulator.layer_activation_table()
        for values in table.values():
            assert values.shape[0] == simulator.num_phosphenes

    def test_sampling_an_image_feeds_both_nuclei(self, bilateral,
                                                 bilateral_params):
        simulator, _ = bilateral
        res_x, res_y = bilateral_params['run']['resolution']
        image = np.ones((res_y, res_x), dtype=np.float32)
        amplitudes = simulator.sample_stimulus(image)
        assert amplitudes.shape[-1] == simulator.num_phosphenes

    def test_visual_field_spans_both_hemifields(self, bilateral):
        """One `Map` over both nuclei, with inclination now using its
        whole range rather than just (-90, 90) deg."""
        simulator, _ = bilateral
        x, y = simulator.visual_field.cartesian
        assert len(x) == simulator.num_phosphenes
        assert np.allclose(np.stack([x, y], axis=-1), simulator.vf_xy,
                           atol=1e-6)
        assert x.min() < 0 < x.max()

    def test_phosphene_sizes_are_reported_over_both_nuclei(self, bilateral):
        simulator, _ = bilateral
        major, minor = simulator.phosphene_sigma_deg(100.0)
        assert major.shape == (simulator.num_phosphenes,)
        assert np.all(major[np.isfinite(major)]
                      >= minor[np.isfinite(major)] - 1e-9)
        for name, sub in simulator.simulators.items():
            span = simulator.electrode_slices[name]
            assert np.allclose(major[span],
                               sub.array.phosphene_sigma_deg(100.0)[0],
                               equal_nan=True)

    def test_shape_broadcasts_against_the_reported_state(self, bilateral):
        """`shape` exists so an optimisation target can be built against
        it, which only helps if it matches what get_state returns."""
        simulator, _ = bilateral
        simulator.reset()
        simulator(self._amplitude(simulator))
        target = torch.full(simulator.shape, 0.5)
        assert target.shape == simulator.get_state()['brightness'].shape

    def test_describe_says_what_the_right_half_is(self, bilateral):
        text = bilateral[0].describe()
        assert '[left LGN]' in text and '[right LGN]' in text
        assert 'reflected' in text

    def test_a_wrongly_sized_amplitude_says_so(self, bilateral):
        simulator, _ = bilateral
        with pytest.raises(ValueError, match='all 10 electrodes'):
            simulator(torch.zeros(simulator.num_phosphenes + 1))

    def test_a_scalar_amplitude_reaches_every_electrode(self, bilateral):
        simulator, _ = bilateral
        simulator.reset()
        image = simulator(torch.tensor(80e-6))
        half = image.shape[-1] // 2
        assert float(image[..., :half].sum()) > 0
        assert float(image[..., half:].sum()) > 0
