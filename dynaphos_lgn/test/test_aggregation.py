"""Multi-electrode aggregation: field summation and independence."""
import numpy as np
import pytest
import torch

from dynaphos_lgn.build import build_electrode_array
from dynaphos_lgn.current_spread import RecruitmentKernel
from dynaphos_lgn.simulator import LGNPhospheneSimulator

SPACING_VOX = 8          # 0.2 mm at 25 um voxels, under one spread radius
CURRENT_A = 60e-6


def _line(atlas, n=2):
    """n valid voxels SPACING_VOX apart along ML, at 2-4 deg so all
    render well inside the field of view."""
    usable = atlas.valid & (atlas.eccentricity_deg > 2) & (
        atlas.eccentricity_deg < 4)
    every = usable.copy()
    for k in range(1, n):
        shifted = np.zeros_like(usable)
        shifted[:-k * SPACING_VOX] = usable[k * SPACING_VOX:]
        every &= shifted
    candidates = np.argwhere(every)
    first = candidates[len(candidates) // 2]
    return first + np.arange(n)[:, None] * [SPACING_VOX, 0, 0]


def _simulator(params, atlas, jacobian, voxels):
    kernel = RecruitmentKernel.from_params(params)
    array = build_electrode_array(
        params, atlas, jacobian, kernel, np.random.default_rng(0),
        voxel_indices=voxels)
    return LGNPhospheneSimulator(params, array,
                                 rng=np.random.default_rng(0))


def _percept(sim, amplitudes):
    sim.reset()
    sim.update(torch.tensor(amplitudes, dtype=torch.float32))
    return sim.unclamped_percept()


@pytest.fixture
def pair_simulator(params_override, synthetic_atlas, synthetic_jacobian):
    def make(**overrides):
        params = params_override(**overrides)
        return _simulator(params, synthetic_atlas, synthetic_jacobian,
                          _line(synthetic_atlas))
    return make


def _superposition_gap(sim):
    both = _percept(sim, [CURRENT_A, CURRENT_A])
    alone = (_percept(sim, [CURRENT_A, 0.0])
             + _percept(sim, [0.0, CURRENT_A]))
    return both, alone


class TestAdditivity:
    @pytest.mark.parametrize('merging', ['merged', 'grown'])
    def test_field_superposition_is_not_the_sum_of_single_electrodes(
            self, pair_simulator, merging):
        sim = pair_simulator(aggregation__mode='field_superposition',
                             aggregation__phosphene_merging=merging)
        both, alone = _superposition_gap(sim)
        assert not torch.allclose(both, alone, rtol=1e-3, atol=1e-4)

    def test_independent_is_the_sum_of_single_electrodes(self,
                                                         pair_simulator):
        sim = pair_simulator(aggregation__mode='independent')
        both, alone = _superposition_gap(sim)
        assert float(both.max()) > 0.0
        assert torch.allclose(both, alone, rtol=1e-5, atol=1e-6)

    def test_interleaved_pulses_fall_back_to_independent(self,
                                                         pair_simulator):
        sim = pair_simulator(aggregation__mode='field_superposition',
                             aggregation__simultaneous_pulses=False)
        both, alone = _superposition_gap(sim)
        assert torch.allclose(both, alone, rtol=1e-5, atol=1e-6)


MERGING = ['merged', 'grown']


def _field(pair_simulator, merging='grown'):
    return pair_simulator(aggregation__mode='field_superposition',
                          aggregation__phosphene_merging=merging)


class TestFieldSuperposition:
    @pytest.mark.parametrize('merging', MERGING)
    def test_one_electrode_alone_matches_independent(self, pair_simulator,
                                                     merging):
        reference = _percept(pair_simulator(aggregation__mode='independent'),
                             [CURRENT_A, 0.0])
        field = _percept(_field(pair_simulator, merging),
                         [CURRENT_A, 0.0])
        assert torch.allclose(field, reference, rtol=1e-4, atol=1e-5)

    def test_neighbors_raise_the_equivalent_current(self, pair_simulator):
        sim = _field(pair_simulator)
        _percept(sim, [CURRENT_A, CURRENT_A])
        equivalent = sim.get_state()['equivalent_current_ua'].flatten()
        assert torch.all(equivalent > CURRENT_A * 1e6 * 1.05)

    def test_a_dense_cluster_never_lowers_the_equivalent_current(
            self, params_override, synthetic_atlas, synthetic_jacobian):
        """Saturating overlap must not make electrodes weaker than
        they are alone."""
        center = _line(synthetic_atlas)[0]
        offsets = np.array([[0, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0],
                            [0, -2, 0], [0, 0, 2], [0, 0, -2]])
        params = params_override(aggregation__mode='field_superposition')
        sim = _simulator(params, synthetic_atlas, synthetic_jacobian,
                         center + offsets)
        _percept(sim, [CURRENT_A] * len(offsets))
        equivalent = sim.get_state()['equivalent_current_ua'].flatten()
        assert torch.all(equivalent >= CURRENT_A * 1e6 * (1 - 1e-5))

    def test_the_equivalent_current_is_capped_at_the_gathered_one(
            self, pair_simulator, caplog):
        """The neighborhood is gathered for max_current_ua only; warn
        once, not every frame."""
        sim = pair_simulator(aggregation__mode='field_superposition',
                             electrodes__max_current_ua=CURRENT_A * 1e6,
                             default_stim__safety_limit_ua=CURRENT_A * 1e6)
        for _ in range(2):
            _percept(sim, [CURRENT_A, CURRENT_A])
        equivalent = sim.get_state()['equivalent_current_ua']
        assert torch.allclose(equivalent,
                              torch.full_like(equivalent, CURRENT_A * 1e6))
        assert sum('max_current_ua' in r.message
                   for r in caplog.records) == 1

    @pytest.mark.parametrize('merging', MERGING)
    def test_gradients_reach_the_currents(self, pair_simulator, merging):
        """Below max_current_ua; at the cap the equivalent current is
        flat in the currents by design."""
        sim = _field(pair_simulator, merging)
        amplitude = torch.full((2,), CURRENT_A / 2, requires_grad=True)
        sim.reset()
        sim.update(amplitude)
        sim.unclamped_percept().sum().backward()
        equivalent = sim.get_state()['equivalent_current_ua']
        assert torch.all(equivalent < sim.params['electrodes'][
            'max_current_ua'] * 0.99)
        assert torch.all(torch.isfinite(amplitude.grad))
        assert torch.all(amplitude.grad != 0)


@pytest.mark.parametrize('overrides', [
    dict(aggregation__mode='independent'),
    dict(aggregation__mode='field_superposition')])
def test_a_silent_electrode_keeps_gradients_finite(pair_simulator,
                                                   overrides):
    """sqrt(I) has an infinite slope at 0; a ReLU encoder hits it."""
    sim = pair_simulator(**overrides)
    amplitude = torch.tensor([CURRENT_A, 0.0], requires_grad=True)
    sim.reset()
    sim.update(amplitude)
    sim.unclamped_percept().sum().backward()
    assert torch.all(torch.isfinite(amplitude.grad))


class TestMerging:
    def test_adjacent_electrodes_merge_into_one_phosphene(self,
                                                          pair_simulator):
        sim = _field(pair_simulator, 'merged')
        _percept(sim, [CURRENT_A, CURRENT_A])
        assert len(np.unique(sim._volumes.labels[0])) == 1
        _percept(sim, [CURRENT_A, 0.0])
        assert len(np.unique(sim._volumes.labels[0])) == 2

    def test_a_middle_electrode_bridges_its_neighbors(
            self, params_override, synthetic_atlas, synthetic_jacobian):
        """Two electrodes too far apart to merge become one phosphene
        once the tissue between them is recruited."""
        params = params_override(aggregation__mode='field_superposition',
                                 aggregation__phosphene_merging='merged')
        sim = _simulator(params, synthetic_atlas, synthetic_jacobian,
                         _line(synthetic_atlas, 3))
        low = 10e-6
        _percept(sim, [low, 0.0, low])
        assert len(np.unique(sim._volumes.labels[0])) == 3
        _percept(sim, [low, low, low])
        assert len(np.unique(sim._volumes.labels[0])) == 1

    def test_a_merged_phosphene_peaks_at_its_brightest_member(
            self, pair_simulator):
        sim = _field(pair_simulator, 'merged')
        merged = _percept(sim, [CURRENT_A, 0.5 * CURRENT_A])
        intensity = sim.brightness.get() * sim.detection_probability()
        assert torch.isclose(merged.max(), intensity.max(), rtol=1e-4)

    def test_merged_differs_from_grown(self, pair_simulator):
        merged = _percept(_field(pair_simulator, 'merged'),
                          [CURRENT_A, CURRENT_A])
        grown = _percept(_field(pair_simulator, 'grown'),
                         [CURRENT_A, CURRENT_A])
        assert not torch.allclose(merged, grown, rtol=1e-3, atol=1e-4)
        # One phosphene: never brighter than its brightest member's blob.
        assert float(merged.max()) <= float(grown.max())

    def test_a_reach_shorter_than_the_lattice_leaves_electrodes_alone(
            self, pair_simulator):
        """At a large K no site may lie within reach of an electrode."""
        sim = pair_simulator(aggregation__mode='field_superposition',
                             aggregation__phosphene_merging='merged',
                             current_spread__k_ua_per_mm2=675_000.)
        _percept(sim, [CURRENT_A, CURRENT_A])
        assert len(np.unique(sim._volumes.labels[0])) == 2

    def test_unknown_merging_is_rejected(self, pair_simulator):
        with pytest.raises(ValueError):
            pair_simulator(aggregation__phosphene_merging='blobs')


class TestReporting:
    @pytest.mark.parametrize('simultaneous,effective', [
        (True, 'field_superposition'), (False, 'independent')])
    def test_state_and_description_name_both_modes(
            self, pair_simulator, simultaneous, effective):
        sim = pair_simulator(aggregation__mode='field_superposition',
                             aggregation__simultaneous_pulses=simultaneous)
        _percept(sim, [CURRENT_A, CURRENT_A])
        state = sim.get_state()
        assert state['aggregation_configured'] == 'field_superposition'
        assert state['aggregation_effective'].startswith(effective)
        text = sim.describe()
        assert 'configured field_superposition' in text
        assert f'effective {effective}' in text

    def test_unknown_mode_is_rejected(self, pair_simulator):
        with pytest.raises(ValueError):
            pair_simulator(aggregation__mode='probability_summation')
