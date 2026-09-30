"""Koniocellular layers and the class-mixture percept."""
import numpy as np
import pytest
import torch

from dynaphos_lgn.atlas import MirroredAtlas
from dynaphos_lgn.build import build_atlas, build_simulator
from dynaphos_lgn.params import resolve_class_codes
from dynaphos_lgn.receptive_fields import center_radius_deg

from .conftest import requires_real_atlas

K_CODES = list(range(5, 11))


@pytest.fixture
def konio_params(params_override):
    return params_override(konio__enabled=True)


@pytest.fixture
def konio_atlas(konio_params):
    return build_atlas(konio_params, synthetic=True)


def build(params, **percept):
    params['percept'].update(percept)
    simulator, parts = build_simulator(params, synthetic=True,
                                       rng=np.random.default_rng(0))
    return simulator, parts['electrode_array']


class TestAnatomy:
    def test_synthetic_gaps_are_labelled_by_their_neighbours(
            self, konio_atlas):
        """Layers 1-4 stacked ventral to dorsal: the gaps above layers 1,
        2 and 3 are K2 (below magno 2), K3 (magno below parvo) and K6
        (below the top parvo layer)."""
        present = set(np.unique(konio_atlas.layer)) & set(K_CODES)
        assert present == {6, 7, 10}

    def test_disabled_leaves_the_atlas_alone(self, synthetic_atlas):
        assert not np.isin(synthetic_atlas.layer, K_CODES).any()
        assert not synthetic_atlas.valid[synthetic_atlas.layer == 0].any()

    def test_k_cells_are_the_configured_share(self, konio_atlas,
                                              konio_params):
        k = np.isin(konio_atlas.layer, K_CODES)
        principal = np.isin(konio_atlas.layer, [1, 2, 3, 4])
        k_cells = konio_atlas.cells_per_voxel[k].sum()
        share = k_cells / (k_cells
                           + konio_atlas.cells_per_voxel[principal].sum())
        assert share == pytest.approx(
            konio_params['konio']['fraction_of_total'], rel=0.01)
        assert np.all(konio_atlas.valid[k])

    def test_mirror_carries_the_labels(self, konio_atlas):
        mirrored = MirroredAtlas(konio_atlas)
        assert np.array_equal(np.unique(mirrored.layer),
                              np.unique(konio_atlas.layer))

    @requires_real_atlas
    def test_real_k_layers_are_about_100_um_thick(self, konio_params):
        from dynaphos_lgn.konio import _median_thickness_vox
        atlas = build_atlas(konio_params, fallback_to_synthetic=False)
        k = np.isin(atlas.layer, K_CODES)
        assert 400_000 < k.sum() < 700_000
        assert 3 <= _median_thickness_vox(atlas, k) <= 5


class TestClasses:
    def test_subclasses_and_lists_resolve(self, lgn_params):
        assert resolve_class_codes(lgn_params, 'konio_ventral') == [5, 6]
        assert resolve_class_codes(lgn_params, ['magno', 'parvo']) == \
            [1, 2, 3, 4]

    def test_subclass_rf_scales_the_parent(self, lgn_params):
        ecc = np.array([2.0, 20.0])
        konio = center_radius_deg(ecc, lgn_params, 'konio')
        scales = lgn_params['receptive_fields']['subclass_rc_scale']
        for name, scale in scales.items():
            assert np.allclose(center_radius_deg(ecc, lgn_params, name),
                               scale * konio)

    def test_excluding_konio_keeps_the_baseline_electrodes(
            self, lgn_params, params_override):
        _, baseline = build(params_override())
        _, with_k = build(params_override(
            konio__enabled=True, electrodes__cell_class=['magno', 'parvo']))
        assert np.array_equal(baseline.voxel_indices, with_k.voxel_indices)

    def test_konio_electrodes_sit_in_k_layers(self, params_override):
        _, array = build(params_override(konio__enabled=True,
                                         electrodes__cell_class='konio'))
        layer = array.atlas.layer[tuple(array.voxel_indices.T)]
        assert np.isin(layer, K_CODES).all()


class TestClassMixture:
    @pytest.fixture
    def amplitude(self):
        return torch.full((8,), 60e-6)

    def test_fractions_sum_to_one_and_peak_stays_below_one(
            self, konio_params, amplitude):
        sim, _ = build(konio_params, model='class_mixture')
        image = sim(amplitude)
        assert torch.allclose(sim.class_fraction.sum(-1),
                              torch.ones(sim.num_phosphenes))
        assert sim.spatial_activation().max() <= 1 + 1e-6
        assert image.max() <= 1

    def test_gradients_reach_every_renderer(self, konio_params):
        sim, _ = build(konio_params, model='class_mixture')
        assert len(sim.ellipse_index) and len(sim.scatter_index)
        amplitude = torch.full((sim.num_phosphenes,), 60e-6,
                               requires_grad=True)
        sim(amplitude).sum().backward()
        assert torch.isfinite(amplitude.grad).all()
        assert (amplitude.grad != 0).any()

    def test_without_rf_blur_ellipses_match_the_single_percept(
            self, konio_params, params_override, amplitude):
        single, _ = build(params_override(konio__enabled=True))
        mixture, _ = build(konio_params, model='class_mixture',
                           rf_blur_scale=0.0)
        single(amplitude)
        mixture(amplitude)
        idx = single.ellipse_index
        assert torch.allclose(single.spatial_activation()[idx],
                              mixture.spatial_activation()[idx], atol=1e-6)

    def test_konio_changes_only_electrodes_that_reach_it(
            self, konio_params, params_override, amplitude):
        mp, _ = build(params_override(konio__enabled=True),
                      model='class_mixture', classes=['magno', 'parvo'])
        mpk, _ = build(konio_params, model='class_mixture')
        mp(amplitude)
        mpk(amplitude)
        change = (mp.spatial_activation() - mpk.spatial_activation()
                  ).abs().amax(dim=(-2, -1))
        reaches_k = mpk.class_fraction[:, -1] > 0
        # A wider scatter kernel truncates slightly differently: ~3e-4
        assert torch.all(change[~reaches_k] < 1e-3)
        assert torch.any(change[reaches_k] > 1e-2)

    def test_unknown_model_is_rejected(self, konio_params):
        with pytest.raises(KeyError, match='percept.model'):
            build(konio_params, model='nonsense')
