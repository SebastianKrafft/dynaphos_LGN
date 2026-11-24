"""Current spread and the recruitment kernel."""
import copy

import pytest
import torch

from dynaphos_lgn import current_spread as cs


def kernel(params, **current_spread_overrides):
    """Build a kernel from the config with a few keys overridden."""
    params = copy.deepcopy(params)
    params['current_spread'].update(current_spread_overrides)
    return cs.RecruitmentKernel.from_params(params)


K = 675.0   # what dynaphos_lgn/config/params_lgn.yaml ships; asserted below


class TestStoneyLaw:
    def test_the_shipped_config_still_carries_the_v1_value(self, lgn_params):
        """These tests hardcode K so they check arithmetic rather than
        re-reading the config and trivially agreeing with it. This is
        the one place that pins the two together."""
        assert cs.excitability_constant_ua_per_mm2(lgn_params) == K

    def test_radius_at_the_lgn_behavioral_threshold(self, lgn_params):
        """40 uA (Pezaris & Reid 2007) under the transplanted V1 K gives
        ~244 um -- the number the multi-electrode spacing argument
        turns on."""
        threshold, _ = cs.lgn_threshold_ua(lgn_params)
        r = float(cs.stoney_radius_mm(threshold, K))
        assert r == pytest.approx(0.2434, abs=1e-3)

    def test_scales_as_the_square_root_of_current(self):
        r1 = float(cs.stoney_radius_mm(40.0, K))
        r4 = float(cs.stoney_radius_mm(160.0, K))
        assert r4 == pytest.approx(2 * r1)

    def test_negative_current_is_clamped_not_nan(self):
        assert float(cs.stoney_radius_mm(-5.0, K)) == 0.0

    def test_torch_and_numpy_paths_agree(self):
        i = 73.0
        assert (float(cs.stoney_radius_mm(torch.tensor(i), K))
                == pytest.approx(float(cs.stoney_radius_mm(i, K))))


class TestClassScaling:
    def test_default_applies_no_scaling(self, lgn_params):
        for cls in ('parvo', 'magno'):
            assert (cs.k_for_cell_class(cls, lgn_params)
                    == cs.excitability_constant_ua_per_mm2(lgn_params))

    def test_larger_cells_get_a_smaller_k(self, params_override):
        """Magnocellular axons are thicker (2.8 vs 2.0 um), so a
        size-scaled model must make them MORE excitable, i.e. smaller K
        and wider spread."""
        params = params_override(current_spread__class_scaling='diameter')
        k_p = cs.k_for_cell_class('parvo', params)
        k_m = cs.k_for_cell_class('magno', params)
        assert k_m < k_p
        assert cs.stoney_radius_mm(40.0, k_m) > cs.stoney_radius_mm(40.0, k_p)

    def test_unknown_scaling_is_rejected(self, params_override):
        params = params_override(current_spread__class_scaling='handwave')
        with pytest.raises(ValueError):
            cs.k_for_cell_class('parvo', params)

    def test_a_list_of_classes_uses_their_mean_diameter(self,
                                                        params_override):
        params = params_override(current_spread__class_scaling='diameter')
        d = params['current_spread']['axon_diameter_um']
        expected = K * d['parvo'] / ((d['parvo'] + d['magno']) / 2)
        assert cs.k_for_cell_class(['magno', 'parvo'], params) \
            == pytest.approx(expected)

    def test_a_class_without_a_diameter_is_rejected(self, params_override):
        params = params_override(current_spread__class_scaling='diameter')
        with pytest.raises(KeyError):
            cs.k_for_cell_class(['parvo', 'konio'], params)


class TestRecruitmentKernel:
    @pytest.mark.parametrize('contour', [0.5, 0.1353352832366127])
    def test_weight_at_the_nominal_radius_equals_the_contour_value(
            self, lgn_params, contour):
        """This anchoring is what stops the size model and the per-layer
        recruitment model from drifting into two different radii."""
        k = kernel(lgn_params, contour_value=contour)
        i = torch.tensor([[55.0]])
        r = k.radius_mm(i)
        assert float(k.weights(r, i)) == pytest.approx(contour, abs=1e-5)

    def test_weight_is_one_at_the_electrode(self, lgn_params):
        k = kernel(lgn_params)
        w = k.weights(torch.zeros(1, 1), torch.tensor([[40.0]]))
        assert float(w) == pytest.approx(1.0, abs=1e-6)

    def test_weight_decreases_with_distance(self, lgn_params):
        k = kernel(lgn_params)
        d = torch.linspace(0, 1.0, 25).unsqueeze(0)
        w = k.weights(d, torch.tensor([[40.0]]))
        assert torch.all(torch.diff(w[0]) < 0)

    def test_weight_increases_with_current(self, lgn_params):
        k = kernel(lgn_params)
        d = torch.full((3, 1), 0.3)
        w = k.weights(d, torch.tensor([[20.0], [60.0], [180.0]]))
        assert torch.all(torch.diff(w.squeeze()) > 0)

    def test_gradient_flows_to_current_everywhere(self, lgn_params):
        """The reason the kernel must be smooth: a hard in/out test has
        zero gradient in I, so the optimizer could never move."""
        k = kernel(lgn_params)
        i = torch.tensor([[10.0], [40.0], [200.0]], requires_grad=True)
        d = torch.linspace(0.01, 1.2, 30).unsqueeze(0)
        k.weights(d, i).sum().backward()
        assert torch.all(i.grad.abs() > 0)

    def test_effective_radius_exceeds_the_nominal_one(self, lgn_params):
        k = kernel(lgn_params)
        assert (k.effective_radius_mm(40.0, 1e-3)
                > k.radius_mm(40.0))

    def test_effective_radius_actually_bounds_the_tail(self, lgn_params):
        k = kernel(lgn_params)
        tail = 1e-3
        r_eff = torch.tensor([[float(k.effective_radius_mm(40.0, tail))]])
        assert float(k.weights(r_eff, torch.tensor([[40.0]]))) \
            == pytest.approx(tail, rel=0.05)

    def test_rejects_nonsense_configuration(self, lgn_params):
        with pytest.raises(ValueError):
            kernel(lgn_params, contour_value=1.0)


class TestFieldForm:
    def test_one_electrode_drive_reproduces_the_distance_kernel(
            self, lgn_params):
        k = kernel(lgn_params)
        d = torch.linspace(0.01, 0.6, 50, dtype=torch.float64)
        current = torch.tensor(60.0, dtype=torch.float64)
        drive = (k.radius_mm(current) / d) ** 2
        assert torch.allclose(k.weights_from_drive(drive),
                              k.weights(d, current), atol=1e-12)

    def test_scale_current_inverts_the_radius(self, lgn_params):
        k = kernel(lgn_params)
        current = torch.tensor([30.0, 60.0], dtype=torch.float64)
        scaled = k.scale_current(current, torch.tensor(1.2,
                                                       dtype=torch.float64))
        assert torch.allclose(k.radius_mm(scaled),
                              1.2 * k.radius_mm(current), rtol=1e-6)

