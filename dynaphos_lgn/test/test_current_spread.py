"""Current spread, recruitment kernels and multi-electrode aggregation."""
import copy
import math

import pytest
import torch

from dynaphos_lgn import current_spread as cs


def kernel(params, **current_spread_overrides):
    """Build a kernel from the config with a few keys overridden.

    Tests exercise kernel variants, but the kernel can only be built
    from a config now -- so they override the config rather than
    constructing one by hand, which also keeps them honest about what
    the config is actually able to express.
    """
    params = copy.deepcopy(params)
    params['current_spread'].update(current_spread_overrides)
    return cs.RecruitmentKernel.from_params(params)


K = 675.0   # what config/params_lgn.yaml ships; asserted below


class TestStoneyLaw:
    def test_the_shipped_config_still_carries_the_v1_value(self, lgn_params):
        """These tests hardcode K so they check arithmetic rather than
        re-reading the config and trivially agreeing with it. This is
        the one place that pins the two together."""
        assert cs.excitability_constant_ua_per_mm2(lgn_params) == K

    def test_radius_at_the_lgn_behavioural_threshold(self, lgn_params):
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
        """Magnocellular nucleoli are larger (3.00 vs 2.25 um), so a
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


class TestRecruitmentKernel:
    @pytest.mark.parametrize('shape', ['gaussian', 'logistic'])
    @pytest.mark.parametrize('contour', [0.5, math.exp(-2)])
    def test_weight_at_the_nominal_radius_equals_the_contour_value(
            self, lgn_params, shape, contour):
        """This anchoring is what stops the size model and the per-layer
        recruitment model from drifting into two different radii."""
        k = kernel(lgn_params, kernel_shape=shape, contour_value=contour)
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

    @pytest.mark.parametrize('shape', ['gaussian', 'logistic'])
    def test_gradient_flows_to_current_everywhere(self, lgn_params, shape):
        """The reason the kernel must be smooth: a hard in/out test has
        zero gradient in I, so the optimiser could never move."""
        k = kernel(lgn_params, kernel_shape=shape)
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

    def test_bosking_saturates_while_stoney_does_not(self, lgn_params):
        stoney = kernel(lgn_params, spread_model='stoney')
        bosking = kernel(lgn_params, spread_model='bosking')
        big = 5000.0
        ceiling = lgn_params['current_spread']['bosking']['max_diameter_mm']
        assert float(bosking.radius_mm(big)) < 0.5 * ceiling * 1.001
        assert float(stoney.radius_mm(big)) > float(bosking.radius_mm(big))

    def test_rejects_nonsense_configuration(self, lgn_params):
        with pytest.raises(ValueError):
            kernel(lgn_params, kernel_shape='triangular')
        with pytest.raises(ValueError):
            kernel(lgn_params, contour_value=1.0)


class TestAggregation:
    def setup_method(self):
        self.w = torch.tensor([[[0.6, 0.2], [0.5, 0.1]]])  # (1, 2, 2)

    def test_independent_sums(self):
        out = cs.aggregate_activation(self.w, 'independent')
        assert torch.allclose(out, torch.tensor([[1.1, 0.3]]))

    def test_probability_summation_never_exceeds_one(self):
        w = torch.rand(1, 8, 5)
        out = cs.aggregate_activation(w, 'probability_summation')
        assert torch.all(out <= 1.0) and torch.all(out >= 0.0)

    def test_probability_summation_is_below_linear(self):
        lin = cs.aggregate_activation(self.w, 'independent')
        prob = cs.aggregate_activation(self.w, 'probability_summation')
        assert torch.all(prob <= lin)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            cs.aggregate_activation(self.w, 'telepathy')


class TestThresholdBracket:
    def test_ceiling_is_never_below_the_floor(self, lgn_params):
        for spacing in (0.05, 0.2, 0.5, 2.0):
            b = cs.threshold_reduction_bracket(4, spacing, lgn_params)
            assert b['ceiling_fraction'] >= b['floor_fraction']

    def test_field_summation_vanishes_at_wide_spacing(self, lgn_params):
        tight = cs.threshold_reduction_bracket(4, 0.05, lgn_params)
        wide = cs.threshold_reduction_bracket(4, 2.0, lgn_params)
        assert tight['ceiling_fraction'] > wide['ceiling_fraction']
        assert wide['ceiling_fraction'] == pytest.approx(
            wide['floor_fraction'])

    def test_regime_follows_spacing_over_radius(self, lgn_params):
        assert 'field-summation' in cs.threshold_reduction_bracket(
            4, 0.1, lgn_params)['regime']
        assert 'probability-summation' in cs.threshold_reduction_bracket(
            4, 1.5, lgn_params)['regime']

    def test_reports_the_geometric_quantity_that_decides_it(self, lgn_params):
        b = cs.threshold_reduction_bracket(4, 0.4, lgn_params)
        assert b['spacing_over_radius'] == pytest.approx(
            0.4 / b['spread_radius_mm'])
