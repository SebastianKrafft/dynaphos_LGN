"""Calibration diagnostics and the one-call build path."""
import numpy as np
import pytest
import torch

from dynaphos_lgn import calibration
from dynaphos_lgn.build import (build_atlas, build_kernel,
                                build_magnification_model, build_simulator)
from dynaphos_lgn.current_spread import excitability_constant_ua_per_mm2


class TestCalibration:
    def test_comparison_covers_every_electrode(self, built):
        _, array = built
        c = calibration.size_comparison(array, 80.0)
        assert len(c['ratio']) == array.n_electrodes
        assert np.all(c['vurro_sigma_deg'] > 0)

    def test_equivalent_sigma_sits_between_the_two_axes(self, built):
        _, array = built
        c = calibration.size_comparison(array, 80.0)
        ok = np.isfinite(c['sigma_major_deg']) & np.isfinite(
            c['sigma_minor_deg'])
        assert np.all(c['sigma_equivalent_deg'][ok]
                      <= c['sigma_major_deg'][ok] + 1e-9)
        assert np.all(c['sigma_equivalent_deg'][ok]
                      >= c['sigma_minor_deg'][ok] - 1e-9)

    def test_implied_k_inverts_the_size_relation(self, built):
        """sigma scales as 1/sqrt(K), so substituting the implied K must
        reproduce the reference size. This is the check that the
        diagnostic is arithmetically self-consistent, whatever one
        thinks of the reference curve."""
        _, array = built
        r = calibration.implied_k(array, 80.0)
        finite = np.isfinite(r['k_implied_per_electrode'])
        predicted = (r['sigma_equivalent_deg'][finite]
                     * np.sqrt(r['k_current_ua_per_mm2']
                               / r['k_implied_per_electrode'][finite]))
        assert np.allclose(predicted, r['vurro_sigma_deg'][finite],
                           rtol=1e-6)

    def test_reports_a_range_because_the_shapes_disagree(self, built):
        """No single K reconciles a magnification-driven size law with a
        linear acuity proxy, and the spread is the point rather than
        noise to be averaged away."""
        _, array = built
        r = calibration.implied_k(array, 80.0)
        lo, hi = r['k_implied_range']
        assert hi > lo

    def test_report_leads_with_the_caveat(self, built):
        _, array = built
        text = calibration.report(array, 80.0)
        assert 'DIAGNOSTIC' in text
        assert 'unvalidated for LGN' in text

    def test_what_if_kernel_uses_the_implied_value(self, built, lgn_params):
        _, array = built
        kernel = calibration.kernel_with_implied_k(array, 80.0)
        assert kernel.k != excitability_constant_ua_per_mm2(lgn_params)
        assert kernel.shape == array.kernel.shape

    def test_larger_k_shrinks_the_rendered_size(self, built):
        _, array = built
        base = array.kernel.sigma_mm(80.0)
        tighter = calibration.kernel_with_implied_k(array, 80.0)
        if tighter.k > array.kernel.k:
            assert tighter.sigma_mm(80.0) < base
        else:
            assert tighter.sigma_mm(80.0) > base


class TestBuild:
    def test_missing_atlas_falls_back_to_synthetic_loudly(
            self, params_override, caplog, tmp_path):
        params = params_override(
            atlas__directory=str(tmp_path / 'nowhere'),
            synthetic_atlas__shape_ml_dv_ap=(32, 32, 48))
        atlas = build_atlas(params)
        assert getattr(atlas, 'is_synthetic', False)
        assert any('SYNTHETIC' in r.message for r in caplog.records)

    def test_kernel_reflects_the_configured_constants(self, lgn_params):
        kernel = build_kernel(lgn_params)
        assert kernel.k == lgn_params['current_spread'][
            'k_ua_per_mm2']
        assert kernel.shape == lgn_params['current_spread']['kernel_shape']

    @pytest.mark.parametrize('choice', ['jacobian', 'atlas_gradient',
                                        'malpeli_density'])
    def test_every_magnification_model_builds_and_answers(
            self, params_override, synthetic_atlas, synthetic_jacobian,
            choice):
        params = params_override(magnification__model=choice)
        model = build_magnification_model(params, synthetic_atlas,
                                          synthetic_jacobian)
        local = model.at_eccentricity(np.array([2.0, 8.0]))
        assert np.all(local.major_deg_per_mm > 0)

    def test_unknown_magnification_model_is_rejected(
            self, params_override, synthetic_atlas, synthetic_jacobian):
        params = params_override(magnification__model='vibes')
        with pytest.raises(ValueError):
            build_magnification_model(params, synthetic_atlas,
                                      synthetic_jacobian)

    def test_build_simulator_returns_every_part(self, params_override):
        params = params_override(electrodes__n_electrodes=4,
                                 synthetic_atlas__shape_ml_dv_ap=(32, 32, 64))
        sim, parts = build_simulator(params, synthetic=True)
        for key in ('atlas', 'jacobian_atlas', 'kernel', 'magnification',
                    'electrode_array'):
            assert key in parts
        img = sim(torch.full((sim.num_phosphenes,), 70e-6))
        assert torch.isfinite(img).all()

    def test_explicit_voxel_indices_are_honoured(self, lgn_params,
                                                 synthetic_atlas):
        params = lgn_params
        chosen = np.argwhere(synthetic_atlas.valid)[::9000][:3]
        sim, parts = build_simulator(params, synthetic=True,
                                     voxel_indices=chosen)
        assert sim.num_phosphenes == len(chosen)
        assert np.array_equal(parts['electrode_array'].voxel_indices, chosen)

    def test_out_of_view_electrodes_are_reported_not_dropped(
            self, params_override, caplog):
        params = params_override(run__view_angle=2.0,      # far too narrow
                                 electrodes__n_electrodes=6)
        sim, _ = build_simulator(params, synthetic=True)
        assert sim.num_phosphenes == 6, "electrodes must not be dropped"
        assert sim.out_of_view.any()
        assert any('outside the rendered' in r.message
                   for r in caplog.records)
