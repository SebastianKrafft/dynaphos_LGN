"""The one-call build path."""
import numpy as np
import pytest
import torch

from dynaphos_lgn.build import (build_atlas, build_kernel,
                                build_magnification_model, build_simulator)


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

    @pytest.mark.parametrize('choice', ['atlas_gradient', 'anisotropic_gradient',
                                        'malpeli_density'])
    def test_non_jacobian_model_actually_sizes_the_electrodes(
            self, params_override, choice):
        """A non-'jacobian' `magnification.model` must drive sizing, not
        just get built and discarded.

        `build_electrode_array` used to hand the raw per-voxel Jacobian
        atlas to every array regardless of `magnification.model`, and
        `LGNElectrodeArray._resolve_magnification` treats that Jacobian
        as authoritative whenever it is given -- so the configured model
        only ever filled its gaps. Every electrode is flagged
        `scalar_magnification` exactly when the chosen model, not the
        per-voxel Jacobian, sized it.
        """
        params = params_override(magnification__model=choice,
                                 electrodes__n_electrodes=6,
                                 synthetic_atlas__shape_ml_dv_ap=(32, 32, 64))
        sim, parts = build_simulator(params, synthetic=True)
        array = parts['electrode_array']
        assert array.jacobian_atlas is None
        assert array.flags.scalar_magnification.all()

    def test_jacobian_model_still_uses_the_per_voxel_jacobian(
            self, params_override):
        params = params_override(magnification__model='jacobian',
                                 electrodes__n_electrodes=6,
                                 synthetic_atlas__shape_ml_dv_ap=(32, 32, 64))
        sim, parts = build_simulator(params, synthetic=True)
        array = parts['electrode_array']
        assert array.jacobian_atlas is not None
        assert not array.flags.scalar_magnification.any()

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
