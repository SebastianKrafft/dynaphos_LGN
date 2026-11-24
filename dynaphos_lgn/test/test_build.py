"""The one-call build path."""
import numpy as np
import pytest
import torch

from dynaphos.utils import load_params
from dynaphos_lgn.build import (build_atlas, build_simulator,
                                n_electrodes_for)
from dynaphos_lgn.current_spread import RecruitmentKernel
from dynaphos_lgn.params import PARAMS_PATH

from .conftest import REAL_ATLAS_DIR, requires_real_atlas


class TestBuild:
    def test_missing_atlas_falls_back_to_synthetic_loudly(
            self, params_override, caplog, tmp_path):
        params = params_override(
            atlas__directory=str(tmp_path / 'nowhere'),
            synthetic_atlas__shape_ml_dv_ap=(32, 32, 48))
        atlas = build_atlas(params)
        assert atlas.is_synthetic
        assert any('synthetic' in r.message for r in caplog.records)

    def test_kernel_reflects_the_configured_constants(self, lgn_params):
        kernel = RecruitmentKernel.from_params(lgn_params)
        assert kernel.k == lgn_params['current_spread']['k_ua_per_mm2']
        assert kernel.contour_value == lgn_params['current_spread'][
            'contour_value']

    def test_electrodes_are_sized_by_the_per_voxel_jacobian(
            self, params_override):
        params = params_override(electrodes__n_electrodes=6,
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

    def test_explicit_voxel_indices_are_honored(self, lgn_params,
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


@requires_real_atlas
def test_shipped_config_builds_the_configured_electrodes():
    """The shipped config, untouched, on the real atlas."""
    params = load_params(str(PARAMS_PATH))
    params['run']['gpu'] = None
    sim, parts = build_simulator(params, atlas_dir=REAL_ATLAS_DIR)
    for name, part in parts['hemispheres'].items():
        assert (part['electrode_array'].n_electrodes
                == n_electrodes_for(params, name))
    img = sim(torch.full((sim.num_phosphenes,), 60e-6))
    assert torch.isfinite(img).all()


class TestFoveola:
    """The foveola file's loader and the placeholder rule, on the
    synthetic atlas: the checks need only a grid and its eccentricity."""

    @staticmethod
    def write(atlas, directory, rows, bom=True):
        text = '\r\n'.join('\t'.join(str(v) for v in row) for row in rows)
        path = directory / atlas.params['atlas']['foveola_file']
        path.write_text(('﻿' if bom else '') + text, encoding='utf-8')

    def test_reads_the_supplied_format(self, synthetic_atlas, tmp_path):
        """Tab-separated, CRLF, with a UTF-8 byte-order mark."""
        rows = synthetic_atlas.foveola_column[:3]
        self.write(synthetic_atlas, tmp_path, rows)
        assert np.array_equal(
            synthetic_atlas.load_foveola_column(tmp_path), rows)

    def test_missing_file_gives_none_loudly(self, synthetic_atlas, tmp_path,
                                            caplog):
        assert synthetic_atlas.load_foveola_column(tmp_path) is None
        assert any('foveola' in r.message for r in caplog.records)

    def test_refuses_a_voxel_off_the_fovea(self, synthetic_atlas, tmp_path):
        off = np.argwhere(synthetic_atlas.valid
                          & (synthetic_atlas.eccentricity > 0))[:1]
        self.write(synthetic_atlas, tmp_path, off)
        with pytest.raises(ValueError, match='zero eccentricity'):
            synthetic_atlas.load_foveola_column(tmp_path)

    def test_refuses_a_voxel_outside_the_grid(self, synthetic_atlas,
                                              tmp_path):
        self.write(synthetic_atlas, tmp_path,
                   [synthetic_atlas.atlas_shape])
        with pytest.raises(ValueError, match='inside the grid'):
            synthetic_atlas.load_foveola_column(tmp_path)

    def test_placeholder_moves_off_zero_eccentricity_only(self,
                                                          params_override):
        from dynaphos_lgn.synthetic_atlas import SyntheticAtlas
        atlas = SyntheticAtlas(params_override(
            synthetic_atlas__shape_ml_dv_ap=(24, 24, 40)))
        flat = atlas.ipsi_sentinel_values[1] * atlas.incl_scale
        fovea = tuple(atlas.foveola_column[:2].T)
        elsewhere = tuple(np.argwhere(atlas.valid
                                      & (atlas.eccentricity > 0))[:1].T)
        atlas.inclination[fovea] = [-flat, flat]
        atlas.inclination[elsewhere] = flat

        atlas.clear_foveal_placeholder()
        assert list(atlas.inclination[fovea]) == [-90 * atlas.incl_scale,
                                                  90 * atlas.incl_scale]
        assert atlas.inclination[elsewhere] == flat
