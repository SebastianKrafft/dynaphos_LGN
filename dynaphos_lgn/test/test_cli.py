"""The `python -m dynaphos_lgn.build` entry point.

The Jacobian cache is the one build step that is expensive enough that
nobody will run it twice to find out it was misconfigured, so its
command line gets tested like a deliverable: exit codes, refusal to
fabricate, and the read-back checks that would catch a wrongly-parsed
atlas.
"""
import numpy as np
import pytest

from dynaphos_lgn import build
from dynaphos_lgn.atlas import JacobianAtlas
from dynaphos_lgn.params import require


@pytest.fixture(scope='module')
def full_scale_params(lgn_params):
    """The shipped config with the real grid shape, for the CLI tests.

    The other suites shrink the synthetic atlas for speed; these tests
    exercise the file-size check and the .DAT round trip, which are only
    meaningful at the layout the real files use.
    """
    import copy
    params = copy.deepcopy(lgn_params)
    params['synthetic_atlas'].update(
        shape_ml_dv_ap=[240, 280, 320], max_eccentricity_deg=90.0,
        eccentricity_scale_deg=2.0, n_layers=6,
        inclination_span_deg=150.0, margin_vox=20)
    return params


@pytest.fixture(scope='module')
def fake_atlas_dir(tmp_path_factory, full_scale_params):
    """Full-shape .DAT files written from the synthetic map.

    Deliberately the real (240, 280, 320) grid rather than a small one:
    the point is to exercise the same file layout, dtypes and Fortran
    ordering the real atlas uses, which a conveniently-shaped test
    volume would not.
    """
    from dynaphos_lgn.synthetic_atlas import SyntheticAtlas
    directory = tmp_path_factory.mktemp('atlas')
    atlas = SyntheticAtlas(full_scale_params)
    for name, array, dtype in (('ECC.DAT', atlas.eccentricity, np.int16),
                               ('INCL.DAT', atlas.inclination, np.int16),
                               ('LAYERS.DAT', atlas.layer, np.int8),
                               ('CELLS.DAT', atlas.cells, np.int16)):
        array.astype(dtype).ravel(order='F').tofile(directory / name)
    return directory


class TestFileSizeCheck:
    def test_correct_sizes_pass(self, fake_atlas_dir, full_scale_params):
        text = build.check_atlas_files(fake_atlas_dir, full_scale_params)
        assert 'MISMATCH' not in text
        assert 'MISSING' not in text

    def test_a_wrong_width_is_caught(self, tmp_path, fake_atlas_dir,
                                     full_scale_params):
        """A headerless binary read at the wrong width does not error --
        it silently yields an entirely wrong volume. This is the check
        that stops that reaching the science."""
        import shutil
        specs = build.atlas_file_specs(full_scale_params)
        for name in specs:
            shutil.copy(fake_atlas_dir / name, tmp_path / name)
        # Rewrite LAYERS.DAT at int16, the width the other loader in this
        # repo assumes, and confirm the mismatch is reported.
        raw = np.fromfile(tmp_path / 'LAYERS.DAT', dtype=np.int8)
        raw.astype(np.int16).tofile(tmp_path / 'LAYERS.DAT')
        text = build.check_atlas_files(tmp_path, full_scale_params)
        assert 'MISMATCH' in text
        assert 'bytes/voxel' in text

    def test_missing_files_are_named(self, tmp_path, full_scale_params):
        text = build.check_atlas_files(tmp_path, full_scale_params)
        assert text.count('MISSING') == len(
            build.atlas_file_specs(full_scale_params))


class TestNoSilentFallback:
    def test_cache_refuses_to_invent_an_atlas(self, lgn_params, tmp_path):
        """Falling back here would write a file named exactly like the
        real cache and containing nothing real -- worse than failing."""
        with pytest.raises(FileNotFoundError):
            build.build_atlas(lgn_params, tmp_path / 'nowhere',
                              fallback_to_synthetic=False)

    def test_exit_code_two_on_a_missing_atlas(self, tmp_path, capsys):
        code = build.main(['--cache-jacobian',
                           '--atlas-dir', str(tmp_path / 'nowhere')])
        assert code == 2
        assert 'error:' in capsys.readouterr().out

    def test_refuses_to_do_nothing_silently(self, tmp_path):
        with pytest.raises(SystemExit):
            build.main(['--atlas-dir', str(tmp_path)])


class TestEndToEnd:
    def test_check_only_writes_nothing(self, fake_atlas_dir, capsys):
        before = sorted(p.name for p in fake_atlas_dir.iterdir())
        code = build.main(['--check-only', '--atlas-dir',
                           str(fake_atlas_dir), '--quiet'])
        assert code == 0
        assert sorted(p.name for p in fake_atlas_dir.iterdir()) == before
        assert 'Atlas read-back check' in capsys.readouterr().out

    @pytest.mark.slow
    def test_cache_jacobian_writes_a_loadable_cache(self, fake_atlas_dir,
                                                    tmp_path,
                                                    full_scale_params):
        code = build.main(['--cache-jacobian', '--atlas-dir',
                           str(fake_atlas_dir), '--quiet',
                           '--report', str(tmp_path / 'report.txt')])
        assert code == 0
        cache = fake_atlas_dir / require(full_scale_params,
                                         'atlas.jacobian_cache')
        assert cache.exists()

        loaded = JacobianAtlas.load(cache, full_scale_params)
        assert loaded.jacobian.shape == (240, 280, 320, 2, 3)
        assert loaded.jacobian_valid.any()
        # The on-disk key is still the historical name.
        assert loaded.unreliable_beyond_neighborhood_flag is not None

        report = (tmp_path / 'report.txt').read_text()
        assert 'Jacobian atlas validation' in report
        assert 'sigma1' in report

    @pytest.mark.slow
    def test_second_run_loads_instead_of_recomputing(self, fake_atlas_dir,
                                                     full_scale_params):
        """The cache is the whole point -- a second run must not redo the
        fit.

        Timed on the fit specifically rather than on the whole command:
        reading 172 MB of .DAT files and building the report cost real
        seconds either way, and folding those into the assertion would
        make it a flaky measure of the wrong thing.
        """
        result = build.cache_jacobian(full_scale_params, fake_atlas_dir)
        assert result['fit_seconds'] < 20


class TestReports:
    def test_atlas_report_flags_a_synthetic_atlas(self, synthetic_atlas,
                                                  lgn_params):
        text = build.atlas_report(synthetic_atlas, lgn_params)
        assert 'SYNTHETIC' in text

    def test_jacobian_report_bins_by_eccentricity(self, synthetic_atlas,
                                                  synthetic_jacobian,
                                                  lgn_params):
        text = build.jacobian_report(synthetic_jacobian, synthetic_atlas,
                                     lgn_params)
        assert 'sigma1' in text and 'aniso' in text
        assert 'unreliable' in text
        # The caveat must travel with the number it qualifies.
        assert 'isotropic by construction' in text
        assert 'QUALITATIVE' in text

    def test_sampled_medians_match_the_exact_ones(self, synthetic_atlas,
                                                   synthetic_jacobian,
                                                   params_override):
        """The report samples for speed, so the sample has to be
        faithful: a sampled median that drifted would misreport the
        magnification field while looking authoritative."""
        sampled = build.jacobian_report(
            synthetic_jacobian, synthetic_atlas,
            params_override(validation__jacobian_report_sample_size=50_000))
        exact = build.jacobian_report(
            synthetic_jacobian, synthetic_atlas,
            params_override(validation__jacobian_report_sample_size=(
                int(synthetic_jacobian.jacobian_valid.sum()) + 1)))

        def sigma1_column(text):
            out = []
            for line in text.splitlines():
                parts = line.split()
                if len(parts) == 6 and '-' in parts[0]:
                    try:
                        out.append(float(parts[2]))
                    except ValueError:
                        pass
            return out

        a, b = sigma1_column(sampled), sigma1_column(exact)
        assert a and len(a) == len(b)
        assert np.allclose(a, b, rtol=0.05)

    def test_decompose_valid_matches_a_full_decomposition(
            self, synthetic_jacobian, params_override):
        """The chunked, valid-only SVD is a memory optimisation, so it
        has to be numerically identical to the thing it replaces."""
        from dynaphos_lgn.magnification import JacobianMagnification
        # Well under the valid-voxel count, so several chunks actually
        # run -- the shipped size would swallow this atlas in one.
        original = synthetic_jacobian.params
        synthetic_jacobian.params = params_override(
            atlas__jacobian__svd_chunk_size=100_000)
        try:
            major, minor, _ = synthetic_jacobian.decompose_valid()
        finally:
            synthetic_jacobian.params = original
        reference = JacobianMagnification.decompose(
            synthetic_jacobian.jacobian)
        ok = synthetic_jacobian.jacobian_valid
        assert np.allclose(major[ok], reference[0][ok], rtol=1e-5)
        assert np.allclose(minor[ok], reference[1][ok], rtol=1e-5)
        assert np.all(np.isnan(major[~ok]))
