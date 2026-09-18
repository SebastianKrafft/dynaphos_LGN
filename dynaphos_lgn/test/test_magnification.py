"""Magnification models, checked against published numbers and against
an analytically-known synthetic map."""
import numpy as np
import pytest

from dynaphos_lgn import magnification as mg
from dynaphos_lgn.params import resolve_class_codes

from .conftest import requires_real_atlas


class TestMalpeliFormulas:
    """Eqs 1-2 of Malpeli, Lee & Baker (1996), verified full text."""

    @pytest.mark.parametrize('ecc, expected', [
        (1.0, 2.611e4), (3.0, 8639.), (5.0, 3958.), (11.0, 872.6),
        (27.0, 112.2)])
    def test_parvo_matches_published_table(self, lgn_params, ecc, expected):
        got = float(mg.malpeli_parvo_density(ecc, lgn_params))
        assert got == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize('ecc, expected', [
        (1.0, 603.0), (5.0, 290.0), (15.0, 41.06), (27.0, 14.81)])
    def test_magno_matches_published_table(self, lgn_params, ecc, expected):
        got = float(mg.malpeli_magno_density(ecc, lgn_params))
        assert got == pytest.approx(expected, rel=1e-3)

    def test_magno_is_non_monotonic_near_the_fovea(self, lgn_params):
        """The quadratic-inside-power-law term exists precisely because
        magnocellular density does not fall monotonically with
        eccentricity -- so this is a feature to protect, not a bug."""
        e = np.linspace(0.0, 4.0, 200)
        d = mg.malpeli_magno_density(e, lgn_params)
        assert np.argmax(d) > 0, "peak should not sit at the fovea"
        assert d[-1] < d.max()

    def test_parvo_is_monotonic(self, lgn_params):
        e = np.linspace(0.1, 60, 500)
        d = mg.malpeli_parvo_density(e, lgn_params)
        assert np.all(np.diff(d) < 0)

    def test_hemifield_integral_reproduces_atlas_cell_counts(self,
                                                             lgn_params):
        """A strong independent check that the density formula, the
        spherical solid-angle convention and the one-hemifield-per-LGN
        reading are mutually consistent.

        Erwin et al. (1999) report 1,048,571 parvocellular and 104,841
        magnocellular cells in the reconstruction. Integrating Malpeli's
        density over a spherical solid angle and halving (one LGN
        represents one hemifield) should land close to those.

        A FLAT (non-spherical) convention would be off by a very
        different amount at large eccentricity, so this also
        discriminates between the two readings the atlas's own
        documentation leaves open.
        """
        e = np.linspace(1e-3, 96.6, 200_000)
        de = np.deg2rad(e[1] - e[0])
        deg2_per_sr = (180.0 / np.pi) ** 2
        solid_angle = 2 * np.pi * np.sin(np.deg2rad(e)) * de * deg2_per_sr

        parvo = float(
            (mg.malpeli_parvo_density(e, lgn_params) * solid_angle).sum()) / 2
        magno = float(
            (mg.malpeli_magno_density(e, lgn_params) * solid_angle).sum()) / 2

        expected = lgn_params['validation']['erwin_1999']
        assert parvo == pytest.approx(expected['parvo_cells'], rel=0.06)
        assert magno == pytest.approx(expected['magno_cells'], rel=0.06)


class TestLocalMagnification:
    def test_isotropic_case_has_unit_anisotropy(self):
        v = np.array([3.0, 5.0])
        local = mg.LocalMagnification(v, v.copy(), np.zeros(2),
                                      np.zeros(2, bool), np.zeros(2, bool),
                                      np.ones(2, bool))
        assert np.allclose(local.anisotropy, 1.0)
        assert np.allclose(local.linear_mm_per_deg, 1.0 / v)

    def test_areal_and_linear_are_consistent(self):
        major, minor = np.array([4.0]), np.array([1.0])
        local = mg.LocalMagnification(major, minor, np.zeros(1),
                                      np.zeros(1, bool), np.zeros(1, bool),
                                      np.ones(1, bool))
        assert local.areal_deg2_per_mm2[0] == pytest.approx(4.0)
        assert local.linear_mm_per_deg[0] == pytest.approx(0.5)
        assert local.anisotropy[0] == pytest.approx(4.0)


class TestJacobianAgainstAnalyticGroundTruth:
    """The synthetic atlas has a closed-form retinotopic map, so the
    Jacobian machinery can be checked against the exact answer rather
    than against another estimate."""

    @pytest.mark.parametrize('ecc', [8.0, 15.0, 30.0])
    def test_recovers_the_analytic_magnifications(
            self, synthetic_atlas, synthetic_jacobian, ecc):
        major, minor, _ = synthetic_jacobian.singular_values
        ok = (synthetic_jacobian.jacobian_valid
              & (np.abs(synthetic_atlas.eccentricity_deg - ecc) < 1.0))
        assert ok.sum() > 100

        fitted = np.sort([np.nanmedian(major[ok]), np.nanmedian(minor[ok])])
        analytic = np.sort(np.asarray(
            synthetic_atlas.analytic_magnification_deg_per_mm(ecc),
            dtype=float))
        # 20% is loose on purpose: the fit sees a 25 um grid carrying
        # eccentricity quantised to 0.1 deg and inclination to 1 deg, so
        # exact agreement would be suspicious rather than reassuring.
        assert np.allclose(fitted, analytic, rtol=0.2)

    def test_singular_values_are_ordered_and_non_negative(
            self, synthetic_jacobian):
        major, minor, _ = synthetic_jacobian.singular_values
        ok = synthetic_jacobian.jacobian_valid
        assert np.all(major[ok] >= minor[ok])
        assert np.all(minor[ok] >= 0)

    def test_invalid_voxels_are_nan_not_zero(self, synthetic_jacobian):
        """Zero would silently read as 'no magnification' downstream;
        NaN cannot be mistaken for a measurement."""
        major, _, _ = synthetic_jacobian.singular_values
        assert np.all(np.isnan(major[~synthetic_jacobian.jacobian_valid]))


@pytest.fixture
def malpeli_model(lgn_params):
    """The density-derived model with rho_v supplied explicitly.

    The shipped config sets `cells_per_mm3` to null, meaning "derive it
    from the atlas" -- so a test that wants the model without an atlas
    has to say what rho_v is, which is exactly the behaviour being
    checked elsewhere.
    """
    return mg.MalpeliDensityMagnification(
        lgn_params, 'parvo', cells_per_mm3=0.36 / 0.025 ** 3)


class TestMalpeliDensityMagnification:
    def test_null_density_requires_the_atlas(self, lgn_params):
        """A null in the config means 'derive this', not 'guess'."""
        with pytest.raises(ValueError, match='from_atlas'):
            mg.MalpeliDensityMagnification(lgn_params, 'parvo')

    def test_interval_is_ordered(self, malpeli_model):
        m = malpeli_model
        lo, mid, hi = m.interval(np.array([1.0, 10.0, 30.0]))
        assert np.all(lo <= mid) and np.all(mid <= hi)

    def test_thicker_column_means_smaller_linear_magnification(
            self, malpeli_model):
        m = malpeli_model
        thin = m.map_length_mm(1.0)
        thick = m.map_length_mm(4.0)
        assert thick < thin

    def test_calibration_inverts_the_map_length(self, malpeli_model):
        m = malpeli_model
        implied = m.calibrate_column_thickness_mm(5.0)
        assert m.map_length_mm(implied) == pytest.approx(5.0, rel=1e-6)

    def test_reports_a_range_not_a_point(self, malpeli_model):
        """The project's rule: more than one arbitrary choice in the
        derivation chain means the output is an interval."""
        m = malpeli_model
        lo, mid, hi = m.interval(10.0)
        assert hi / lo > 1.5

    def test_rejects_a_class_it_was_not_built_for(self, malpeli_model):
        m = malpeli_model
        with pytest.raises(ValueError):
            m.at_eccentricity(10.0, cell_class='magno')

    def test_column_thickness_may_be_split_by_class(self,
                                                    params_override):
        """The measured T is ~2.4x larger for parvo than for magno, so
        a single shared triple is necessarily wrong for one class. Both
        config shapes have to work: the original shared triple, and a
        per-class split.
        """
        params = params_override(
            magnification__density_derived__column_thickness_mm={
                'parvo': [0.40, 0.84, 1.77],
                'magno': [0.20, 0.39, 0.74]})
        rho = 0.36 / 0.025 ** 3
        parvo = mg.MalpeliDensityMagnification(params, 'parvo',
                                               cells_per_mm3=rho)
        magno = mg.MalpeliDensityMagnification(params, 'magno',
                                               cells_per_mm3=rho)
        assert parvo.column_thickness_mm == (0.40, 0.84, 1.77)
        assert magno.column_thickness_mm == (0.20, 0.39, 0.74)

    def test_shared_column_thickness_still_works(self, lgn_params):
        """The original single-triple shape must keep working -- the
        split is an added option, not a migration."""
        model = mg.MalpeliDensityMagnification(
            lgn_params, 'magno', cells_per_mm3=0.36 / 0.025 ** 3)
        assert len(model.column_thickness_mm) == 3

    def test_partial_class_split_is_refused(self, params_override):
        """Half a split is worse than none: the missing class would
        quietly inherit whatever the dict iteration happened to give."""
        params = params_override(
            magnification__density_derived__column_thickness_mm={
                'parvo': [0.40, 0.84, 1.77]})
        with pytest.raises(ValueError, match='magno'):
            mg.MalpeliDensityMagnification(params, 'magno',
                                           cells_per_mm3=1e4)

    def test_rho_v_may_be_a_profile(self, lgn_params):
        """Packing density measures as eccentricity-dependent (1.5x
        parvo, 2.6x magno centre-to-periphery), so `cells_per_mm3` has
        to be expressible as a function of eccentricity and not only as
        a scalar."""
        def profile(ecc):
            return 20000.0 + 0.0 * np.asarray(ecc, float)

        model = mg.MalpeliDensityMagnification(lgn_params, 'parvo',
                                               cells_per_mm3=profile)
        flat = mg.MalpeliDensityMagnification(lgn_params, 'parvo',
                                              cells_per_mm3=20000.0)
        e = np.array([1.0, 10.0, 30.0])
        # A constant profile must reproduce the scalar exactly -- if it
        # does not, the profile path is not wired to the same arithmetic.
        assert np.allclose(model.volume_magnification_mm3_per_deg2(e),
                           flat.volume_magnification_mm3_per_deg2(e))

    def test_profile_rho_v_changes_the_magnification(self, lgn_params):
        """And a non-constant one has to actually bite, in the right
        direction: more cells per mm^3 means less tissue per degree."""
        def dense_centre(ecc):
            return np.where(np.asarray(ecc, float) < 5.0, 40000.0, 20000.0)

        model = mg.MalpeliDensityMagnification(lgn_params, 'parvo',
                                               cells_per_mm3=dense_centre)
        flat = mg.MalpeliDensityMagnification(lgn_params, 'parvo',
                                              cells_per_mm3=20000.0)
        _, central, _ = model.interval(np.array([1.0]))
        _, central_flat, _ = flat.interval(np.array([1.0]))
        _, outer, _ = model.interval(np.array([30.0]))
        _, outer_flat, _ = flat.interval(np.array([30.0]))
        assert central[0] == pytest.approx(
            central_flat[0] / np.sqrt(2.0), rel=1e-9)
        assert outer[0] == pytest.approx(outer_flat[0], rel=1e-9)


def test_from_atlas_can_build_an_eccentricity_resolved_rho_v(
        synthetic_atlas, lgn_params):
    """The profile has to be buildable from an atlas object, not only
    hand-supplied. Checked on the synthetic atlas so it runs without
    the 145 MB download -- this is a wiring test, not a biology one."""
    model = mg.MalpeliDensityMagnification.from_atlas(
        synthetic_atlas, lgn_params, cell_class='parvo',
        eccentricity_resolved=True)
    assert model.cells_per_mm3_profile is not None
    sampled = model.cells_per_mm3_at(np.array([2.0, 10.0, 30.0]))
    assert np.all(np.isfinite(sampled)) and np.all(sampled > 0)

    flat = mg.MalpeliDensityMagnification.from_atlas(
        synthetic_atlas, lgn_params, cell_class='parvo')
    assert flat.cells_per_mm3_profile is None


def test_rho_v_profile_does_not_extrapolate_to_zero(synthetic_atlas,
                                                    lgn_params):
    """rho_v falls monotonically with eccentricity, so extrapolating the
    trend past the last populated bin would run it towards zero and send
    the magnification to infinity. np.interp holds the endpoints flat;
    pinned because switching to a fitted extrapolation would break it
    silently and only far out in the periphery."""
    model = mg.MalpeliDensityMagnification.from_atlas(
        synthetic_atlas, lgn_params, cell_class='parvo',
        eccentricity_resolved=True)
    far = model.cells_per_mm3_at(np.array([500.0]))
    last = model.cells_per_mm3_profile.cells_per_mm3[-1]
    assert far[0] == pytest.approx(last, rel=1e-9)
    assert np.all(model.cells_per_mm3_at(np.array([0.0, 1e-6])) > 0)


class TestAtlasGradient:
    def test_from_jacobian_is_isotropic_by_construction(
            self, synthetic_atlas, synthetic_jacobian, lgn_params):
        jm = mg.JacobianMagnification(synthetic_jacobian, lgn_params,
                                      synthetic_atlas)
        scalar = mg.AtlasGradientMagnification.from_jacobian(jm)
        local = scalar.at_eccentricity(np.array([5.0, 15.0]))
        assert np.allclose(local.anisotropy, 1.0)
        assert np.all(local.isotropic_by_construction)


class TestBinReliability:
    """`_bin_reliability` in isolation, with hand-built grids -- this is
    the mechanism that is supposed to catch the real atlas's inclination
    seam once the table is fine enough to isolate it to a few bins."""

    def test_flags_low_count_bins(self):
        value_grid = np.full((2, 5), 10.0)
        n_grid = np.full((2, 5), 100.0)
        n_grid[0, 2] = 1.0
        reliable = mg._bin_reliability(value_grid, n_grid,
                                       min_voxels_per_bin=20,
                                       outlier_z_threshold=4.0)
        assert not reliable[0, 2]
        assert reliable.sum() == reliable.size - 1

    def test_flags_a_row_outlier(self):
        """A single bin far from its row's typical value -- the tear
        signature -- gets flagged even though it has plenty of voxels."""
        value_grid = np.full((1, 9), 10.0)
        value_grid[0, 4] = 100.0
        n_grid = np.full((1, 9), 100.0)
        reliable = mg._bin_reliability(value_grid, n_grid,
                                       min_voxels_per_bin=20,
                                       outlier_z_threshold=4.0)
        assert not reliable[0, 4]
        assert reliable[0].sum() == 8

    def test_does_not_flag_a_smooth_row(self):
        value_grid = np.linspace(1.0, 20.0, 10).reshape(1, 10)
        n_grid = np.full((1, 10), 100.0)
        reliable = mg._bin_reliability(value_grid, n_grid,
                                       min_voxels_per_bin=20,
                                       outlier_z_threshold=4.0)
        assert np.all(reliable)


class TestAnisotropicGradient:
    @pytest.fixture
    def model(self, synthetic_atlas, synthetic_jacobian, lgn_params):
        jm = mg.JacobianMagnification(synthetic_jacobian, lgn_params,
                                      synthetic_atlas)
        return mg.AnisotropicGradientMagnification.from_jacobian(
            jm, params=lgn_params)

    def test_requires_inclination(self, model):
        with pytest.raises(ValueError, match='inclination_deg'):
            model.at_eccentricity(np.array([15.0]))

    def test_recovers_the_analytic_magnifications(self, model,
                                                   synthetic_atlas):
        ecc = 15.0
        local = model.at_eccentricity(np.array([ecc]),
                                      inclination_deg=np.array([0.0]))
        fitted = np.sort([local.major_deg_per_mm[0], local.minor_deg_per_mm[0]])
        analytic = np.sort(np.asarray(
            synthetic_atlas.analytic_magnification_deg_per_mm(ecc),
            dtype=float))
        # Looser than the per-voxel Jacobian check: the table adds
        # spatial binning on top of the per-voxel fit's own noise.
        assert np.allclose(fitted, analytic, rtol=0.35)

    def test_central_bins_are_isotropic_by_construction(self, model):
        """Inside `atlas.isotropic_construction_radius_deg`, Erwin et
        al. assigned eccentricity from an isotropic formula -- any
        anisotropy the fit reports there is an artefact of that
        construction, so orientation must read as the flagged zero, not
        as a measurement."""
        local = model.at_eccentricity(np.array([0.3]),
                                      inclination_deg=np.array([5.0]))
        assert local.isotropic_by_construction[0]
        assert local.orientation_rad[0] == 0.0

    def test_anisotropic_beyond_the_central_radius(self, model):
        """A sanity check that the table is not isotropic everywhere --
        otherwise `test_central_bins_are_isotropic_by_construction` would
        pass vacuously."""
        local = model.at_eccentricity(np.array([15.0]),
                                      inclination_deg=np.array([0.0]))
        assert not local.isotropic_by_construction[0]
        assert local.anisotropy[0] > 1.05

    def test_at_voxels_matches_at_eccentricity(self, model, synthetic_atlas):
        idx = np.argwhere(synthetic_atlas.valid)
        sample = idx[np.linspace(0, len(idx) - 1, 20, dtype=int)]
        sel = tuple(sample.T)
        from_voxels = model.at_voxels(sample, erwin_atlas=synthetic_atlas)
        from_ecc = model.at_eccentricity(
            synthetic_atlas.eccentricity_deg[sel],
            inclination_deg=synthetic_atlas.inclination_deg[sel])
        assert np.allclose(from_voxels.major_deg_per_mm,
                           from_ecc.major_deg_per_mm, equal_nan=True)

    def test_unreliable_bin_does_not_leak_into_its_neighbours(self):
        """The whole point of a fine grid: an isolated bad bin (standing
        in for the real atlas's inclination seam) must not drag a
        neighbouring query's value toward it, even though bilinear
        interpolation would ordinarily blend across the shared edge."""
        ecc_edges = np.array([0., 10., 20., 30.])
        incl_edges = np.array([-20., -10., 0., 10., 20.])
        major_grid = np.full((3, 4), 10.0)
        minor_grid = np.full((3, 4), 5.0)
        minor_grid[1, 1] = 0.5  # the "torn" bin: ecc in [10,20), incl in [-10,0)
        orientation_grid = np.zeros((3, 4))
        n_grid = np.full((3, 4), 100.0)
        reliable_grid = np.ones((3, 4), dtype=bool)
        reliable_grid[1, 1] = False
        isotropic_grid = np.ones((3, 4), dtype=bool)

        model = mg.AnisotropicGradientMagnification(
            ecc_edges, incl_edges, major_grid, minor_grid, orientation_grid,
            n_grid, reliable_grid, isotropic_grid,
            orientation_reliable_min_anisotropy=1.0)

        # Deep inside the bad bin: routed to the nearest reliable bin
        # instead of returning the corrupted 0.5, and flagged.
        at_bad = model.at_eccentricity(np.array([15.0]),
                                       inclination_deg=np.array([-5.0]))
        assert at_bad.minor_deg_per_mm[0] == pytest.approx(5.0)
        assert at_bad.unreliable_beyond_neighborhood[0]

        # Straddling the bad bin's edge: bilinear would ordinarily blend
        # 0.5 into the result, but the corrupted corner must be dropped
        # and the remaining weight renormalised over the clean one(s).
        at_edge = model.at_eccentricity(np.array([15.0]),
                                        inclination_deg=np.array([0.0]))
        assert at_edge.minor_deg_per_mm[0] == pytest.approx(5.0, rel=1e-6)
        assert at_edge.unreliable_beyond_neighborhood[0]

        # Far from the bad bin: clean bilinear, not flagged.
        far = model.at_eccentricity(np.array([5.0]),
                                    inclination_deg=np.array([15.0]))
        assert far.minor_deg_per_mm[0] == pytest.approx(5.0)
        assert not far.unreliable_beyond_neighborhood[0]


@requires_real_atlas
class TestRealAtlasCellDistribution:
    """Properties of the real atlas that the density route depends on.

    These are characterisation tests of fixed input data, not of code.
    They exist because the model makes an assumption about the atlas
    (uniform packing density) that the atlas does not satisfy, and
    because the one existing whole-nucleus check cannot see that.
    """

    @staticmethod
    def _profile(atlas, params, cell_class, edges):
        mask = (atlas.valid
                & np.isin(atlas.layer,
                          resolve_class_codes(params, cell_class))
                & ~atlas.is_ipsi_sentinel())
        sel = tuple(np.argwhere(mask).T)
        ecc = atlas.eccentricity_deg[sel]
        cells = atlas.cells_per_voxel[sel]
        means = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = (ecc >= lo) & (ecc < hi)
            if in_bin.sum() < 50:
                means.append(np.nan)
                continue
            means.append(float(cells[in_bin].mean()))
        return np.array(means), float(cells.mean())

    @pytest.mark.parametrize('cell_class, min_spread',
                             [('parvo', 1.3), ('magno', 2.0)])
    def test_packing_density_is_not_uniform(self, real_atlas, lgn_params,
                                            cell_class, min_spread):
        """`from_atlas` defaults to ONE rho_v for the whole nucleus.
        Cells per voxel actually spans 1.5x (parvo) to 2.6x (magno) from
        centre to periphery, so that default is wrong by up to 2x where
        it is worst. Pinned so the assumption cannot be reintroduced as
        a simplification.
        """
        edges = np.array([1., 2., 4., 7., 10., 15., 20., 30., 45., 60., 90.])
        means, _ = self._profile(real_atlas, lgn_params, cell_class, edges)
        finite = means[np.isfinite(means)]
        assert finite.max() / finite.min() > min_spread

    @pytest.mark.parametrize('cell_class', ['parvo', 'magno'])
    def test_packing_density_falls_monotonically(self, real_atlas,
                                                 lgn_params, cell_class):
        """It is a gradient, not scatter -- which is what makes a
        tabulated rho_v(E) the right fix rather than a wider interval."""
        edges = np.array([1., 2., 4., 7., 10., 15., 20., 30., 45., 60., 90.])
        means, _ = self._profile(real_atlas, lgn_params, cell_class, edges)
        finite = means[np.isfinite(means)]
        assert np.all(np.diff(finite) < 0)

    def test_totals_agree_while_the_distribution_does_not(self, real_atlas,
                                                          lgn_params):
        """Why the whole-nucleus check is weaker than it looks.

        `test_hemifield_integral_reproduces_atlas_cell_counts` compares
        integrated totals and passes. Resolved by eccentricity, the same
        comparison is off by more than 20% in the central bins -- the
        errors compensate, and only a per-bin test can see that.
        """
        deg2_per_sr = (180.0 / np.pi) ** 2
        edges = np.array([1., 2., 4., 7., 10., 15., 20., 30., 45., 60., 90.])
        mask = (real_atlas.valid
                & np.isin(real_atlas.layer,
                          resolve_class_codes(lgn_params, 'parvo'))
                & ~real_atlas.is_ipsi_sentinel())
        sel = tuple(np.argwhere(mask).T)
        ecc = real_atlas.eccentricity_deg[sel]
        cells = real_atlas.cells_per_voxel[sel]

        ratios, total_predicted, total_atlas = [], 0.0, 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            e = np.linspace(lo, hi, 4001)
            d_omega = (np.pi * np.sin(np.deg2rad(e))
                       * np.deg2rad(e[1] - e[0]) * deg2_per_sr)
            predicted = float(np.sum(
                mg.malpeli_density(e, lgn_params, 'parvo') * d_omega))
            in_bin = (ecc >= lo) & (ecc < hi)
            observed = float(cells[in_bin].sum())
            ratios.append(observed / predicted)
            total_predicted += predicted
            total_atlas += observed

        ratios = np.array(ratios)
        total_ratio = total_atlas / total_predicted
        # The structural claim, stated as a comparison rather than as a
        # tolerance: the pooled total sits much closer to 1 than the
        # worst individual bin does. An absolute tolerance on the total
        # would be a claim about the integration window instead (this
        # one excludes the central 1 deg and everything past 90 deg).
        assert abs(total_ratio - 1.0) < 0.5 * np.max(np.abs(ratios - 1.0))
        # Resolved by eccentricity they do not agree: the central bins
        # run ~25% short and the mid-periphery ~15% long, and those
        # compensate in the total.
        assert ratios[0] < 0.85, (
            f"central 1-2 deg parvo ratio is {ratios[0]:.3f}; the "
            f"documented shortfall is ~0.74. If this now passes, the "
            f"atlas or the loader changed.")
        assert ratios.max() > 1.10
