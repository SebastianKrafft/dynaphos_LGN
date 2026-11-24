"""Magnification models, checked against published numbers and against
an analytically-known synthetic map."""
from types import SimpleNamespace

import numpy as np
import pytest

from dynaphos_lgn import magnification as mg
from dynaphos_lgn.params import resolve_class_codes

from .conftest import requires_real_atlas


class TestMalpeliFormulas:
    """Eqs 3-4 of Malpeli, Lee & Baker (1996), verified full text."""

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
        """Malpeli's density integrated over one hemifield reproduces the
        atlas's published cell counts."""
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
        # eccentricity quantized to 0.1 deg and inclination to 1 deg, so
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


@requires_real_atlas
class TestRealAtlasCellDistribution:
    """Properties of the real atlas that the density route depends on.

    These are characterization tests of fixed input data, not of code.
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
        center to periphery, so that default is wrong by up to 2x where
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


def test_eccentricity_table_averages_orientation_as_an_axis(
        lgn_params, monkeypatch):
    """Axes at +-89 deg are both nearly vertical; their median is 0."""
    orientation = np.deg2rad([89.0, -89.0, 89.0, -89.0])
    monkeypatch.setattr(mg.JacobianMagnification, '_table_voxels',
                        lambda model, cell_class: (
        np.arange(4), np.ones(4), np.ones(4), orientation))
    atlas = SimpleNamespace(eccentricity_deg=np.array([1.0, 1.2, 3.0, 3.2]))
    model = mg.JacobianMagnification(None, lgn_params, atlas)
    local = model.at_eccentricity(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(np.abs(local.orientation_rad), np.pi / 2)
