"""The isoeccentricity/isopolar geometry, against analytically-known
Jacobians.

The sign of `M_isoecc / M_isopolar` is the single easiest thing to get
backwards in this whole comparison -- "magnification greater along
isoeccentricity" means *fewer* deg/mm along isoeccentricity, so the
ratio exceeds 1 exactly where the Jacobian's major (deg/mm) axis points
*radially*. These tests pin that down with Jacobians whose answer can be
worked out by hand, so a future refactor cannot quietly flip it.
"""
import numpy as np
import pytest

from dynaphos_lgn.check_anisotropy_local_frame import (
    alignment_efficiency, contour_magnification_ratio, geometric_summary,
    meridian_class)


def _jacobian(sx, sy):
    """A 2x3 Jacobian that stretches visual x by sx and y by sy deg/mm.

    The third tissue column is zero: LGN layers are thin curved sheets,
    so one physical direction carries no retinotopic gradient -- the same
    assumption `JacobianMagnification` documents.
    """
    return np.array([[[sx, 0.0, 0.0],
                      [0.0, sy, 0.0]]], dtype=float)


class TestContourMagnificationRatio:
    def test_isotropic_map_gives_unity_everywhere(self):
        """No anisotropy, no directional ratio, at any polar angle."""
        for inclination in (0.0, 30.0, 90.0, 143.0, -60.0):
            got = contour_magnification_ratio(_jacobian(3.0, 3.0),
                                              np.array([inclination]))
            assert float(got[0]) == pytest.approx(1.0, rel=1e-12)

    def test_on_the_horizontal_meridian_the_ratio_is_sx_over_sy(self):
        """At I = 0 the radial direction is x and tangential is y, so
        M_isopolar = 1/sx, M_isoecc = 1/sy, ratio = sx/sy."""
        got = contour_magnification_ratio(_jacobian(2.0, 1.0),
                                          np.array([0.0]))
        assert float(got[0]) == pytest.approx(2.0, rel=1e-12)

    def test_on_the_vertical_meridian_the_ratio_inverts(self):
        """At I = 90 the roles of x and y swap, so the same Jacobian
        must give the reciprocal."""
        got = contour_magnification_ratio(_jacobian(2.0, 1.0),
                                          np.array([90.0]))
        assert float(got[0]) == pytest.approx(0.5, rel=1e-12)

    def test_ratio_above_one_means_the_major_axis_is_radial(self):
        """The sign convention, stated as the assertion it is.

        sx > sy puts the major deg/mm axis along x; at I = 0 that is the
        radial (isopolar) direction; and C&VE's layer-6 orientation --
        more tissue per degree along isoeccentricity -- is the ratio
        exceeding 1. All three must agree.
        """
        radial_major = contour_magnification_ratio(_jacobian(4.0, 1.0),
                                                   np.array([0.0]))
        tangential_major = contour_magnification_ratio(_jacobian(1.0, 4.0),
                                                       np.array([0.0]))
        assert float(radial_major[0]) > 1.0
        assert float(tangential_major[0]) < 1.0
        assert float(radial_major[0]) == pytest.approx(
            1.0 / float(tangential_major[0]), rel=1e-12)

    def test_at_45_degrees_an_aligned_ellipse_shows_no_ratio(self):
        """An ellipse at 45 deg to both contours has no *directional*
        magnification difference, however anisotropic it is. This is
        what `alignment_efficiency` exists to separate out."""
        got = contour_magnification_ratio(_jacobian(5.0, 1.0),
                                          np.array([45.0]))
        assert float(got[0]) == pytest.approx(1.0, rel=1e-12)

    def test_ratio_never_exceeds_the_anisotropy(self):
        """The anisotropy is the ceiling: no direction can differ by
        more than the ratio of the two principal magnifications."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            sx, sy = rng.uniform(0.5, 8.0, size=2)
            inclination = rng.uniform(-180, 180)
            ratio = float(contour_magnification_ratio(
                _jacobian(sx, sy), np.array([inclination]))[0])
            anisotropy = max(sx, sy) / min(sx, sy)
            assert 1.0 / anisotropy - 1e-9 <= ratio <= anisotropy + 1e-9

    def test_degenerate_jacobian_is_nan_not_a_number(self):
        """A rank-deficient local map has no well-defined ratio, and
        silently returning one would be worse than a gap in the table."""
        got = contour_magnification_ratio(_jacobian(2.0, 0.0),
                                          np.array([0.0]))
        assert not np.isfinite(got[0])

    def test_nan_jacobian_propagates(self):
        jac = _jacobian(2.0, 1.0)
        jac[0, 0, 0] = np.nan
        assert not np.isfinite(
            contour_magnification_ratio(jac, np.array([0.0]))[0])


class TestAlignmentEfficiency:
    def test_fully_radial_major_axis_scores_plus_one(self):
        ratio = contour_magnification_ratio(_jacobian(4.0, 1.0),
                                            np.array([0.0]))
        eff = alignment_efficiency(ratio, np.array([4.0]))
        assert float(eff[0]) == pytest.approx(1.0, rel=1e-9)

    def test_fully_tangential_major_axis_scores_minus_one(self):
        ratio = contour_magnification_ratio(_jacobian(1.0, 4.0),
                                            np.array([0.0]))
        eff = alignment_efficiency(ratio, np.array([4.0]))
        assert float(eff[0]) == pytest.approx(-1.0, rel=1e-9)

    def test_diagonal_ellipse_scores_zero(self):
        ratio = contour_magnification_ratio(_jacobian(4.0, 1.0),
                                            np.array([45.0]))
        eff = alignment_efficiency(ratio, np.array([4.0]))
        assert float(eff[0]) == pytest.approx(0.0, abs=1e-9)

    def test_isotropic_voxels_are_excluded_not_scored_zero(self):
        """An isotropic voxel has no axis to align, so the statistic is
        undefined there. Scoring it 0 would drag every pooled median
        towards 0 and read as 'axes point diagonally'."""
        eff = alignment_efficiency(np.array([1.0]), np.array([1.0]))
        assert not np.isfinite(eff[0])


class TestHelpers:
    @pytest.mark.parametrize('inclination, expected', [
        (0.0, 'horizontal'), (180.0, 'horizontal'), (-175.0, 'horizontal'),
        (90.0, 'vertical'), (-90.0, 'vertical'), (45.0, 'oblique'),
        (135.0, 'oblique')])
    def test_meridian_class(self, inclination, expected):
        got = meridian_class(np.array([inclination]))
        assert got[0] == expected

    def test_geometric_summary_treats_reciprocals_symmetrically(self):
        """2x and 0.5x are equal and opposite departures from 1; an
        arithmetic mean would report 1.25 and invent an effect."""
        values = np.concatenate([np.full(50, 2.0), np.full(50, 0.5)])
        median, _, _ = geometric_summary(values)
        assert median == pytest.approx(1.0, rel=1e-9)

    def test_geometric_summary_refuses_tiny_samples(self):
        median, lo, hi = geometric_summary(np.array([1.0, 2.0, 3.0]))
        assert not np.isfinite(median)
