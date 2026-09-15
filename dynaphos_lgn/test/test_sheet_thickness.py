"""The distance-transform thickness estimator, against geometries whose
answer is known by construction.

Two things are pinned here. First the arithmetic: the exact discrete
inverse, not the continuous `4 * mean`, which is off by about a voxel and
therefore by 10-20% on laminae only 5-9 voxels thick. Second, and more
important, the estimator's *failure mode*: scattered holes destroy it
silently, returning a plausible-looking small number rather than an
error. The report that uses it checks the precondition first, so the
test suite pins that the precondition actually matters.
"""
import numpy as np
import pytest

ndi = pytest.importorskip('scipy.ndimage')

from dynaphos_lgn.check_sheet_thickness_morphology import (  # noqa: E402
    sheet_quality_report, thickness_from_mean_edt)


def _measure(mask):
    return thickness_from_mean_edt(
        ndi.distance_transform_edt(mask)[mask].mean())


def _flat_slab(thickness, size=60):
    mask = np.zeros((size, size, size), bool)
    lo = size // 2 - thickness // 2
    mask[:, lo:lo + thickness, :] = True
    return mask


def _wide_flat_slab(thickness, span=200):
    """A slab whose in-plane extent dwarfs its thickness.

    `surface_frac` only approaches 2/T when the slab's edges are
    negligible next to its two faces; at span=60 the edge contribution is
    already ~40% of the surface for T=15, which is a property of the test
    fixture and not of the estimator.
    """
    mask = np.zeros((span, thickness + 20, span), bool)
    lo = 10
    mask[:, lo:lo + thickness, :] = True
    return mask


def _curved_shell(thickness, radius=35, size=120):
    _, yy, xx = np.indices((size, size, size))
    r = np.hypot(yy - size / 2, xx - size / 2)
    return (r >= radius) & (r < radius + thickness) & (xx > size / 2)


class TestThicknessFromMeanEdt:
    @pytest.mark.parametrize('thickness', [5, 7, 9, 11, 15, 21, 33])
    def test_exact_on_a_flat_slab(self, thickness):
        """The whole point of the exact inverse: no bias at all on the
        geometry it is derived for."""
        assert _measure(_flat_slab(thickness)) == pytest.approx(
            thickness, abs=1e-6)

    @pytest.mark.parametrize('thickness', [5, 9, 15])
    def test_the_naive_continuous_formula_is_biased(self, thickness):
        """`4 * mean(EDT)` is what the continuous derivation gives and
        it overestimates by about one voxel. Pinned so nobody
        'simplifies' the exact inverse back into it."""
        mask = _flat_slab(thickness)
        naive = 4.0 * ndi.distance_transform_edt(mask)[mask].mean()
        assert naive > thickness + 1.5
        assert _measure(mask) == pytest.approx(thickness, abs=1e-6)

    @pytest.mark.parametrize('thickness', [5, 9, 15])
    def test_curved_shells_are_within_ten_percent(self, thickness):
        """LGN layers are curved sheets, so this is the geometry that
        matters. The bias is negative (a curved sheet's outer face is
        longer than its inner one) and about 8%, which is the value
        `CURVATURE_BIAS` corrects for in the report."""
        got = _measure(_curved_shell(thickness))
        assert 0.85 * thickness < got < thickness

    def test_degenerate_input_does_not_return_a_complex_number(self):
        """mean EDT below 1 is not physically reachable for a non-empty
        mask, but the sqrt would go imaginary if it were."""
        assert thickness_from_mean_edt(0.4) == pytest.approx(0.0, abs=0.5)
        assert np.isreal(thickness_from_mean_edt(0.0))

    def test_monotone_in_mean_edt(self):
        values = [thickness_from_mean_edt(m)
                  for m in np.linspace(1.0, 10.0, 40)]
        assert np.all(np.diff(values) > 0)


class TestTheFailureMode:
    @pytest.mark.parametrize('hole_frac', [0.02, 0.05, 0.10])
    def test_scattered_holes_collapse_the_estimate(self, hole_frac):
        """The reason the report checks a precondition instead of just
        running the estimator.

        Every hole manufactures a new boundary, so the mean distance to
        boundary collapses towards 1 regardless of the true thickness --
        and it does so *quietly*, returning a plausible small number.
        """
        rng = np.random.default_rng(0)
        mask = _curved_shell(9)
        holed = mask & (rng.random(mask.shape) > hole_frac)
        assert _measure(holed) < 0.7 * _measure(mask)

    def test_component_count_does_not_detect_raggedness(self):
        """Pinned because it is the natural thing to reach for and it
        does not work: a 9-voxel shell with 5% of its voxels removed at
        random is still a single connected component under
        6-connectivity, while the thickness estimate has already
        collapsed by more than half. A precondition built on component
        count would pass exactly the input that breaks the estimator.
        """
        rng = np.random.default_rng(0)
        clean = _curved_shell(9)
        holed = clean & (rng.random(clean.shape) > 0.05)
        assert sheet_quality_report(holed, ndi)['components'] == 1
        assert _measure(holed) < 0.5 * _measure(clean)

    def test_enclosed_holes_do_detect_raggedness(self):
        rng = np.random.default_rng(0)
        clean = _curved_shell(9)
        clean_q = sheet_quality_report(clean, ndi)
        assert clean_q['enclosed_hole_frac'] == pytest.approx(0.0, abs=1e-9)
        for hole_frac in (0.01, 0.05):
            holed = clean & (rng.random(clean.shape) > hole_frac)
            assert sheet_quality_report(
                holed, ndi)['enclosed_hole_frac'] > 0.5 * hole_frac

    def test_surface_times_thickness_is_a_weak_secondary_check(self):
        """`surface_frac * T` lands near 2 on a clean sheet and FALLS
        with raggedness -- the thickness estimate collapses faster than
        the surface fraction rises.

        Pinned with the direction explicit because it is easy to assume
        the opposite, and because the check is weak: it moves 1.9 -> 1.6
        at 1% holes and then saturates around 1.45, so it corroborates
        the enclosed-hole test rather than replacing it.
        """
        rng = np.random.default_rng(0)
        clean = _curved_shell(9)
        clean_product = (sheet_quality_report(clean, ndi)['surface_frac']
                         * _measure(clean))
        assert clean_product == pytest.approx(2.0, rel=0.25)

        previous = clean_product
        for hole_frac in (0.01, 0.05):
            holed = clean & (rng.random(clean.shape) > hole_frac)
            product = (sheet_quality_report(holed, ndi)['surface_frac']
                       * _measure(holed))
            assert product < previous
            previous = product
        assert previous < 0.85 * clean_product


class TestSheetQualityReport:
    @pytest.mark.parametrize('thickness', [5, 9, 15])
    def test_surface_fraction_tracks_two_over_thickness(self, thickness):
        """A clean slab of T voxels has ~2/T of its voxels on a face.
        This is the cheap sanity number quoted alongside the atlas's own
        sheets -- but only once the slab's edges are negligible, hence
        the wide fixture."""
        q = sheet_quality_report(_wide_flat_slab(thickness), ndi)
        assert q['surface_frac'] == pytest.approx(2.0 / thickness, rel=0.2)

    def test_empty_mask_is_reported_not_raised(self):
        q = sheet_quality_report(np.zeros((8, 8, 8), bool), ndi)
        assert q['n'] == 0
        assert not np.isfinite(q['surface_frac'])
