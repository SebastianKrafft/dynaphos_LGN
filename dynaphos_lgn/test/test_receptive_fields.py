"""Receptive-field size models, and the cross-species caveats they carry."""
from pathlib import Path

import numpy as np
import pytest

from dynaphos_lgn import receptive_fields as rf

CRONER_CSV = (Path(__file__).resolve().parents[1] / 'research'
              / 'croner_kaplan_fig4_extracted.csv')


class TestClassOrdering:
    @pytest.mark.parametrize('ecc', [5.0, 10.0, 20.0, 30.0])
    def test_konio_larger_than_magno_larger_than_parvo(self, lgn_params, ecc):
        """The one robust, cross-species-replicated finding: K > M > P
        in centre size at matched eccentricity. Note the magnitudes mix
        a real class difference with a species difference, since the K
        numbers are owl monkey and M/P are macaque."""
        sizes = rf.class_ordering_check(lgn_params, ecc)
        assert sizes['konio'] > sizes['magno'] > sizes['parvo']

    @pytest.mark.parametrize('ecc', [0.0, 5.0, 20.0, 35.0])
    def test_magno_is_roughly_twice_parvo(self, lgn_params, ecc):
        """Croner & Kaplan state M ~ 2x P. The in-project fit should
        bracket that rather than contradict it."""
        ratio = (float(rf.center_radius_deg(ecc, lgn_params, 'magno'))
                 / float(rf.center_radius_deg(ecc, lgn_params, 'parvo')))
        assert 1.6 < ratio < 3.2

    def test_the_m_over_p_ratio_is_not_constant_with_eccentricity(self, lgn_params):
        """Worth pinning down rather than glossing: because the two
        exponential fits have different rate constants, the M/P centre
        ratio is not one number. It runs ~2.97x at the fovea down to
        ~1.68x at 35 deg, so Croner & Kaplan's "about 2x" is a
        mid-range summary, and quoting a single ratio at an unstated
        eccentricity is a ~1.8x ambiguity."""
        ratios = {e: float(rf.center_radius_deg(e, lgn_params, 'magno'))
                  / float(rf.center_radius_deg(e, lgn_params, 'parvo'))
                  for e in (0.0, 35.0)}
        assert ratios[0.0] == pytest.approx(2.97, abs=0.05)
        assert ratios[35.0] == pytest.approx(1.68, abs=0.05)
        assert ratios[0.0] > ratios[35.0]


class TestSizeVsEccentricity:
    @pytest.mark.parametrize('cls', ['magno', 'parvo', 'konio'])
    def test_grows_with_eccentricity(self, lgn_params, cls):
        e = np.linspace(1.0, 35.0, 40)
        rc = rf.center_radius_deg(e, lgn_params, cls)
        assert np.all(np.diff(rc) >= 0)

    def test_konio_linear_fit_is_floored_not_negative(self, lgn_params):
        """Xu et al.'s linear rc(E) goes negative below ~4.9 deg. The
        same paper reports a measured 0.31 deg mean below 15 deg, so the
        floor is theirs, not invented."""
        assert float(rf.center_radius_deg(0.0, lgn_params, 'konio')) == pytest.approx(
            lgn_params['receptive_fields']['xu_konio']['rc_below_15deg'][0])
        assert np.all(rf.center_radius_deg(np.linspace(0, 20, 50),
                                           lgn_params, 'konio') > 0)

    def test_power_and_exponential_forms_differ_at_the_fovea(self, lgn_params):
        """The functional-form choice is not cosmetic: the power law
        goes to zero at the fovea, the exponential stays finite."""
        expo = float(rf.center_radius_deg(0.01, lgn_params, 'parvo', 'exponential'))
        power = float(rf.center_radius_deg(0.01, lgn_params, 'parvo', 'power'))
        assert power < 0.25 * expo

    def test_unknown_class_and_form_are_rejected(self, lgn_params):
        with pytest.raises(ValueError):
            rf.center_radius_deg(10.0, lgn_params, 'koniocellularish')
        with pytest.raises(ValueError):
            rf.center_radius_deg(10.0, lgn_params, 'parvo', form='spline')


class TestDifferenceOfGaussians:
    def test_profile_is_centre_positive_and_surround_negative(self, lgn_params):
        r = rf.receptive_field(10.0, lgn_params, 'parvo')
        assert float(r.profile(0.0)) > 0.9
        assert float(r.profile(r.center_radius_deg * 3)) < 0
        assert float(r.profile(r.surround_radius_deg * 6)) == pytest.approx(
            0.0, abs=1e-3)

    def test_volume_and_peak_ratios_are_not_the_same_number(self, lgn_params):
        """The trap this class exists to prevent: 0.55 is an INTEGRATED
        ratio, and using it as Ks/Kc makes the DoG negative almost
        everywhere."""
        r = rf.receptive_field(10.0, lgn_params, 'parvo')
        assert r.surround_center_volume_ratio == pytest.approx(0.55)
        assert float(r.surround_center_peak_ratio) < 0.05

    def test_peak_ratio_reconstructs_the_volume_ratio(self, lgn_params):
        r = rf.receptive_field(15.0, lgn_params, 'magno')
        recovered = (float(r.surround_center_peak_ratio)
                     * float(r.surround_center_ratio) ** 2)
        assert recovered == pytest.approx(r.surround_center_volume_ratio)

    def test_sigma_and_1_over_e_radius_differ_by_root_two(self, lgn_params):
        """Croner & Kaplan's rc is a 1/e radius, not a standard
        deviation. Collapsing the two is a factor of 1.41 in every
        derived size."""
        r = rf.receptive_field(10.0, lgn_params, 'parvo')
        assert (float(r.center_radius_deg)
                == pytest.approx(float(r.center_sigma_deg) * np.sqrt(2)))

    def test_konio_uses_its_own_surround_fit(self, lgn_params):
        """Xu et al. publish a separate rs(E); the M/P path instead
        multiplies rc by a fixed ratio, which is a flagged assumption."""
        e = 20.0
        k = rf.receptive_field(e, lgn_params, 'konio')
        xu = lgn_params['receptive_fields']['xu_konio']
        expected = xu['rs_slope_per_deg'] * e + xu['rs_intercept_deg']
        assert float(k.surround_radius_deg) == pytest.approx(expected)


@pytest.mark.skipif(not CRONER_CSV.exists(),
                    reason='digitised Croner & Kaplan points not present')
class TestRefit:
    def test_refit_reproduces_the_configured_coefficients(self, lgn_params):
        """The coefficients in the config are a fit, not a citation, so
        the fit has to stay reproducible from the data it came from."""
        fitted = rf.fit_croner_kaplan(CRONER_CSV)
        stored_all = lgn_params['receptive_fields']['croner_kaplan'][
            'exponential']
        for cls, stored in stored_all.items():
            assert fitted[cls]['a_deg'] == pytest.approx(stored['a_deg'],
                                                         rel=1e-4)
            assert fitted[cls]['b_per_deg'] == pytest.approx(
                stored['b_per_deg'], rel=1e-4)
            assert fitted[cls]['n'] == stored['n']

    def test_high_confidence_only_shrinks_the_parvo_sample(self, lgn_params):
        """Excluding the cluster-split points is a real bias/variance
        trade-off, not a free improvement -- 80 P points become 16."""
        strict = rf.fit_croner_kaplan(CRONER_CSV, confidence='high')
        assert strict['parvo']['n'] < lgn_params['receptive_fields'][
            'croner_kaplan']['exponential']['parvo']['n']

    def test_power_form_refits_to_the_stored_power_coefficients(self, lgn_params):
        fitted = rf.fit_croner_kaplan(CRONER_CSV, form='power')
        stored_all = lgn_params['receptive_fields']['croner_kaplan']['power']
        for cls, stored in stored_all.items():
            assert fitted[cls]['a_deg'] == pytest.approx(stored['a_deg'],
                                                         rel=1e-4)
            assert fitted[cls]['b'] == pytest.approx(stored['b'], rel=1e-4)
