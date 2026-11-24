"""Receptive-field size models."""
import numpy as np
import pytest

from dynaphos_lgn import receptive_fields as rf


class TestClassOrdering:
    @pytest.mark.parametrize('ecc', [0.0, 5.0, 20.0, 35.0])
    def test_magno_is_roughly_twice_parvo(self, lgn_params, ecc):
        """Croner & Kaplan state M ~ 2x P. The in-project fit should
        bracket that rather than contradict it."""
        ratio = (float(rf.center_radius_deg(ecc, lgn_params, 'magno'))
                 / float(rf.center_radius_deg(ecc, lgn_params, 'parvo')))
        assert 1.6 < ratio < 3.2

    def test_the_m_over_p_ratio_is_not_constant_with_eccentricity(self, lgn_params):
        """Worth pinning down rather than glossing: because the two
        exponential fits have different rate constants, the M/P center
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
    @pytest.mark.parametrize('cls', ['magno', 'parvo'])
    def test_grows_with_eccentricity(self, lgn_params, cls):
        e = np.linspace(1.0, 35.0, 40)
        rc = rf.center_radius_deg(e, lgn_params, cls)
        assert np.all(np.diff(rc) >= 0)

    def test_power_and_exponential_forms_differ_at_the_fovea(self, lgn_params):
        """The functional-form choice is not cosmetic: the power law
        goes to zero at the fovea, the exponential stays finite."""
        expo = float(rf.center_radius_deg(0.01, lgn_params, 'parvo', 'exponential'))
        power = float(rf.center_radius_deg(0.01, lgn_params, 'parvo', 'power'))
        assert power < 0.25 * expo

    def test_per_element_class_matches_the_single_class_radius(self,
                                                              lgn_params):
        e = np.array([2.0, 10.0, 20.0])
        classes = np.array(['magno', 'parvo', 'magno'])
        rc = rf.center_radius_by_class(e, lgn_params, classes)
        for i, name in enumerate(classes):
            assert rc[i] == pytest.approx(
                float(rf.center_radius_deg(e[i], lgn_params, name)))

    def test_unknown_class_and_form_are_rejected(self, lgn_params):
        with pytest.raises(ValueError):
            rf.center_radius_deg(10.0, lgn_params, 'koniocellularish')
        with pytest.raises(ValueError):
            rf.center_radius_deg(10.0, lgn_params, 'parvo', form='spline')


class TestDifferenceOfGaussians:
    def test_profile_is_center_positive_and_surround_negative(self, lgn_params):
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
