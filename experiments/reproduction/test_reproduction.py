"""The Dynaphos paper's calibration fits, reproduced with this repo's
simulators (see reproduction/README.md)."""
import numpy as np
import pytest
import scipy.signal

from experiments.reproduction import brightness, temporal_dynamics, thresholding
from experiments.reproduction.common import (PAPER, fit_charge_duration, fit_probit,
                                 load_config, probit, r2_pearson,
                                 run_protocol, make_simulator,
                                 train_amplitudes)

pytestmark = pytest.mark.slow


def test_charge_duration_fit():
    _, thresholds = thresholding.load_data()
    rheobase, _, r2 = fit_charge_duration(thresholds.corrected_duration,
                                          thresholds.charge_per_trial)
    assert rheobase == pytest.approx(PAPER['rheobase'], rel=1e-3)
    assert r2 == pytest.approx(PAPER['charge_duration_r2'], abs=1e-3)


def test_psychometric_fit_and_cross_validation():
    proportions, thresholds = thresholding.load_data()
    activation = thresholding.max_activation('v1', proportions)
    observed = proportions.proportion_phosphene_perception
    mu, sigma = fit_probit(activation, observed)
    assert mu == pytest.approx(PAPER['activation_threshold'], rel=2e-3)
    assert sigma == pytest.approx(PAPER['activation_threshold_sd'], rel=2e-3)
    assert r2_pearson(observed, probit(activation, mu, sigma)) == \
        pytest.approx(PAPER['psychometric_r2'], abs=1e-3)
    pooled, _, _ = thresholding.cross_validate(proportions, thresholds,
                                               activation)
    assert pooled == pytest.approx(PAPER['loocv_r2'], abs=1e-3)


def test_brightness_fit():
    res = brightness.evaluate('v1', brightness.load_data())
    assert res['brightness_r2'] == pytest.approx(PAPER['brightness_r2'],
                                                 abs=1e-4)
    # The paper fitted with older trace parameters; 1-2% apart
    assert res['slope_brightness'] == pytest.approx(
        PAPER['slope_brightness'], rel=0.02)
    assert res['cps_half'] == pytest.approx(PAPER['cps_half'], rel=0.02)


def test_lgn_temporal_model_matches_v1():
    """Same activation, trace and brightness for the same input."""
    amplitudes = np.linspace(10e-6, 100e-6, 10)
    stim = train_amplitudes(amplitudes, 0.166, 500, 300)
    keys = ('activation', 'trace', 'brightness')
    runs = {kind: run_protocol(make_simulator(kind, len(amplitudes), 500),
                               stim, 170e-6, 300, traces=keys)
            for kind in ('v1', 'lgn_hard', 'lgn_soft')}
    for kind in ('lgn_hard', 'lgn_soft'):
        for key in keys:
            np.testing.assert_allclose(runs[kind][key], runs['v1'][key],
                                       rtol=1e-6, atol=1e-15)


def test_fast_temporal_evaluator_matches_simulator():
    params = load_config('v1')
    td = params['temporal_dynamics']
    fast = temporal_dynamics.fast_peaks(td['trace_decay_per_second'],
                                        td['trace_increase_rate'], params)
    run = temporal_dynamics.simulate('v1', total_duration=200)
    brightness_trace = run['brightness'][:, 0]
    peaks, _ = scipy.signal.find_peaks(brightness_trace)
    np.testing.assert_allclose(fast[:len(peaks)], brightness_trace[peaks],
                               rtol=1e-4)


def test_schmidt_fit():
    params = load_config('v1')
    td = params['temporal_dynamics']
    peaks = temporal_dynamics.fast_peaks(td['trace_decay_per_second'],
                                         td['trace_increase_rate'], params)
    data = temporal_dynamics.load_data()
    assert r2_pearson(data.y_mean, peaks) == pytest.approx(
        PAPER['schmidt_r2'], abs=1e-3)
