"""Shared pieces for reproducing the Dynaphos calibration fits.

Every experiment runs its stimulation conditions in parallel along the
simulator's batch dimension, with a single electrode, so one protocol
runner drives both the V1 `GaussianSimulator` and the
`LGNPhospheneSimulator`:

    v1        dynaphos.GaussianSimulator, config/params.yaml
    lgn_hard  LGNPhospheneSimulator, config/params_lgn.yaml, hard gate
    lgn_soft  as lgn_hard, logistic detection probability
"""
import functools
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import linregress, norm

from dynaphos.cortex_models import \
    get_visual_field_coordinates_probabilistically
from dynaphos.simulator import GaussianSimulator
from dynaphos.utils import load_params

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / 'data'
OUT_DIR = Path(__file__).resolve().parent / 'out'
FIGURE_STYLE = ROOT / '_figurestyle' / 'default.mplstyle'
COLORS = load_params(str(ROOT / '_figurestyle' / 'colors.yaml'))
KINDS = ('v1', 'lgn_hard', 'lgn_soft')

# Values reported by the original notebooks (dynaphos-experiments@e62ff44).
# 'r2' is the squared Pearson correlation between data and prediction,
# the convention used there.
PAPER = {
    'rheobase': 22.37e-6,           # notebook fit; the paper uses 23.9e-6
    'charge_duration_r2': 0.915,
    'activation_threshold': 9.141886000943878e-08,
    'activation_threshold_sd': 6.715877869833961e-08,
    'psychometric_r2': 0.868,
    'loocv_r2': 0.844,
    'slope_brightness': 19152642.438042317,
    'cps_half': 1.0576313268881034e-07,
    'brightness_r2': 0.9501389853338781,
    'trace_decay_per_second': 0.99949191,
    'trace_increase_rate': 13.95528162,
    'schmidt_r2': 0.9299566685193392,
}

# Parameters the original fits were run with, where they differ from
# the published config (see README, "calibration provenance").
PAPER_BRIGHTNESS_FIT_TRACE = {
    'temporal_dynamics.trace_decay_per_second': 0.999469,
    'temporal_dynamics.trace_increase_rate': 12.212884}
PAPER_TRACE_FIT_BRIGHTNESS = {
    'brightness_saturation.slope_brightness': 36279530.38871381,
    'brightness_saturation.cps_half': 5.5862273679415555e-08}


def set_path(params: dict, path: str, value):
    """Set a dotted-path entry, e.g. 'run.fps'."""
    *parents, leaf = path.split('.')
    node = params
    for key in parents:
        node = node[key]
    node[leaf] = value


def get_path(params: dict, path: str):
    node = params
    for key in path.split('.'):
        node = node[key]
    return node


def load_config(kind: str) -> dict:
    name = 'params.yaml' if kind == 'v1' else 'params_lgn.yaml'
    return load_params(str(ROOT / 'config' / name))


@functools.lru_cache(maxsize=None)
def _lgn_electrode_array():
    """One LGN electrode, built once per process (atlas loading is slow).
    The array does not depend on frame rate, batch size or thresholds."""
    from dynaphos_lgn.build import build_simulator
    params = load_config('lgn')
    params['run']['gpu'] = None
    params['electrodes']['n_electrodes'] = 1
    params['electrodes']['max_eccentricity_deg'] = 5.0
    _, parts = build_simulator(params)
    return parts['electrode_array']


def make_simulator(kind: str, batch_size: int, fps: float,
                   overrides: Optional[Dict[str, object]] = None,
                   use_threshold: bool = True, seed: int = 42):
    """A single-electrode simulator with `batch_size` independent trials.

    :param overrides: Dotted-path config overrides, e.g.
        {'thresholding.rheobase': 22.37e-6}.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    params = load_config(kind)
    params['run'].update(fps=fps, batch_size=batch_size, gpu=None, seed=seed)
    params['thresholding']['use_threshold'] = use_threshold
    for path, value in (overrides or {}).items():
        set_path(params, path, value)
    rng = np.random.default_rng(seed)

    if kind == 'v1':
        params['cortex_model']['dropout_rate'] = 0
        coordinates = get_visual_field_coordinates_probabilistically(
            params, 1, rng)
        return GaussianSimulator(params, coordinates, rng)

    from dynaphos_lgn.simulator import LGNPhospheneSimulator
    params['thresholding']['soft_threshold'] = kind == 'lgn_soft'
    return LGNPhospheneSimulator(params, _lgn_electrode_array(), rng)


@torch.no_grad()
def run_protocol(sim, amplitude: np.ndarray, pulse_width, frequency,
                 traces: Iterable[str] = (), peaks: Iterable[str] = ()
                 ) -> Dict[str, np.ndarray]:
    """Drive the simulator frame by frame from a zero state.

    :param amplitude: (n_frames, batch) stimulation amplitudes in A.
    :param pulse_width: Scalar or (batch,) pulse widths in s.
    :param frequency: Scalar or (batch,) pulse frequencies in Hz.
    :param traces: State keys to record every frame, as (n_frames, batch).
    :param peaks: State keys whose maximum over frames to record, as
        (batch,), stored under '<key>_max'.
    """
    traces, peaks = tuple(traces), tuple(peaks)
    kwargs = sim.data_kwargs
    batch = amplitude.shape[1]

    def per_trial(x):
        return torch.as_tensor(np.broadcast_to(x, (batch,)).copy(), **kwargs)

    pulse_width, frequency = per_trial(pulse_width), per_trial(frequency)
    amplitude = torch.as_tensor(amplitude, **kwargs)
    recorded = {key: torch.empty(amplitude.shape, **kwargs) for key in traces}
    running = {key: torch.full((batch,), -np.inf, **kwargs) for key in peaks}

    sim.reset()
    for t, frame in enumerate(amplitude):
        sim.update(frame, pulse_width, frequency)
        state = state_of(sim, set(traces) | set(peaks))
        for key in traces:
            recorded[key][t] = state[key]
        for key in peaks:
            running[key] = torch.maximum(running[key], state[key])

    out = {key: value.cpu().numpy() for key, value in recorded.items()}
    out.update({f'{key}_max': value.cpu().numpy()
                for key, value in running.items()})
    return out


def state_of(sim, keys) -> Dict[str, torch.Tensor]:
    """Per-trial (batch,) state values. 'detection_probability' is the hard
    gate (activation > threshold) for the V1 simulator."""
    state = sim.get_state()
    if 'detection_probability' in keys and 'detection_probability' not in state:
        state['detection_probability'] = (
            state['activation'] > sim.threshold.get()).to(
                state['activation'].dtype)
    if 'threshold' in keys and 'threshold' not in state:
        state['threshold'] = sim.threshold.get()
    return {key: state[key].reshape(-1) for key in keys}


def train_amplitudes(amplitude, train_duration, fps: float,
                     n_frames: int) -> np.ndarray:
    """(n_frames, batch) amplitudes for trains starting at frame 0."""
    amplitude = np.atleast_1d(np.asarray(amplitude, dtype=float))
    on_frames = (np.broadcast_to(train_duration, amplitude.shape)
                 * fps).astype(int)
    on = np.arange(n_frames)[:, None] < on_frames[None, :]
    return on * amplitude[None, :]


# ----------------------------------------------------------------------
# Fits
# ----------------------------------------------------------------------
def r2_pearson(y_true, y_pred) -> float:
    """Squared Pearson correlation, the R^2 the original notebooks report."""
    return float(linregress(y_true, y_pred).rvalue ** 2)


def r2_determination(y_true, y_pred) -> float:
    """Coefficient of determination, 1 - SS_res / SS_tot."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1 - ss_res / np.sum((y_true - y_true.mean()) ** 2))


def probit(x, mu, sigma):
    return norm.cdf(x, mu, sigma)


def fit_probit(max_activation, proportion):
    """Psychometric fit as in thresholding_Oct_28: P = Phi((A - mu)/sigma)."""
    (mu, sigma), _ = curve_fit(probit, max_activation, proportion,
                               p0=[50e-9, 2.2e-7],
                               bounds=[[1e-11, 1e-10], [1e-6, 1e-5]])
    return mu, sigma


def sigmoid_curve(x, slope, half):
    return 1 / (1 + np.exp(-slope * (x - half)))


def fit_sigmoid(max_activation, brightness):
    """Brightness fit as in brightness_Oct_28."""
    (slope, half), _ = curve_fit(sigmoid_curve, max_activation, brightness,
                                 p0=[3e7, 5e-8], bounds=([0, 0], [1e8, 1e9]))
    return slope, half


def fit_charge_duration(corrected_duration, charge_per_trial):
    """Q = I0 * t + I0 * c, returns (rheobase I0, chronaxie c, r2)."""
    def line(t, rheobase, chronaxie):
        return rheobase * t + chronaxie * rheobase
    (rheobase, chronaxie), _ = curve_fit(line, corrected_duration,
                                         charge_per_trial)
    pred = line(np.asarray(corrected_duration), rheobase, chronaxie)
    return rheobase, chronaxie, r2_determination(charge_per_trial, pred)


def add_stimulus_columns(data):
    """Derived columns used by the Fernandez fits."""
    data['amplitude'] = data.charge_per_phase / data.pulse_width
    data['charge_per_trial'] = (data.charge_per_phase * data.frequency
                                * data.train_duration)
    data['duty_cycle'] = data.pulse_width * data.frequency
    data['corrected_duration'] = data.duty_cycle * data.train_duration
    return data


def comparison_table(rows) -> str:
    """Plain-text table of (quantity, paper, {kind: value}) rows."""
    kinds = sorted({k for _, _, values in rows for k in values},
                   key=lambda k: (KINDS + (k,)).index(k))
    header = f"{'quantity':<28}{'paper':>14}" + ''.join(
        f'{k:>14}' for k in kinds)
    lines = [header, '-' * len(header)]
    for name, paper, values in rows:
        cells = ''.join(f"{values.get(k, np.nan):>14.5g}" for k in kinds)
        paper = f'{paper:>14.5g}' if paper is not None else f"{'':>14}"
        lines.append(f'{name:<28}{paper}{cells}')
    return '\n'.join(lines)

