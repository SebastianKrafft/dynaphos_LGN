"""Habituation calibration against Schmidt et al. (1996).

Port of dynaphos-experiments/temporal_dynamics/temporal_dynamics_Oct_22
.ipynb. 125 ms trains (90 uA, 100 us, 200 Hz) every 4 s for 200 s, then
every 200 s up to 1200 s, simulated at 256 fps. The peak brightness of
each train is compared with the mean relative brightness Schmidt et al.
reported (56 points).

`--refit` refits (trace_decay_per_second, trace_increase_rate) with a
fast evaluator that applies the model's per-frame Euler decay in closed
form between trains; it is checked against the full simulator run first.

Run from the repo root:  python -m experiments.reproduction.temporal_dynamics [--refit]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from scipy.optimize import curve_fit

from experiments.reproduction.common import (COLORS, DATA_DIR, FIGURE_STYLE, KINDS,
                                 OUT_DIR, PAPER,
                                 PAPER_TRACE_FIT_BRIGHTNESS, ROOT,
                                 comparison_table, get_path, load_config,
                                 make_simulator, r2_pearson, run_protocol,
                                 set_path, sigmoid_curve)

FPS = 256
TOTAL_DURATION = 1210
TRAIN_DURATION = 0.125
AMPLITUDE = 90e-6
PULSE_WIDTH = 100e-6
FREQUENCY = 200
ONSETS = np.concatenate([np.linspace(0, 200, 50, endpoint=False),
                         np.linspace(200, 1200, 6)])


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / 'Schmidt_et_al_1996.csv')


def stimulus(total_duration: float = TOTAL_DURATION) -> np.ndarray:
    """(n_frames, 1) amplitude sequence."""
    stim = np.zeros(int(FPS * total_duration))
    n_on = int(FPS * TRAIN_DURATION)
    for onset in ONSETS:
        idx = int(onset * FPS)
        stim[idx:idx + n_on] = AMPLITUDE
    return stim[:, None]


def train_peaks(brightness: np.ndarray) -> np.ndarray:
    """Peak brightness per train, found as in the original notebook."""
    peaks, _ = scipy.signal.find_peaks(brightness)
    if len(peaks) != len(ONSETS):
        raise RuntimeError(f"found {len(peaks)} peaks, expected "
                           f"{len(ONSETS)}")
    return brightness[peaks]


def simulate(kind: str, overrides=None, total_duration=TOTAL_DURATION):
    sim = make_simulator(kind, 1, FPS, overrides, use_threshold=False)
    return run_protocol(sim, stimulus(total_duration), PULSE_WIDTH, FREQUENCY,
                        traces=('activation', 'trace', 'brightness'))


def fast_peaks(trace_decay_per_second: float, trace_increase_rate: float,
               params: dict) -> np.ndarray:
    """Per-train peak brightness of the Dynaphos model, simulating only
    the stimulated frames. Between trains the input is zero, so each
    frame multiplies activation and trace by a constant factor."""
    dt = 1 / FPS
    act_decay = -np.log(get_path(params, 'temporal_dynamics.'
                                         'activation_decay_per_second'))
    trace_decay = -np.log(trace_decay_per_second)
    rheobase = params['thresholding']['rheobase']
    slope = params['brightness_saturation']['slope_brightness']
    half = params['brightness_saturation']['cps_half']
    n_on = int(FPS * TRAIN_DURATION)

    activation = trace = 0.0
    last_frame = -1
    peaks = []
    for onset in ONSETS:
        start = int(onset * FPS)
        gap = start - last_frame - 1
        activation *= (1 - act_decay * dt) ** gap
        trace *= (1 - trace_decay * dt) ** gap
        best = -np.inf
        for _ in range(n_on):
            current = max(0.0, (AMPLITUDE - trace - rheobase)
                          * PULSE_WIDTH * FREQUENCY)
            activation += (-act_decay * activation + current) * dt
            trace += -trace_decay * dt * trace + trace_increase_rate * dt * current
            best = max(best, activation)
        last_frame = start + n_on - 1
        peaks.append(best)
    return sigmoid_curve(np.array(peaks), slope, half)


def refit(data: pd.DataFrame, params: dict):
    """(trace_decay_per_second, trace_increase_rate, r2) fitted to the
    Schmidt means, as in the original notebook (peaks normalised)."""
    def model(_, decay, increase):
        peaks = fast_peaks(decay, increase, params)
        return peaks / peaks.max()
    (decay, increase), _ = curve_fit(model, data.time, data.y_mean,
                                     p0=[.9995, 15],
                                     bounds=([0, 0], [1 - 1e-12, 100]))
    return decay, increase, r2_pearson(data.y_mean,
                                       model(None, decay, increase))


def regression_check() -> bool:
    """V1 over the first 200 s against dynaphos's stored regression run."""
    res = simulate('v1', total_duration=200)
    expected = pd.read_pickle(ROOT / 'test' / 'data' / 'results_dynamics.pkl')
    return all(np.allclose(res[key][:, 0], expected[key])
               for key in ('activation', 'trace', 'brightness'))


@plt.style.context(FIGURE_STYLE)
def plot(data, peaks, path):
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharey=True)
    for ax, n in zip(axs, [50, len(data)]):
        ax.fill_between(data.time[:n], data.conf_95_lower[:n],
                        data.conf_95_upper[:n], color=COLORS['turquoise'][2],
                        label='95% CI')
        ax.plot(data.time[:n], data.y_mean[:n], '--', color=COLORS['turquoise'][0],
                label='Schmidt 1996')
        for (kind, y), style in zip(peaks.items(), ['-o', '--', ':']):
            ax.plot(data.time[:n], (y / y.max())[:n], style, ms=3, label=kind)
        ax.set(xlabel='Time [s]', ylabel='Relative brightness', ylim=[0, 1.1])
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(kinds=KINDS, do_refit=False):
    OUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    runs = {kind: simulate(kind) for kind in kinds}
    peaks = {kind: train_peaks(res['brightness'][:, 0])
             for kind, res in runs.items()}
    r2 = {kind: r2_pearson(data.y_mean, y) for kind, y in peaks.items()}
    print(comparison_table([('schmidt_r2', PAPER['schmidt_r2'], r2)]))

    traces_match = {kind: all(np.allclose(runs[kind][key], runs['v1'][key],
                                          rtol=1e-5, atol=1e-12)
                              for key in ('activation', 'trace', 'brightness'))
                    for kind in runs if 'v1' in runs}
    print(f"\nTraces identical to V1: {traces_match}")
    print(f"V1 matches test/data regression run: {regression_check()}")

    table = data[['time', 'y_mean']].copy()
    for kind, y in peaks.items():
        table[f'peak_brightness_{kind}'] = y
    table.to_csv(OUT_DIR / 'temporal_dynamics.csv', index=False)
    plot(data, peaks, OUT_DIR / 'temporal_dynamics.png')

    if do_refit:
        params = load_config('v1')
        td = params['temporal_dynamics']
        fast = fast_peaks(td['trace_decay_per_second'],
                          td['trace_increase_rate'], params)
        reference = (peaks['v1'] if 'v1' in peaks
                     else train_peaks(simulate('v1')['brightness'][:, 0]))
        print(f"\nFast evaluator max relative deviation from the simulator: "
              f"{np.max(np.abs(fast / reference - 1)):.2e}")
        for label, overrides in [('published brightness', {}),
                                 ('brightness the paper fit used',
                                  PAPER_TRACE_FIT_BRIGHTNESS)]:
            p = load_config('v1')
            for path, value in overrides.items():
                set_path(p, path, value)
            decay, increase, fit_r2 = refit(data, p)
            print(f"Refit with {label}: trace_decay_per_second {decay:.8f} "
                  f"(paper {PAPER['trace_decay_per_second']}), "
                  f"trace_increase_rate {increase:.6f} "
                  f"(paper {PAPER['trace_increase_rate']}), R^2 {fit_r2:.4f}")
    return r2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--kinds', nargs='+', default=list(KINDS),
                        choices=KINDS)
    parser.add_argument('--refit', action='store_true')
    args = parser.parse_args()
    main(args.kinds, args.refit)
