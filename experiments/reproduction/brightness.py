"""Brightness calibration against Fernandez et al. (2021), Fig. 6A.

Port of dynaphos-experiments/brightness/brightness_Oct_28.ipynb. A 166 ms
train (170 us, 300 Hz) at 10-100 uA is followed by 834 ms of rest, at
500 fps. The peak brightness per amplitude is compared with the reported
relative brightness, and the brightness sigmoid is refitted on the peak
activation.

Run from the repo root:  python -m experiments.reproduction.brightness
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reproduction.common import (COLORS, DATA_DIR, FIGURE_STYLE, KINDS,
                                 OUT_DIR, PAPER,
                                 PAPER_BRIGHTNESS_FIT_TRACE, ROOT,
                                 comparison_table, fit_sigmoid,
                                 make_simulator, r2_pearson, run_protocol,
                                 sigmoid_curve, train_amplitudes)

FPS = 500
N_FRAMES = 500
TRAIN_DURATION = 83 / FPS
PULSE_WIDTH = 170e-6
FREQUENCY = 300


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / 'Fernandez_2021_fig6A.csv')


def simulate(kind: str, amplitudes, fps: float = FPS, n_frames: int = N_FRAMES,
             overrides=None, traces=()):
    """Peak activation and brightness per amplitude (and optional traces)."""
    sim = make_simulator(kind, len(amplitudes), fps, overrides,
                         use_threshold=False)
    stim = train_amplitudes(amplitudes, TRAIN_DURATION, fps, n_frames)
    return run_protocol(sim, stim, PULSE_WIDTH, FREQUENCY, traces=traces,
                        peaks=('activation', 'brightness'))


def evaluate(kind: str, data: pd.DataFrame, overrides=None) -> dict:
    """Reproduced R^2 and refitted sigmoid, as in the original notebook
    (zero-activation amplitudes are left out of the regression)."""
    res = simulate(kind, data.amplitude.to_numpy(), overrides=overrides)
    active = res['activation_max'] != 0
    slope, half = fit_sigmoid(res['activation_max'], data.brightness)
    return {
        'brightness_r2': r2_pearson(data.brightness[active],
                                    res['brightness_max'][active]),
        'slope_brightness': slope,
        'cps_half': half,
        'activation_max': res['activation_max'],
        'brightness_max': res['brightness_max'],
    }


def regression_check(data: pd.DataFrame) -> bool:
    """V1 peaks against dynaphos's own stored regression values."""
    res = simulate('v1', data.amplitude.to_numpy())
    expected_a = np.load(ROOT / 'test' / 'data' / 'fernandez_activation_fit.npy')
    expected_b = np.load(ROOT / 'test' / 'data' / 'fernandez_brightness_fit.npy')
    return bool(np.allclose(res['activation_max'], expected_a)
                and np.allclose(res['brightness_max'], expected_b))


@plt.style.context(FIGURE_STYLE)
def plot(data, results, traces, path):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    time = np.arange(N_FRAMES) / FPS
    colors = plt.cm.magma_r(np.linspace(0.2, 0.9, len(data)))
    for i, color in enumerate(colors):
        axs[0].plot(time, traces['activation'][:, i], color=color)
        axs[1].plot(time, traces['brightness'][:, i], color=color)
    axs[0].set(xlim=[0, 0.7], xlabel='Time [s]', ylabel='Activation')
    axs[1].set(xlim=[0, 0.7], xlabel='Time [s]', ylabel='Brightness')

    amp_ua = data.amplitude * 1e6
    axs[2].scatter(amp_ua, data.brightness, c=COLORS['red'][2], label='Fernandez 2021',
                   zorder=3)
    for (kind, res), style in zip(results.items(), ['-', '--', ':']):
        axs[2].plot(amp_ua, res['brightness_max'], style,
                    label=f"{kind} ($R^2$={res['brightness_r2']:.3f})")
    axs[2].set(xlabel=r'Amplitude [$\mu A$]', ylabel='Peak brightness')
    axs[2].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(kinds=KINDS):
    OUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    results = {kind: evaluate(kind, data) for kind in kinds}
    paper_trace = evaluate('v1', data, PAPER_BRIGHTNESS_FIT_TRACE)

    rows = [(key, PAPER[key], {k: r[key] for k, r in results.items()})
            for key in ('brightness_r2', 'slope_brightness', 'cps_half')]
    print(comparison_table(rows))
    print('\nRefit with the trace parameters the paper fit used '
          '(v1): slope {slope_brightness:.6g}, cps_half {cps_half:.6g}'
          .format(**paper_trace))
    print(f"V1 matches test/data regression values: {regression_check(data)}")

    table = data.copy()
    for kind, res in results.items():
        table[f'activation_max_{kind}'] = res['activation_max']
        table[f'brightness_max_{kind}'] = res['brightness_max']
    table.to_csv(OUT_DIR / 'brightness.csv', index=False)
    traces = simulate('v1', data.amplitude.to_numpy(),
                      traces=('activation', 'brightness'))
    plot(data, results, traces, OUT_DIR / 'brightness.png')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--kinds', nargs='+', default=list(KINDS),
                        choices=KINDS)
    main(parser.parse_args().kinds)
