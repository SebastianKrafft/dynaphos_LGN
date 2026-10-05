"""Consistency checks on the published calibration (V1 simulator).

1. Joint refit. The paper fitted brightness with older trace parameters
   and the trace with older brightness parameters. Alternating the two
   fits from the published values shows whether the set is
   self-consistent.
2. Rheobase. The notebook's charge-duration fit gives 22.37 uA; the
   config uses the paper's 23.9 uA. Effect on the threshold and
   brightness fits.
3. Frame rate. The fits ran at 256-1000 fps; the simulator usually runs
   at 35 fps. The activation is a per-frame Euler step and trains last a
   whole number of frames, so peak activation depends on fps. Reported:
   the 50% detection current for Fernandez's default stimulus (170 us,
   300 Hz, 166 ms) and the brightness R^2, per fps.

Run from the repo root:  python -m experiments.reproduction.consistency
"""
import numpy as np
import pandas as pd

from experiments.reproduction import brightness, temporal_dynamics, thresholding
from experiments.reproduction.common import (OUT_DIR, PAPER, fit_probit, load_config,
                                 probit, r2_pearson, set_path)

FPS_VALUES = (35, 60, 256, 500, 1000)


def joint_refit(max_iterations: int = 10, tol: float = 1e-4) -> pd.DataFrame:
    fern, schmidt = brightness.load_data(), temporal_dynamics.load_data()
    td = {'trace_decay_per_second': PAPER['trace_decay_per_second'],
          'trace_increase_rate': PAPER['trace_increase_rate']}
    bs = {'slope_brightness': PAPER['slope_brightness'],
          'cps_half': PAPER['cps_half']}
    rows = [{'iteration': 0, **td, **bs}]
    for i in range(1, max_iterations + 1):
        res = brightness.evaluate('v1', fern, {
            f'temporal_dynamics.{k}': v for k, v in td.items()})
        new_bs = {k: res[k] for k in bs}

        params = load_config('v1')
        for k, v in new_bs.items():
            set_path(params, f'brightness_saturation.{k}', v)
        decay, increase, schmidt_r2 = temporal_dynamics.refit(schmidt, params)
        new_td = {'trace_decay_per_second': decay,
                  'trace_increase_rate': increase}

        change = max(abs(new[k] / old[k] - 1)
                     for new, old in ((new_bs, bs), (new_td, td))
                     for k in old)
        td, bs = new_td, new_bs
        rows.append({'iteration': i, **td, **bs,
                     'brightness_r2': res['brightness_r2'],
                     'schmidt_r2': schmidt_r2, 'max_rel_change': change})
        if change < tol:
            break
    return pd.DataFrame(rows).set_index('iteration')


def rheobase_effect(rheobase: float = PAPER['rheobase']) -> pd.DataFrame:
    proportions, _ = thresholding.load_data()
    fern = brightness.load_data()
    rows = []
    for value in (23.9e-6, rheobase):
        overrides = {'thresholding.rheobase': value}
        activation = thresholding.max_activation('v1', proportions,
                                                 overrides=overrides)
        observed = proportions.proportion_phosphene_perception
        mu, sigma = fit_probit(activation, observed)
        b = brightness.evaluate('v1', fern, overrides)
        rows.append({'rheobase_ua': value * 1e6, 'activation_threshold': mu,
                     'activation_threshold_sd': sigma,
                     'psychometric_r2': r2_pearson(
                         observed, probit(activation, mu, sigma)),
                     'slope_brightness': b['slope_brightness'],
                     'cps_half': b['cps_half'],
                     'brightness_r2': b['brightness_r2']})
    return pd.DataFrame(rows).set_index('rheobase_ua')


def fps_sensitivity(fps_values=FPS_VALUES) -> pd.DataFrame:
    params = load_config('v1')['thresholding']
    mu, sigma = params['activation_threshold'], params['activation_threshold_sd']
    amplitudes = np.linspace(0, 150e-6, 601)
    default = pd.DataFrame({'pulse_width': 170e-6, 'frequency': 300,
                            'train_duration': 0.166,
                            'amplitude': amplitudes})
    fern = brightness.load_data()
    rows = []
    for fps in fps_values:
        activation = thresholding.max_activation('v1', default, fps=fps)
        p = probit(activation, mu, sigma) * (activation > 0)
        res = brightness.simulate('v1', fern.amplitude.to_numpy(), fps=fps,
                                  n_frames=int(fps))
        active = res['activation_max'] > 0
        rows.append({
            'fps': fps,
            'train_frames_166ms': int(0.166 * fps),
            'threshold_50_ua': np.interp(0.5, p, amplitudes) * 1e6,
            'peak_activation_100ua': np.interp(100e-6, amplitudes,
                                               activation),
            'brightness_r2': r2_pearson(fern.brightness[active],
                                        res['brightness_max'][active]),
            'peak_brightness_100ua': res['brightness_max'][-1]})
    table = pd.DataFrame(rows).set_index('fps')
    reference = table.loc[max(fps_values)]
    table['peak_activation_rel'] = (table.peak_activation_100ua
                                    / reference.peak_activation_100ua)
    return table


def main():
    OUT_DIR.mkdir(exist_ok=True)
    pd.set_option('display.width', 160)
    pd.set_option('display.float_format', '{:.6g}'.format)
    for name, func in [('Joint brightness/trace refit', joint_refit),
                       ('Rheobase 23.9 vs 22.37 uA', rheobase_effect),
                       ('Frame-rate sensitivity', fps_sensitivity)]:
        table = func()
        print(f'\n== {name} ==\n{table}')
        table.to_csv(OUT_DIR / f'consistency_{func.__name__}.csv')


if __name__ == '__main__':
    main()
