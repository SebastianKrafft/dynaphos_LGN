"""Threshold calibration against Fernandez et al. (2021), Fig. 2.

Port of dynaphos-experiments/thresholding/thresholding_Oct_28.ipynb and
thresholding_LOOCV.ipynb:

1. charge-duration fit Q = I0 t + I0 c on the 50% detection thresholds;
2. peak tissue activation for every Fig. 2 condition (1 s window,
   1000 fps), and a probit fit P = Phi((A - mu) / sigma) of the reported
   detection proportions;
3. leave-conditions-out cross-validation of that fit (seed-42 folds);
4. detection proportions simulated with each simulator's own threshold
   draws: V1 and LGN-hard count a trial as seen when activation exceeds
   its threshold, LGN-soft takes the trial's peak detection probability.
   A probit refit of those proportions gives each simulator's effective
   psychometric spread.

Run from the repo root:  python -m experiments.reproduction.thresholding
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reproduction.common import (DATA_DIR, FIGURE_STYLE, KINDS, OUT_DIR,
                                 PAPER,
                                 add_stimulus_columns, comparison_table,
                                 fit_charge_duration, fit_probit,
                                 make_simulator, probit, r2_pearson,
                                 run_protocol, train_amplitudes)

FPS = 1000
WINDOW = 1.0           # seconds simulated per trial
N_DRAWS = 200          # threshold draws per condition
CONDITION = ['pulse_width', 'frequency', 'train_duration', 'label']
PANELS = [('fig2A', 'pulse_width'), ('fig2B', 'frequency'),
          ('fig2C', 'train_duration')]


def load_data():
    proportions = add_stimulus_columns(
        pd.read_csv(DATA_DIR / 'Fernandez_et_al_2021_fig2.csv',
                    float_precision='high'))
    thresholds = add_stimulus_columns(
        pd.read_csv(DATA_DIR / 'Fernandez_et_al_2021_fig2_50perc_thresholds'
                                '.csv', float_precision='high'))
    return proportions, thresholds


def simulate(kind: str, rows: pd.DataFrame, n_draws: int = 1,
             fps: float = FPS, overrides=None, use_threshold=False):
    """Peak activation and detection per trial; each row is repeated
    `n_draws` times along the batch, each with its own threshold."""
    rows = rows.loc[rows.index.repeat(n_draws)]
    sim = make_simulator(kind, len(rows), fps, overrides,
                         use_threshold=use_threshold)
    stim = train_amplitudes(rows.amplitude.to_numpy(),
                            rows.train_duration.to_numpy(), fps,
                            int(WINDOW * fps))
    peaks = ('activation', 'detection_probability') if use_threshold \
        else ('activation',)
    return run_protocol(sim, stim, rows.pulse_width.to_numpy(),
                        rows.frequency.to_numpy(), peaks=peaks)


def max_activation(kind: str, rows: pd.DataFrame, **kwargs) -> np.ndarray:
    return simulate(kind, rows, **kwargs)['activation_max']


def simulated_proportion(kind: str, rows: pd.DataFrame,
                         n_draws: int = N_DRAWS, **kwargs) -> np.ndarray:
    """Fraction of seen trials per row, with per-trial threshold draws."""
    res = simulate(kind, rows, n_draws, use_threshold=True, **kwargs)
    return res['detection_probability_max'].reshape(len(rows),
                                                    n_draws).mean(1)


def cross_validate(proportions, thresholds, activation, seed=42):
    """Leave-conditions-out probit fits, as in thresholding_LOOCV.

    :return: (pooled r2, per-condition r2, per-fold rheobase)
    """
    idx = np.arange(len(thresholds))
    np.random.default_rng(seed).shuffle(idx)
    data = proportions.assign(max_activation=activation)
    keys = data[CONDITION].apply(tuple, axis=1)

    y_true, y_pred, per_condition, rheobases = [], [], [], []
    for test in np.split(idx, [3, 6]):
        train = np.setdiff1d(idx, test)
        rheobases.append(fit_charge_duration(
            thresholds.corrected_duration.iloc[train],
            thresholds.charge_per_trial.iloc[train])[0])
        train_keys = set(thresholds[CONDITION].iloc[train].apply(tuple, axis=1))
        train_rows = data[keys.isin(train_keys)]
        mu, sigma = fit_probit(train_rows.max_activation,
                               train_rows.proportion_phosphene_perception)
        for i in test:
            rows = data[keys == tuple(thresholds[CONDITION].iloc[i])]
            pred = probit(rows.max_activation, mu, sigma)
            per_condition.append(r2_pearson(
                rows.proportion_phosphene_perception, pred))
            y_true.append(rows.proportion_phosphene_perception)
            y_pred.append(pred)
    pooled = r2_pearson(np.concatenate(y_true), np.concatenate(y_pred))
    return pooled, per_condition, rheobases


@plt.style.context(FIGURE_STYLE)
def plot(proportions, predicted, simulated, path):
    """Data (dots) against the V1 probit fit (line) and each simulator's
    simulated proportions."""
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
    styles = {'v1': '-', 'lgn_hard': '--', 'lgn_soft': ':'}
    for ax, (label, param) in zip(axs, PANELS):
        sub = proportions[proportions.label == label]
        for j, (value, rows) in enumerate(sub.groupby(param)):
            color = f'C{j}'
            x = rows.charge_per_phase * 1e9
            ax.plot(x, rows.proportion_phosphene_perception, 'o', ms=3,
                    color=color, label=f'{param}={value:g}')
            ax.plot(x, predicted[rows.index], '-', color=color, alpha=.4,
                    lw=3)
            for kind, p in simulated.items():
                ax.plot(x, p[rows.index], styles[kind], color=color, lw=1)
        ax.set(title=label, xlabel='Charge per phase [nC]',
               ylabel='P(phosphene)')
        ax.legend()
    for kind, p in simulated.items():
        axs[3].plot(proportions.proportion_phosphene_perception, p, 'o',
                    ms=3, label=kind)
    axs[3].plot([0, 1], [0, 1], 'k--', lw=1)
    axs[3].set(xlabel='Observed', ylabel='Simulated')
    axs[3].legend()
    fig.text(0.01, 0.01, 'thick faint line: probit fit; thin lines: '
             'simulated (solid v1, dashed lgn_hard, dotted lgn_soft)')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main(kinds=KINDS, n_draws=N_DRAWS):
    OUT_DIR.mkdir(exist_ok=True)
    proportions, thresholds = load_data()
    observed = proportions.proportion_phosphene_perception

    rheobase, chronaxie, cd_r2 = fit_charge_duration(
        thresholds.corrected_duration, thresholds.charge_per_trial)
    print(f"Charge-duration fit: rheobase {rheobase * 1e6:.2f} uA "
          f"(paper {PAPER['rheobase'] * 1e6:.2f}; config uses 23.9), "
          f"chronaxie {chronaxie * 1e3:.3f} ms, R^2 {cd_r2:.3f} "
          f"(paper {PAPER['charge_duration_r2']})\n")

    rows, table = [], proportions.copy()
    fitted, simulated = {}, {}
    for kind in kinds:
        activation = max_activation(kind, proportions)
        mu, sigma = fit_probit(activation, observed)
        predicted = probit(activation, mu, sigma)
        pooled, per_condition, _ = cross_validate(proportions, thresholds,
                                                  activation)
        sim_p = simulated_proportion(kind, proportions, n_draws)
        # Zero activation is never seen, whatever the threshold; leave
        # those rows out so the fit measures the curve's spread
        active = activation > 0
        eff_mu, eff_sigma = fit_probit(activation[active], sim_p[active])
        fitted[kind] = predicted * (activation > 0)   # as in the paper's plot
        simulated[kind] = sim_p
        table[f'max_activation_{kind}'] = activation
        table[f'simulated_proportion_{kind}'] = sim_p
        rows.append({'kind': kind, 'activation_threshold': mu,
                     'activation_threshold_sd': sigma,
                     'psychometric_r2': r2_pearson(observed, predicted),
                     'loocv_r2': pooled,
                     'simulated_r2': r2_pearson(observed, sim_p),
                     'effective_threshold': eff_mu,
                     'effective_sd': eff_sigma,
                     'effective_sd_ratio': eff_sigma /
                     PAPER['activation_threshold_sd']})

    results = pd.DataFrame(rows).set_index('kind')
    print(comparison_table([(col, PAPER.get(col), results[col].to_dict())
                            for col in results.columns]))
    print("\n'effective_*' are probit fits to each simulator's own "
          f"detection proportions ({n_draws} threshold draws per condition).")

    table.to_csv(OUT_DIR / 'thresholding.csv', index=False)
    results.to_csv(OUT_DIR / 'thresholding_summary.csv')
    plot(proportions, fitted[kinds[0]], simulated,
         OUT_DIR / 'thresholding.png')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--kinds', nargs='+', default=list(KINDS),
                        choices=KINDS)
    parser.add_argument('--draws', type=int, default=N_DRAWS)
    args = parser.parse_args()
    main(args.kinds, args.draws)
