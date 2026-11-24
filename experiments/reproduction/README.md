# Reproducing the Dynaphos calibration fits

Ports of the calibration notebooks from
[neuralcodinglab/dynaphos-experiments](https://github.com/neuralcodinglab/dynaphos-experiments)
(commit `e62ff44`, which pins `dynaphos-core` 0.1.2) to this repository's
simulators. Each experiment runs three ways:

| kind       | simulator                                   | config                    |
|------------|---------------------------------------------|---------------------------|
| `v1`       | `dynaphos.GaussianSimulator` (unmodified)   | `config/params.yaml`      |
| `lgn_hard` | `dynaphos_lgn.LGNPhospheneSimulator`, step  | `dynaphos_lgn/config/params_lgn.yaml`  |
| `lgn_soft` | same, logistic detection probability        | `dynaphos_lgn/config/params_lgn.yaml`  |

Conditions run in parallel along the batch dimension with one electrode,
so spatial parameters play no part; only the temporal model, brightness
and threshold are tested.

| script                 | source notebook                                   | data                                   |
|------------------------|---------------------------------------------------|----------------------------------------|
| `thresholding.py`      | `thresholding/thresholding_Oct_28`, `..._LOOCV`   | Fernandez et al. 2021, Fig. 2          |
| `brightness.py`        | `brightness/brightness_Oct_28`                    | Fernandez et al. 2021, Fig. 6A         |
| `temporal_dynamics.py` | `temporal_dynamics/temporal_dynamics_Oct_22`      | Schmidt et al. 1996                    |
| `consistency.py`       | (new) joint refit, rheobase, frame rate           |                                        |

Run from the repository root; tables are printed and CSVs and figures go
to `experiments/reproduction/out/`:

```bash
python -m experiments.reproduction.thresholding
python -m experiments.reproduction.brightness
python -m experiments.reproduction.temporal_dynamics --refit
python -m experiments.reproduction.consistency
pytest -m slow experiments/reproduction/test_reproduction.py
```

`temporal_dynamics.py` simulates 310k frames per kind (about 1.5 min
for V1, 5 min per LGN kind on CPU).

## Results

R^2 is the squared Pearson correlation between data and prediction, the
convention of the original notebooks.

| quantity                 | paper        | v1           | lgn_hard     | lgn_soft     |
|--------------------------|--------------|--------------|--------------|--------------|
| rheobase (charge-dur.)   | 22.37 uA     | 22.37 uA     |              |              |
| charge-duration R^2      | 0.915        | 0.915        |              |              |
| activation threshold mu  | 9.1419e-8    | 9.1361e-8    | 9.1361e-8    | 9.1361e-8    |
| activation threshold sd  | 6.7159e-8    | 6.7127e-8    | 6.7127e-8    | 6.7127e-8    |
| psychometric R^2         | 0.868        | 0.868        | 0.868        | 0.868        |
| LOOCV pooled R^2         | 0.844        | 0.843        | 0.843        | 0.843        |
| brightness R^2           | 0.9501       | 0.9501       | 0.9501       | 0.9501       |
| slope_brightness (refit) | 1.9153e7     | 1.9349e7     | 1.9349e7     | 1.9349e7     |
| cps_half (refit)         | 1.0576e-7    | 1.0469e-7    | 1.0469e-7    | 1.0469e-7    |
| Schmidt R^2              | 0.930        | 0.930        | 0.930        | 0.930        |

V1 also matches dynaphos's stored regression values
(`test/data/fernandez_*_fit.npy`, `results_dynamics.pkl`), and the LGN
activation, trace and brightness equal V1's for the same input
(`test_lgn_temporal_model_matches_v1`).

The small brightness refit differences come from the trace parameters:
refitted with the older trace values the paper used (0.999469, 12.2129)
the slope is 1.9176e7 and cps_half 1.0563e-7, within 0.2%. Likewise
`temporal_dynamics.py --refit` with the brightness values the paper's
trace fit used (slope 3.628e7, cps_half 5.586e-8) recovers
trace_decay_per_second 0.9994927 and trace_increase_rate 13.951
(published 0.9994919, 13.955); with the published brightness values it
gives 0.999606 and 16.17 instead.

### Detection with the soft threshold

Probit fits to the detection proportions each simulator produces with its
own threshold draws (200 per condition, activation > 0):

| kind                     | 50% activation | spread  | spread / calibrated sd | R^2 vs data |
|--------------------------|----------------|---------|------------------------|-------------|
| `v1`                     | 9.17e-8        | 6.57e-8 | 0.98                   | 0.856       |
| `lgn_hard`               | 9.17e-8        | 6.56e-8 | 0.98                   | 0.857       |
| `lgn_soft`, slope 10/sd  | 9.21e-8        | 6.61e-8 | 0.98                   | 0.860       |
| `lgn_soft`, slope 1/sd   | 1.30e-7        | 9.58e-8 | 1.43                   | 0.850       |

A trial counts as seen with the trial's peak detection probability; with
200 draws each proportion carries about +-0.035 sampling noise, so the
ratio of 0.98 is noise.

The soft gate applies a logistic *on top of* thresholds already drawn
from N(mu, sd) per electrode, and the `zero_activation` baseline subtracts
p(0) per electrode. With the former default slope 1/sd the logistic adds
its own spread, and the baseline pushes up the curve (p = 0.37 at
A = T for a typical threshold), giving a psychometric curve 42% higher
and 43% wider than the one fitted to Fernandez et al. The default slope
is now 10/sd: the logistic's 10-90% rise spans 0.44 sd, p(0) is about
1e-6 for a typical threshold, and the curve matches the calibration
while subthreshold electrodes still get a gradient. The variability
stays between electrodes, as in the literature, rather than within one.

Slope sweep (multiples of 1/sd, spread / calibrated sd):

| slope | `zero_activation` | `none` |
|-------|-------------------|--------|
| 1     | 1.43              | 1.94   |
| 3     | 1.02              | 1.13   |
| 10    | 0.98              | 1.00   |
| 30    | 0.98              | 0.98   |

### Calibration provenance (`consistency.py`)

* **Joint refit.** The brightness fit used older trace parameters and the
  trace fit older brightness parameters. Alternating both fits from the
  published values converges in 4 iterations to trace_decay_per_second
  0.999597, trace_increase_rate 15.91, slope 1.954e7, cps_half 1.036e-7,
  with brightness R^2 0.949 and Schmidt R^2 0.927. The published set is
  not a fixed point but fits about as well; the trace parameters are
  weakly constrained by the Schmidt data.
* **Rheobase.** The config's 23.9 uA is the paper's value, fitted on
  Fernandez et al. (2021). The notebook's own charge-duration fit on the
  digitized thresholds gives 22.37 uA; with that value the psychometric
  R^2 is 0.875 (vs 0.868) and mu is 5% higher. Kept for reference only.
* **Frame rate.** The fits ran at 256-1000 fps, the simulator usually at
  35 fps. Trains last `int(duration * fps)` frames and the activation is
  a per-frame Euler step, so short trains depend on fps. Peak activation
  relative to 1000 fps (60 uA):

  | fps | 50 ms train  | 83 ms train  | 166 ms train |
  |-----|--------------|--------------|--------------|
  | 35  | 0.72 (1 fr)  | 0.87 (2 fr)  | 1.02 (5 fr)  |
  | 60  | 1.07 (3 fr)  | 0.92 (4 fr)  | 1.00 (9 fr)  |
  | 256 | 0.96         | 1.00         | 1.00         |

  For the default 166 ms train the effect is negligible (50% detection
  current 45.9 uA at 35 fps vs 46.2 uA at 1000 fps); trains shorter
  than about 5 frames are not.

## Notes on the original code

* `dynaphos-core` 0.1.2 `sample_stimulus` multiplies by an undefined
  `self.stimulus_scale` and raises; 0.1.3 (this repo) returns unscaled
  values unless `rescale=True`. The notebooks bypassed it.
* The notebooks use paths from an older layout (`../../config`) and
  pandas < 2 (`DataFrame.append`).
* The original brightness notebook prints r, not R^2 (0.975 vs 0.950).
* The paper text gives lambda = 19.2e7 and A50 = 1.06e-6; the code and
  the fit give 1.92e7 and 1.06e-7. Both text values are 10x typos: the
  Fig. 2 activation peaks (about 3e-7) and the resting brightness of
  0.117 = S(-lambda A50) only fit the code values. kappa = 14.0 in the
  text is `trace_increase_rate` (13.955).
* The charge axis in the paper's Fig. 3 is labeled uC; it is nC.
* `Fernandez_2021_fig6A.csv` came with the original repository's first
  commit and is normalized to 1.0 at 90 uA; how it was digitized is not
  recorded. The brightness fit depends on that normalization.
* `PAPER_*` constants in `common.py` record the values the original fits
  ran with where they differ from the config.
