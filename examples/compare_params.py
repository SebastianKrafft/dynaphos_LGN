"""Compare LGN simulator parameter sets, quantitatively, on the same
electrodes and the same images, under optimised stimulation.

Every condition is a set of config overrides applied to one shared base
config. Each condition is built with the same `run.seed`, so the
electrode array is the same draw every time and only the parameter under
test changes. The script checks this and reports it (`same_electrodes`).

Stimulation is optimised, not mapped from brightness. For every image,
the per-electrode amplitudes are found by gradient descent through the
simulator so that the percept matches the image, each amplitude bounded
to [0, `default_stim.safety_limit_ua`] from the config. The optimiser is
free to use low currents where high ones would merge phosphenes into
one bright mass, so a condition is judged by the best percept it can
produce, not by what a fixed brightness-to-current rule makes of it.

    loss       mean squared difference between percept and image, both
               blurred to ANALYSIS_BLUR_DEG and pooled to the analysis
               grid, inside the covered region. The percept is taken
               before the [0, 1] clamp, so light spilling into the
               background is penalised even where it saturates.
    start      every electrode at the rheobase plus START_FRACTION of
               the way to the safety limit. Below the rheobase the model
               delivers no charge, so there is no gradient: an electrode
               started there could never switch on.
    final      amplitudes that end below the rheobase are set to zero
               (they deliver no charge, so a protocol would not send
               them) and the percept is rendered once more.

The optimiser knows each electrode's threshold, as it would after
per-electrode calibration in a patient. A hard threshold passes no
gradient, so `thresholding.soft_threshold: False` is optimised against
its logistic surrogate (the 'none' baseline, whose 50% point is the hard
gate's step) and then rendered with the hard gate.

There is no ground-truth LGN percept to score against, so the metrics
fall into three groups, reported side by side, never combined:

1. Benchmarks against measured LGN data
     threshold_ua        median current at which p(detect) crosses 0.5,
                         one frame (1 / run.fps) of stimulation
     threshold_z         (threshold_ua - 40) / 12, Pezaris & Reid (2007)
     sigma_deg_40ua,     median phosphene sigma (geometric mean of major
     sigma_deg_100ua     and minor), degrees
2. Percept quality -- what a person would have to work with
     digit_acc           MNIST accuracy of a CNN trained on CLEAN digits
                         (blurred to the analysis scale) and tested on
                         the percepts, unchanged. It never sees a percept
                         in training, so it only scores a percept that
                         looks like its digit: a proxy for recognition,
                         not a measure of what the percept encodes
     fit_loss            the optimiser's final loss on the test digits
     bars_contrast_*     (on - off) / (on + off) of the percept over the
                         bars and gaps of each band of the bars image
     e_discrim           1 - mean pairwise correlation of the percepts of
                         the four tumbling-E orientations
     pearson_r, ssim,    percept vs image, both blurred to the analysis
     mse_norm            scale, inside the covered region, averaged over
                         the bars, E and shapes images
3. The stimulation the optimiser chose (test digits)
     frac_active         fraction of electrodes driven above the rheobase
     current_active_ua   mean current of those electrodes
     frac_at_limit       fraction at 99% of the safety limit or more

The threshold noise is the model's only stochastic element, so every
metric is computed `--repeats` times, each with a fresh threshold draw
(and a fresh classifier initialisation). The draws are seeded by repeat
index, so condition A's repeat k and condition B's repeat k use the same
standard-normal draw: the comparison is paired. Stimulation is
re-optimised for every draw.

A reference row, `ideal_input`, scores the classifier on the clean test
digits: the ceiling for digit_acc. Chance is 0.10.

Output (default examples/out/compare_params/):
    metrics.csv        mean and sd per condition
    metrics_raw.csv    one row per condition and repeat
    summary.png        one panel per metric, conditions against baseline
    examples.png       what each condition's percepts look like
    checkpoints/       every finished repeat of every condition, and the
                       trained classifiers

Checkpoints make a run resumable: if it stops (a crash, a CUDA error
after the laptop slept), rerun the same command with --resume and only
the unfinished repeats are computed. Draws and classifier seeds are
fixed by index, so a resumed run gives the numbers an uninterrupted one
would. --resume refuses checkpoints written with different settings or
a different config, and a run into a folder that already holds
checkpoints must say --resume or --fresh, so finished work is never
discarded by accident.

Usage::

    python examples/compare_params.py --quick           # smoke test
    python examples/compare_params.py                   # all conditions
    python examples/compare_params.py --conditions baseline,K_x2
    python examples/compare_params.py --resume          # after a stop

Optimisation dominates the run time: ~0.8 s per image on an RTX 4060
Laptop GPU at the default 60 steps, so about 20 minutes per condition
at the defaults. --n-test and --repeats trade precision for time.

MNIST is read from data/MNIST/ as the four original .gz files. Pass
--download to fetch them there (~11.6 MB) if they are missing.

NOTE `sampling.*` has no effect here: the optimiser, not receptive-field
sampling, turns images into amplitudes. `current_spread.aggregation` is
not a condition either: the forward pass does not call
`aggregate_activation`, so sweeping it would show no effect at all.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import itertools
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# The simulator turns on torch's deterministic mode, which on CUDA
# refuses cuBLAS matmuls unless this is set before CUDA initialises.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch                                                        # noqa: E402
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynaphos.utils import load_params, set_deterministic           # noqa: E402
from dynaphos_lgn.build import build_simulator                      # noqa: E402
from dynaphos_lgn.params import require                             # noqa: E402
from dynaphos_lgn.simulator import LGNPhospheneSimulator            # noqa: E402
from examples.make_test_image import (BAR_BANDS, X_MAX, X_MIN,      # noqa: E402
                                      coordinate_grid, pattern_bars,
                                      pattern_e, pattern_shapes)

REPO = Path(__file__).resolve().parents[1]

# ---- EDIT THIS ----------------------------------------------------------
# name -> {dotted config path: value}. Every path must already exist in
# the config: a typo fails loudly instead of silently adding a key.
CONDITIONS = {
    'baseline': {},
    # K is transplanted from macaque V1; halve and double it.
    #'K_x0.5': {'current_spread.k_ua_per_mm2': 337.5},
    # 'K_x2': {'current_spread.k_ua_per_mm2': 1350.0},
    'bosking_spread': {'current_spread.spread_model': 'bosking'},
    # The four magnification models (baseline is 'jacobian').
    'mag_atlas_gradient': {'magnification.model': 'atlas_gradient'},
    'mag_anisotropic': {'magnification.model': 'anisotropic_gradient'},
    'mag_malpeli_density': {'magnification.model': 'malpeli_density'},
    # Threshold form.
    'hard_threshold': {'thresholding.soft_threshold': False},
    'soft_no_baseline': {'thresholding.soft_threshold_baseline': 'none'},
}
# -------------------------------------------------------------------------

MNIST_URL = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
MNIST_FILES = {'train_images': 'train-images-idx3-ubyte.gz',
               'train_labels': 'train-labels-idx1-ubyte.gz',
               'test_images': 't10k-images-idx3-ubyte.gz',
               'test_labels': 't10k-labels-idx1-ubyte.gz'}

# Where the digit sits: a 28 px MNIST frame scaled to DIGIT_DEG and
# centred here, inside the wedge one left LGN covers. The digit itself
# fills the central ~20 px, so its strokes land around 0.4-0.5 deg wide
# -- near the array's resolution limit, where parameters matter most.
DIGIT_CENTER_DEG = (3.25, 0.0)
E_CENTER_DEG = (3.1, 0.0)     # centre of make_test_image's E

# Analysis scale. Percepts and images are blurred by ANALYSIS_BLUR_DEG
# before being compared, and pooled to ANALYSIS_PIXEL_DEG for the loss,
# the classifier and the image metrics. Fixed across conditions, never
# derived from them.
ANALYSIS_BLUR_DEG = 0.5
ANALYSIS_PIXEL_DEG = 0.25
SSIM_WINDOW_SIGMA_PX = 1.5
CHANCE = 0.1

# Where the optimiser starts, as a fraction of the way from the rheobase
# to the safety limit (~31 uA for 100 uA). Measured on MNIST: starting
# at the rheobase or below leaves electrodes stuck without gradient, and
# starting at half the limit converges more slowly to a worse fit.
START_FRACTION = 0.1

# Reading off a detection threshold.
THRESHOLD_SWEEP_UA = np.arange(1.0, 201.0, 1.0)

# Colours (dataviz reference palette, light mode).
INK, INK_SECONDARY, SERIES, SURFACE = '#0b0b0b', '#52514e', '#2a78d6', \
    '#fcfcfb'
REFERENCE_GREY = '#8f8e89'


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
def base_params(args) -> dict:
    """The shipped LGN config, with the settings every condition shares."""
    params = load_params(str(REPO / 'config' / 'params_lgn.yaml'))
    params['run']['gpu'] = (None if args.device.type == 'cpu'
                            else args.device.index or 0)
    params['run']['batch_size'] = 0
    params['run']['print_stats'] = False
    params['run']['fps'] = args.fps
    params['atlas']['hemispheres'] = ['left']
    params['electrodes']['n_electrodes'] = args.n_electrodes
    params['electrodes']['max_eccentricity_deg'] = args.max_ecc
    return params


def apply_overrides(params: dict, overrides: dict) -> dict:
    params = copy.deepcopy(params)
    for path, value in overrides.items():
        require(params, path)                # the key must already exist
        *parents, leaf = path.split('.')
        node = params
        for key in parents:
            node = node[key]
        node[leaf] = value
    return params


# -------------------------------------------------------------------------
# Checkpoints
# -------------------------------------------------------------------------
def differences(a, b, prefix='') -> list:
    """Dotted paths at which two JSON-like values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [path for key in sorted(set(a) | set(b))
                for path in differences(a.get(key), b.get(key),
                                        f'{prefix}{key}.')]
    return [] if a == b else [prefix.rstrip('.')]


class Checkpoints:
    """One file per finished unit of work, under `directory`.

    `run_config` is everything a checkpoint's numbers depend on apart
    from its condition's overrides, which each checkpoint carries and
    `run_condition` checks. Resuming against a different run_config is
    refused rather than mixing numbers from two setups.
    """

    def __init__(self, directory: Path, run_config: dict, resume: bool,
                 fresh: bool):
        self.dir = directory
        run_config = json.loads(json.dumps(run_config))   # as it reloads
        existing = list(directory.glob('*.pt')) if directory.exists() else []
        if existing and not (resume or fresh):
            sys.exit(f"{directory} holds {len(existing)} checkpoints from an "
                     f"earlier run. Pass --resume to continue it, or --fresh "
                     f"to discard them and start over.")
        if fresh:
            shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / 'run_config.json'
        if resume and existing and config_path.exists():
            changed = differences(json.loads(config_path.read_text()),
                                  run_config)
            if changed:
                sys.exit(f"Cannot resume: the checkpoints in {directory} "
                         f"were written with different settings "
                         f"({', '.join(changed)}). Rerun with the same "
                         f"settings, or pass --fresh to start over.")
        if resume:
            print(f"resuming from {len(existing)} checkpoints in {directory}")
        config_path.write_text(json.dumps(run_config, indent=1,
                                          sort_keys=True))

    def _path(self, name: str) -> Path:
        return self.dir / f'{name}.pt'

    def load(self, name: str):
        path = self._path(name)
        return (torch.load(path, map_location='cpu', weights_only=False)
                if path.exists() else None)

    def save(self, name: str, obj):
        """Write-then-rename, so a stop mid-write leaves no half file."""
        path = self._path(name)
        partial = path.with_name(path.name + '.partial')
        torch.save(obj, partial)
        os.replace(partial, path)


# -------------------------------------------------------------------------
# MNIST
# -------------------------------------------------------------------------
def load_mnist(directory: Path, download: bool) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, name in MNIST_FILES.items():
        path = directory / name
        if not path.exists():
            if not download:
                sys.exit(f"{path} is missing. Place the MNIST files in "
                         f"{directory} or rerun with --download.")
            print(f"downloading {name}")
            urllib.request.urlretrieve(MNIST_URL + name, path)
        with gzip.open(path, 'rb') as f:
            data = f.read()
        if 'images' in key:
            out[key] = np.frombuffer(data, np.uint8, offset=16).reshape(
                -1, 28, 28)
        else:
            out[key] = np.frombuffer(data, np.uint8, offset=8)
    return out


def mnist_subset(mnist: dict, n_train: int, n_test: int, seed: int = 0):
    """A fixed random subset, the same for every condition."""
    rng = np.random.default_rng(seed)
    train = rng.permutation(len(mnist['train_labels']))[:n_train]
    test = rng.permutation(len(mnist['test_labels']))[:n_test]
    return (mnist['train_images'][train], mnist['train_labels'][train],
            mnist['test_images'][test], mnist['test_labels'][test])


def place_digits(digits, X: np.ndarray, Y: np.ndarray, digit_deg: float,
                 device=None) -> torch.Tensor:
    """(N, 28, 28) uint8 -> (N, H, W) float in [0, 1] on the visual-field
    grid, row 0 = bottom (MNIST's row 0 is the top, hence the sign).

    Full-resolution images are ~0.25 MB each, so callers place digits a
    chunk at a time rather than the whole set at once.
    """
    cx, cy = DIGIT_CENTER_DEG
    half = digit_deg / 2
    grid = torch.as_tensor(np.stack([(X - cx) / half, -(Y - cy) / half],
                                    axis=-1), dtype=torch.float32,
                           device=device)[None]
    source = torch.as_tensor(digits, device=device)[:, None].float() / 255
    return F.grid_sample(source, grid.expand(len(source), -1, -1, -1),
                         mode='bilinear', padding_mode='zeros',
                         align_corners=False)[:, 0]


def clean_digits(digits: torch.Tensor, X, Y, digit_deg, grid, device=None,
                 chunk=256) -> torch.Tensor:
    """The digits as the classifier sees them: placed, blurred and
    pooled to the analysis grid, (N, h, w)."""
    return torch.cat([grid.pool_blurred(place_digits(part, X, Y, digit_deg,
                                                     device))
                      for part in digits.split(chunk)])


# -------------------------------------------------------------------------
# Test images
# -------------------------------------------------------------------------
def rotated(pattern, X, Y, center, degrees):
    """`pattern` rotated by `degrees` about `center`."""
    cx, cy = center
    t = np.deg2rad(degrees)
    xr = cx + (X - cx) * np.cos(t) + (Y - cy) * np.sin(t)
    yr = cy - (X - cx) * np.sin(t) + (Y - cy) * np.cos(t)
    return pattern(xr, yr)


def test_images(X, Y, device=None) -> dict:
    images = {'bars': pattern_bars(X, Y), 'shapes': pattern_shapes(X, Y)}
    for angle in (0, 90, 180, 270):
        images[f'e{angle}'] = rotated(pattern_e, X, Y, E_CENTER_DEG, angle)
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in images.items()}


# -------------------------------------------------------------------------
# Optimised stimulation
# -------------------------------------------------------------------------
def threshold_draws(simulator, repeats: int) -> list:
    """One per-electrode threshold tensor per repeat, seeded by index, so
    every condition sees the same standard-normal draws."""
    draws = []
    for r in range(repeats):
        simulator.threshold.rng = np.random.default_rng(1000 + r)
        simulator.threshold.reinitialize()
        draws.append(simulator.threshold.get().clone())
    return draws


class Stimulator:
    """Finds, per image, the amplitudes whose percept best matches it.

    Runs on its own batched copy of the simulator, sharing the electrode
    array, so `batch` images are optimised side by side. Adam scales
    every amplitude's step separately, so images in one batch do not
    influence each other.
    """

    def __init__(self, simulator, params, grid, args):
        self.sim = LGNPhospheneSimulator(
            apply_overrides(params, {'run.batch_size': args.batch}),
            simulator.array, rng=np.random.default_rng(0))
        self.unit = require(params, 'run.current_unit_to_ua')
        self.limit = self.sim.safety_limit_ua / self.unit
        self.rheobase = require(params, 'thresholding.rheobase')
        if self.limit <= self.rheobase:
            sys.exit(f"default_stim.safety_limit_ua "
                     f"({self.sim.safety_limit_ua} uA) is at or below the "
                     f"rheobase ({self.rheobase * self.unit} uA): no "
                     f"permitted current delivers any charge.")
        self.start = (self.rheobase
                      + START_FRACTION * (self.limit - self.rheobase))
        self.grid = grid
        self.mask = grid.mask.to(torch.float32)
        self.batch, self.steps, self.lr = args.batch, args.steps, args.lr

    @property
    def limit_ua(self) -> float:
        return self.sim.safety_limit_ua

    def percept(self, amplitude: torch.Tensor) -> torch.Tensor:
        """One frame from rest, not yet clamped -> (batch, H, W)."""
        self.sim.reset()
        self.sim.update(amplitude)
        return self.sim.unclamped_percept()

    def loss(self, percept: torch.Tensor, wanted: torch.Tensor):
        """Per-image masked MSE on the analysis grid -> (batch,)."""
        diff = (self.grid.pool_blurred(percept) - wanted) ** 2
        return (diff * self.mask).sum((-2, -1)) / self.mask.sum()

    def __call__(self, images: torch.Tensor, threshold: torch.Tensor):
        """(N, H, W) images -> (clamped percepts (N, H, W), amplitudes in
        uA (N, n), final loss (N,))."""
        parts = [self._optimise(part, threshold)
                 for part in images.split(self.batch)]
        return tuple(torch.cat(p) for p in zip(*parts))

    def _optimise(self, images: torch.Tensor, threshold: torch.Tensor):
        n_images = len(images)
        if n_images < self.batch:        # the batched simulator's size is
            images = torch.cat([images, images[-1:].expand(   # fixed: pad
                self.batch - n_images, -1, -1)])
        sim = self.sim
        sim.threshold.state = threshold
        wanted = self.grid.pool_blurred(images)

        hard = not sim.soft_threshold
        baseline = sim.soft_threshold_baseline
        if hard:                         # optimise the logistic surrogate
            sim.soft_threshold, sim.soft_threshold_baseline = True, 'none'
        z = torch.full((self.batch, sim.num_phosphenes),
                       self.start / self.limit, **sim.data_kwargs)
        z = torch.logit(z).requires_grad_(True)
        optimiser = torch.optim.Adam([z], lr=self.lr)
        for _ in range(self.steps):
            optimiser.zero_grad()
            amplitude = self.limit * torch.sigmoid(z)
            self.loss(self.percept(amplitude), wanted).sum().backward()
            optimiser.step()
        sim.soft_threshold, sim.soft_threshold_baseline = not hard, baseline

        with torch.no_grad():
            amplitude = self.limit * torch.sigmoid(z)
            amplitude = torch.where(amplitude > self.rheobase, amplitude,
                                    torch.zeros_like(amplitude))
            percept = self.percept(amplitude).clamp(0, 1)
            loss = self.loss(percept, wanted)
        return (percept[:n_images], amplitude[:n_images] * self.unit,
                loss[:n_images])


@torch.no_grad()
def check_batched(simulator, stimulator, threshold):
    """The batched copy must render row by row what the simulator itself
    renders for the same amplitudes."""
    generator = torch.Generator().manual_seed(0)
    amplitude = (torch.rand(stimulator.batch, simulator.num_phosphenes,
                            generator=generator)
                 * stimulator.limit).to(**simulator.data_kwargs)
    stimulator.sim.threshold.state = threshold
    batched = stimulator.percept(amplitude).clamp(0, 1)
    simulator.reset()
    simulator.threshold.state = threshold
    single = simulator(amplitude[0])
    if not torch.allclose(batched[0], single, atol=1e-5):
        raise RuntimeError("The batched simulator no longer matches the "
                           "unbatched one.")


# -------------------------------------------------------------------------
# Analysis grid
# -------------------------------------------------------------------------
class AnalysisGrid:
    """Crop to the covered (right) half, blur and pool to a fixed grid."""

    def __init__(self, params, X, Y, max_ecc, device=None):
        deg_per_px = (X[0, 1] - X[0, 0])
        self.pool = max(1, int(round(ANALYSIS_PIXEL_DEG / deg_per_px)))
        cols = np.nonzero(X[0] >= 0)[0]
        n_cols = len(cols) // self.pool * self.pool
        n_rows = X.shape[0] // self.pool * self.pool
        self.cols = slice(cols[0], cols[0] + n_cols)
        self.rows = slice(0, n_rows)
        self.blur_sigma_px = ANALYSIS_BLUR_DEG / deg_per_px
        px = self.pool_plain(torch.as_tensor(X, dtype=torch.float32,
                                             device=device))
        py = self.pool_plain(torch.as_tensor(Y, dtype=torch.float32,
                                             device=device))
        self.mask = (torch.hypot(px, py) <= max_ecc)

    def pool_plain(self, image: torch.Tensor) -> torch.Tensor:
        """(..., H, W) -> (..., h, w), no blur."""
        x = image[..., self.rows, self.cols]
        lead = x.shape[:-2]
        x = F.avg_pool2d(x.reshape(-1, 1, *x.shape[-2:]), self.pool)
        return x.reshape(*lead, *x.shape[-2:])

    def pool_blurred(self, image: torch.Tensor) -> torch.Tensor:
        return self.pool_plain(gaussian_blur(image, self.blur_sigma_px))


def gaussian_blur(image: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Separable Gaussian blur over the last two dims, zero padding."""
    radius = int(np.ceil(3 * sigma_px))
    t = torch.arange(-radius, radius + 1, dtype=image.dtype,
                     device=image.device)
    k = torch.exp(-0.5 * (t / sigma_px) ** 2)
    k = k / k.sum()
    lead = image.shape[:-2]
    x = image.reshape(-1, 1, *image.shape[-2:])
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0)), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, radius, radius)), k.view(1, 1, -1, 1))
    return x.reshape(*lead, *x.shape[-2:])


# -------------------------------------------------------------------------
# Image metrics
# -------------------------------------------------------------------------
def pearson(a: torch.Tensor, b: torch.Tensor, mask) -> float:
    a, b = a[mask], b[mask]
    a, b = a - a.mean(), b - b.mean()
    denom = torch.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 0 else float('nan')


def max_normalised(x: torch.Tensor, mask) -> torch.Tensor:
    peak = x[mask].max()
    return x / peak if peak > 0 else torch.zeros_like(x)


def ssim(a: torch.Tensor, b: torch.Tensor, mask) -> float:
    """Mean SSIM inside `mask`, on max-normalised images (L = 1)."""
    a, b = max_normalised(a, mask), max_normalised(b, mask)
    c1, c2 = 0.01 ** 2, 0.03 ** 2

    def local(x):
        return gaussian_blur(x, SSIM_WINDOW_SIGMA_PX)

    mu_a, mu_b = local(a), local(b)
    var_a = local(a * a) - mu_a ** 2
    var_b = local(b * b) - mu_b ** 2
    cov = local(a * b) - mu_a * mu_b
    s = (((2 * mu_a * mu_b + c1) * (2 * cov + c2))
         / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)))
    return float(s[mask].mean())


def mse_norm(a, b, mask) -> float:
    a, b = max_normalised(a, mask), max_normalised(b, mask)
    return float(((a - b)[mask] ** 2).mean())


def bars_contrast(percept: torch.Tensor, bars: torch.Tensor, X, Y) -> dict:
    """Per band, (on - off) / (on + off) over bar and gap pixels."""
    out = {}
    X = torch.as_tensor(X, device=percept.device)
    Y = torch.as_tensor(Y, device=percept.device)
    for y_lo, y_hi, width in BAR_BANDS:
        band = (X >= X_MIN) & (X <= X_MAX) & (Y >= y_lo) & (Y <= y_hi)
        on = percept[band & (bars > 0.5)].mean()
        off = percept[band & (bars < 0.5)].mean()
        total = on + off
        out[f'bars_contrast_{width}deg'] = (float((on - off) / total)
                                            if total > 0 else 0.0)
    return out


def test_image_metrics(percepts: dict, images: dict, grid: AnalysisGrid,
                       X, Y) -> dict:
    """Metrics for one threshold draw. `percepts`: name -> (H, W)."""
    m = grid.mask
    out = bars_contrast(percepts['bars'], images['bars'], X, Y)

    e = [grid.pool_plain(percepts[f'e{a}']) for a in (0, 90, 180, 270)]
    out['e_discrim'] = 1 - float(np.mean(
        [pearson(p, q, m) for p, q in itertools.combinations(e, 2)]))

    scores = {'pearson_r': [], 'ssim': [], 'mse_norm': []}
    for name in ('bars', 'e0', 'shapes'):
        p = grid.pool_blurred(percepts[name])
        s = grid.pool_blurred(images[name])
        scores['pearson_r'].append(pearson(p, s, m))
        scores['ssim'].append(ssim(p, s, m))
        scores['mse_norm'].append(mse_norm(p, s, m))
    out.update({k: float(np.mean(v)) for k, v in scores.items()})
    return out


def stimulation_metrics(amplitudes_ua: torch.Tensor, limit_ua: float
                        ) -> dict:
    active = amplitudes_ua > 0
    return {'frac_active': float(active.float().mean()),
            'current_active_ua': (float(amplitudes_ua[active].mean())
                                  if active.any() else 0.0),
            'frac_at_limit': float((amplitudes_ua >= 0.99 * limit_ua)
                                   .float().mean())}


# -------------------------------------------------------------------------
# Benchmarks
# -------------------------------------------------------------------------
@torch.no_grad()
def benchmark_metrics(simulator, params, thresholds) -> list:
    """One dict per threshold draw."""
    unit = require(params, 'run.current_unit_to_ua')
    ones = torch.ones(simulator.num_phosphenes, **simulator.data_kwargs)
    activations = []
    for current in THRESHOLD_SWEEP_UA:
        simulator.reset()
        simulator.update(ones * current / unit)
        activations.append(simulator.activation.get().clone())

    major, minor = {}, {}
    for current in (40.0, 100.0):
        major[current], minor[current] = \
            simulator.array.phosphene_sigma_deg(current)

    mean_ua, sd_ua = require(params, 'benchmarks.pezaris_2007.threshold_ua')
    out = []
    for threshold in thresholds:
        simulator.threshold.state = threshold
        p = []
        for activation in activations:
            simulator.activation.state = activation
            p.append(simulator.detection_probability().flatten().cpu()
                     .numpy())
        i50 = crossing(THRESHOLD_SWEEP_UA, np.stack(p), 0.5)
        median = float(np.nanmedian(i50)) if np.isfinite(i50).any() \
            else float('nan')
        row = {'threshold_ua': median,
               'threshold_z': (median - mean_ua) / sd_ua,
               'frac_never_detected': float(np.mean(~np.isfinite(i50)))}
        for current in (40.0, 100.0):
            sigma = np.sqrt(major[current] * minor[current])
            row[f'sigma_deg_{current:.0f}ua'] = float(np.nanmedian(sigma))
        out.append(row)
    return out


def crossing(x: np.ndarray, p: np.ndarray, level: float) -> np.ndarray:
    """First x at which each column of p (len(x), n) reaches `level`,
    linearly interpolated; NaN where it never does."""
    out = np.full(p.shape[1], np.nan)
    for j in range(p.shape[1]):
        above = np.nonzero(p[:, j] >= level)[0]
        if len(above) == 0:
            continue
        i = above[0]
        if i == 0 or p[i, j] == p[i - 1, j]:
            out[j] = x[i]
        else:
            t = (level - p[i - 1, j]) / (p[i, j] - p[i - 1, j])
            out[j] = x[i - 1] + t * (x[i] - x[i - 1])
    return out


# -------------------------------------------------------------------------
# Digit classifier
# -------------------------------------------------------------------------
def standardise(images: torch.Tensor) -> torch.Tensor:
    """Per image: zero mean, unit sd. Removes overall brightness, which
    a viewer adapts to and the optimiser does not control freely."""
    flat = images.flatten(1)
    flat = ((flat - flat.mean(1, keepdim=True))
            / flat.std(1, keepdim=True).clamp(min=1e-6))
    return flat.view_as(images)[:, None]


class SmallCNN(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Linear(32 * (h // 4) * (w // 4), 10)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def classifiers_for(data, args, checkpoints) -> list:
    """One classifier per repeat, loaded from its checkpoint if there is
    one: retraining is fast but on CUDA not bit-for-bit repeatable."""
    models = []
    for r in range(args.repeats):
        state = checkpoints.load(f'classifier_r{r}')
        if state is None:
            model = train_classifier(data['train_clean'],
                                     data['train_labels'], seed=r,
                                     epochs=args.epochs)
            checkpoints.save(f'classifier_r{r}', model.state_dict())
        else:
            model = SmallCNN(*data['train_clean'].shape[-2:]).to(args.device)
            model.load_state_dict(state)
            model.eval()
        models.append(model)
    return models


def train_classifier(x_train, y_train, seed, epochs) -> SmallCNN:
    """A CNN on clean digits only; it never sees a percept."""
    torch.manual_seed(seed)
    x_train = standardise(x_train)
    model = SmallCNN(*x_train.shape[-2:]).to(x_train.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x_train), generator=generator).to(
            x_train.device)
        for batch in order.split(64):
            optimiser.zero_grad()
            F.cross_entropy(model(x_train[batch]), y_train[batch]).backward()
            optimiser.step()
    model.eval()
    return model


@torch.no_grad()
def accuracy(model, images, labels) -> float:
    return float((model(standardise(images)).argmax(1) == labels)
                 .float().mean())


# -------------------------------------------------------------------------
# One condition
# -------------------------------------------------------------------------
def run_condition(name, params, data, images, classifiers, grid, X, Y, args,
                  reference_xy, checkpoints):
    """Every repeat of one condition, each checkpointed as it finishes;
    repeats already checkpointed are loaded instead of recomputed."""
    t0 = time.time()
    overrides = CONDITIONS[name]
    saved = [checkpoints.load(f'{name}__r{r}') for r in range(args.repeats)]
    for s in saved:
        if s is not None and s['overrides'] != overrides:
            sys.exit(f"Cannot resume {name!r}: its checkpoints were written "
                     f"with overrides {s['overrides']}, CONDITIONS now says "
                     f"{overrides}. Pass --fresh to start over.")
    rows = [s and s['row'] for s in saved]
    examples = saved[0] and saved[0]['examples']
    xy = next((s['xy'] for s in saved if s is not None), None)
    missing = [r for r, s in enumerate(saved) if s is None]
    if len(missing) < args.repeats:
        print(f"  {name:<22} {args.repeats - len(missing)} of "
              f"{args.repeats} repeats from checkpoints")

    if missing:
        simulator, _ = build_simulator(
            params, atlas_dir=str(REPO / require(params, 'atlas.directory')))
        stimulator = Stimulator(simulator, params, grid, args)
        thresholds = threshold_draws(simulator, args.repeats)
        check_batched(simulator, stimulator, thresholds[0])
        xy = simulator.array.vf_xy
        benchmarks = benchmark_metrics(simulator, params, thresholds)
    for r in missing:
        row = rows[r] = {'condition': name, 'repeat': r, **benchmarks[r]}
        threshold = thresholds[r]
        names = list(images)
        stacked = torch.stack([images[k] for k in names])
        percepts = dict(zip(names, stimulator(stacked, threshold)[0]))
        row.update(test_image_metrics(percepts, images, grid, X, Y))

        pooled, amplitudes, losses = [], [], []
        for part in data['test_digits'].split(args.batch):
            p, a, loss = stimulator(
                place_digits(part, X, Y, args.digit_deg, args.device),
                threshold)
            pooled.append(grid.pool_blurred(p))
            amplitudes.append(a)
            losses.append(loss)
            if r == 0 and 'digit' not in percepts:
                percepts['digit'] = p[0]
        row['digit_acc'] = accuracy(classifiers[r], torch.cat(pooled),
                                    data['test_labels'])
        row['fit_loss'] = float(torch.cat(losses).mean())
        row.update(stimulation_metrics(torch.cat(amplitudes),
                                       stimulator.limit_ua))
        row['n_electrodes'] = simulator.num_phosphenes
        if r == 0:
            examples = {k: v.cpu() for k, v in percepts.items()}
        checkpoints.save(f'{name}__r{r}', {
            'overrides': overrides, 'row': row, 'xy': xy,
            'examples': examples if r == 0 else None})
        print(f"  {name:<22} repeat {r + 1}/{args.repeats}  "
              f"{time.time() - t0:7.1f} s   digit_acc {row['digit_acc']:.3f}"
              f"   active {row['frac_active']:.2f} at "
              f"{row['current_active_ua']:.1f} uA")

    # Against the reference as it stands in this run, so loaded and
    # fresh repeats are judged the same way.
    same = (reference_xy is None
            or (xy.shape == reference_xy.shape
                and np.allclose(xy, reference_xy, equal_nan=True)))
    for row in rows:
        row['same_electrodes'] = same
    return rows, examples, xy


def ideal_rows(data, classifiers, args) -> list:
    """The classifier on the clean test digits: the ceiling."""
    return [{'condition': 'ideal_input', 'repeat': r,
             'digit_acc': accuracy(classifiers[r], data['test_clean'],
                                   data['test_labels'])}
            for r in range(args.repeats)]


# -------------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------------
SUMMARY_PANELS = [
    ('digit_acc', 'Digit accuracy, classifier trained on clean digits'),
    ('ssim', 'Percept vs image, SSIM'),
    ('e_discrim', 'Tumbling-E discriminability'),
    ('bars_contrast_1.2deg', 'Bar contrast, 1.2 deg band'),
    ('bars_contrast_0.6deg', 'Bar contrast, 0.6 deg band'),
    ('fit_loss', 'Optimiser loss on digits (lower is better)'),
    ('current_active_ua', 'Mean current of active electrodes (uA)'),
    ('threshold_ua', 'Detection threshold (uA)'),
    ('sigma_deg_100ua', 'Phosphene sigma at 100 uA (deg)'),
]


def plot_summary(summary: pd.DataFrame, path: Path, limit_ua: float):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    conditions = [c for c in summary.index if c != 'ideal_input']
    y = np.arange(len(conditions))[::-1]
    fig, axes = plt.subplots(3, 3, figsize=(13, 2.2 + 0.32 * len(conditions)
                                              * 3), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (metric, title) in zip(axes.flat, SUMMARY_PANELS):
        ax.set_facecolor(SURFACE)
        mean = summary.loc[conditions, (metric, 'mean')].to_numpy(float)
        sd = np.nan_to_num(summary.loc[conditions, (metric, 'sd')]
                           .to_numpy(float))
        if metric == 'threshold_ua':
            ax.axvspan(40 - 12, 40 + 12, color=REFERENCE_GREY, alpha=0.18,
                       lw=0)
            ax.text(40, y.max() + 0.6, 'Pezaris & Reid 40 ± 12',
                    color=INK_SECONDARY, fontsize=7, ha='center')
        if metric == 'current_active_ua':
            ax.axvline(limit_ua, color=REFERENCE_GREY, lw=1, ls=':')
            ax.text(limit_ua, y.max() + 0.6, 'safety limit',
                    color=INK_SECONDARY, fontsize=7, ha='right')
        if metric == 'digit_acc':
            ax.axvline(CHANCE, color=REFERENCE_GREY, lw=1, ls=':')
            if 'ideal_input' in summary.index:
                ceiling = summary.loc['ideal_input', (metric, 'mean')]
                ax.axvline(ceiling, color=REFERENCE_GREY, lw=1, ls=':')
                ax.text(ceiling, y.max() + 0.6, 'clean digits',
                        color=INK_SECONDARY, fontsize=7, ha='right')
        if 'baseline' in conditions:
            ax.axvline(summary.loc['baseline', (metric, 'mean')],
                       color=INK_SECONDARY, lw=1, ls='--')
        ax.errorbar(mean, y, xerr=sd, fmt='o', color=SERIES, ms=5,
                    elinewidth=2, capsize=0)
        ax.set_title(title, fontsize=9, color=INK, loc='left')
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.grid(axis='x', color='#e4e3df', lw=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#c9c8c3')
    axes[0, 0].set_yticks(y, conditions)
    fig.suptitle('Parameter comparison under optimised stimulation: mean ± '
                 'sd over threshold draws (dashed = baseline, dotted = '
                 'chance / ceiling / limit)',
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def plot_examples(examples: dict, images: dict, digit, params, path: Path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    half = require(params, 'run.view_angle') / 2
    extent = [-half, half, -half, half]
    columns = [('digit', 'MNIST digit'), ('bars', 'bars'),
               ('e0', 'tumbling E'), ('shapes', 'shapes')]
    rows = [('stimulus', {**images, 'digit': digit})] + list(examples.items())
    fig, axes = plt.subplots(len(rows), len(columns),
                             figsize=(2.3 * len(columns), 2.1 * len(rows)))
    for i, (name, row) in enumerate(rows):
        for j, (key, label) in enumerate(columns):
            ax = axes[i, j]
            ax.imshow(row[key].cpu().numpy(), cmap='gray', origin='lower',
                      vmin=0, vmax=1, extent=extent)
            ax.set(xlim=(-0.5, half), ylim=(-half / 1.3, half / 1.3),
                   xticks=[], yticks=[])
            if i == 0:
                ax.set_title(label, fontsize=9)
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# -------------------------------------------------------------------------
def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in raw.columns if c not in ('condition', 'repeat')]
    numeric = raw[['condition'] + metrics].copy()
    numeric[metrics] = numeric[metrics].astype(float)
    grouped = numeric.groupby('condition', sort=False)
    summary = pd.concat({'mean': grouped.mean(), 'sd': grouped.std()},
                        axis=1).swaplevel(axis=1)
    return summary[[(m, s) for m in metrics for s in ('mean', 'sd')]]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--conditions', default=None,
                        help='comma-separated subset of CONDITIONS')
    parser.add_argument('--n-train', type=int, default=10000,
                        help='clean digits the classifier is trained on')
    parser.add_argument('--n-test', type=int, default=500,
                        help='test digits, each optimised per repeat')
    parser.add_argument('--repeats', type=int, default=3,
                        help='threshold draws (and classifier seeds) per '
                             'condition')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps', type=int, default=60,
                        help='optimiser steps per image')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Adam learning rate on the amplitude logits')
    parser.add_argument('--batch', type=int, default=16,
                        help='images optimised side by side (GPU memory: '
                             '~2.8 GB at 16)')
    parser.add_argument('--n-electrodes', type=int, default=100)
    parser.add_argument('--max-ecc', type=float, default=6.0)
    parser.add_argument('--fps', type=float, default=10,
                        help='one frame = 1/fps s of stimulation')
    parser.add_argument('--digit-deg', type=float, default=5.0,
                        help='size of the 28 px MNIST frame, degrees')
    parser.add_argument('--quick', action='store_true',
                        help='small smoke-test run: 2000/64 digits, '
                             '2 repeats, 5 epochs, 40 steps')
    parser.add_argument('--device', default='auto',
                        help="'auto' (CUDA if available), 'cpu', 'cuda' "
                             "or 'cuda:N'")
    parser.add_argument('--download', action='store_true',
                        help='fetch MNIST into data/MNIST if missing')
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument('--resume', action='store_true',
                         help='continue from the checkpoints in --out')
    restart.add_argument('--fresh', action='store_true',
                         help='discard the checkpoints in --out first')
    parser.add_argument('--out', default=str(REPO / 'examples' / 'out'
                                             / 'compare_params'))
    args = parser.parse_args(argv)
    if args.quick:              # new defaults: explicit flags still win
        parser.set_defaults(n_train=2000, n_test=64, repeats=2, epochs=5,
                            steps=40)
        args = parser.parse_args(argv)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(args.device)
    if args.device.type == 'cuda' and not torch.cuda.is_available():
        sys.exit("CUDA requested but torch.cuda.is_available() is False; "
                 "this torch build is " + torch.__version__)
    # Before the classifiers are trained, not only once the first
    # simulator turns it on, so that two runs train identical ones.
    set_deterministic(0)

    names = (list(CONDITIONS) if args.conditions is None
             else [c.strip() for c in args.conditions.split(',')])
    unknown = [n for n in names if n not in CONDITIONS]
    if unknown:
        sys.exit(f"Unknown condition(s) {unknown}. "
                 f"Known: {', '.join(CONDITIONS)}")
    if 'baseline' in names:                 # first, so it is the reference
        names = ['baseline'] + [n for n in names if n != 'baseline']

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = base_params(args)
    X, Y = coordinate_grid(base)
    grid = AnalysisGrid(base, X, Y, args.max_ecc, args.device)
    images = test_images(X, Y, args.device)

    mnist = load_mnist(REPO / 'data' / 'MNIST', args.download)
    train_x, train_y, test_x, test_y = mnist_subset(mnist, args.n_train,
                                                    args.n_test)
    data = {'test_digits': torch.as_tensor(test_x),
            'train_labels': torch.as_tensor(train_y, dtype=torch.long,
                                            device=args.device),
            'test_labels': torch.as_tensor(test_y, dtype=torch.long,
                                           device=args.device)}
    for split, digits in (('train', train_x), ('test', test_x)):
        data[f'{split}_clean'] = clean_digits(
            torch.as_tensor(digits), X, Y, args.digit_deg, grid, args.device)
    data['example_digit'] = place_digits(test_x[:1], X, Y, args.digit_deg,
                                         args.device)[0]

    # Everything a checkpoint depends on besides its condition's
    # overrides. Not --repeats (draw r is the same whatever the count)
    # nor --conditions (each checkpoint names its own).
    run_config = {'params': base, 'device': str(args.device),
                  **{k: getattr(args, k) for k in (
                      'n_train', 'n_test', 'epochs', 'steps', 'lr', 'batch',
                      'n_electrodes', 'max_ecc', 'fps', 'digit_deg')}}
    checkpoints = Checkpoints(out / 'checkpoints', run_config, args.resume,
                              args.fresh)

    limit_ua = require(base, 'default_stim.safety_limit_ua')
    print(f"{len(names)} conditions, {args.n_test} test digits, "
          f"{args.repeats} repeats, {args.steps} steps, safety limit "
          f"{limit_ua:g} uA, on {args.device}")
    classifiers = classifiers_for(data, args, checkpoints)
    rows = ideal_rows(data, classifiers, args)
    print(f"  classifier on clean digits: "
          f"{np.mean([r['digit_acc'] for r in rows]):.3f}")
    examples, reference_xy = {}, None
    for name in names:
        params = apply_overrides(base, CONDITIONS[name])
        condition_rows, examples[name], xy = run_condition(
            name, params, data, images, classifiers, grid, X, Y, args,
            reference_xy, checkpoints)
        reference_xy = xy if reference_xy is None else reference_xy
        rows.extend(condition_rows)

    raw = pd.DataFrame(rows)
    order = names + ['ideal_input']
    raw['condition'] = pd.Categorical(raw['condition'], order, ordered=True)
    raw = raw.sort_values(['condition', 'repeat'])
    raw.to_csv(out / 'metrics_raw.csv', index=False)
    summary = summarise(raw.astype({'condition': str}))
    summary = summary.loc[order]
    summary.to_csv(out / 'metrics.csv')
    plot_summary(summary, out / 'summary.png', limit_ua)
    plot_examples(examples, images, data['example_digit'], base,
                  out / 'examples.png')

    if not raw.loc[raw['condition'] != 'ideal_input',
                   'same_electrodes'].astype(bool).all():
        print("WARNING: some conditions placed different electrodes; "
              "see the same_electrodes column.")
    print(f"\nWrote {out}")
    with pd.option_context('display.width', 200,
                           'display.max_columns', 30,
                           'display.float_format', '{:.3f}'.format):
        print(summary.xs('mean', axis=1, level=1))


if __name__ == '__main__':
    main()
