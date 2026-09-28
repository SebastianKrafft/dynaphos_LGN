"""Compare LGN simulator parameter sets, quantitatively, on the same
electrodes and the same images.

Every condition is a set of config overrides applied to one shared base
config. Each condition is built with the same `run.seed`, so the
electrode array is the same draw every time and only the parameter under
test changes. The script checks this and reports it (`same_electrodes`).

There is no ground-truth LGN percept to score against, so the metrics
fall into three groups. They are reported side by side, never combined:

1. Benchmarks against measured LGN data
     threshold_ua        median current at which p(detect) crosses 0.5,
                         one frame (1 / run.fps) of stimulation
     threshold_z         (threshold_ua - 40) / 12, Pezaris & Reid (2007)
     sigma_deg_40ua,     median phosphene sigma (geometric mean of major
     sigma_deg_100ua     and minor), degrees
2. Task performance
     acc_linear,         held-out MNIST accuracy of a linear probe and
     acc_cnn             of a small CNN, trained on the percepts
     acc_linear_amps     linear probe on the electrode amplitudes, i.e.
                         before the simulator: what the encoder kept
     bars_contrast_*     (on - off) / (on + off) of the percept over the
                         bars and gaps of each band of the bars image
     e_discrim           1 - mean pairwise correlation of the percepts of
                         the four tumbling-E orientations
     pearson_r, ssim,    percept vs stimulus, both blurred to a common
     mse_norm            analysis scale, inside the covered region only,
                         averaged over the bars, E and shapes images.
                         mse_norm uses max-normalised images: it is a
                         secondary column, for the reasons pearson_r and
                         ssim are the primary ones
3. Sensitivity
     the table itself: how far each condition moves each metric away
     from the baseline row.

The threshold noise is the model's only stochastic element, so every
metric is computed `--repeats` times, each with a fresh threshold draw
(and a fresh CNN initialisation). The draws are seeded by repeat index,
so condition A's repeat k and condition B's repeat k use the same
standard-normal draw: the comparison is paired.

A reference row, `ideal_input`, trains the same decoders on the digit
images themselves, pooled to the same grid: the ceiling for any
condition. Chance is 0.10.

Output (default examples/out/compare_params/):
    metrics.csv        mean and sd per condition
    metrics_raw.csv    one row per condition and repeat
    summary.png        one panel per metric, conditions against baseline
    examples.png       what each condition's percepts look like

Usage::

    python examples/compare_params.py --quick           # smoke test
    python examples/compare_params.py                   # all conditions
    python examples/compare_params.py --conditions baseline,K_x2

MNIST is read from data/MNIST/ as the four original .gz files. Pass
--download to fetch them there (~11.6 MB) if they are missing.

The operating point is --max-current-ua (default 60 uA), NOT the
shipped `sampling.stimulus_scale` of 100 uA. At 100 uA with 100
electrodes the percepts clamp to one white blob and every condition
looks the same (bar contrast ~0 in every band), so the parameters
under test would have nothing to act on. `stim_100ua` keeps the shipped
value in the table as its own condition.

NOTE `current_spread.aggregation` is not in the default conditions: the
simulator's forward pass does not call `aggregate_activation`, so
sweeping it would show no effect at all.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import itertools
import os
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

from dynaphos.utils import load_params                              # noqa: E402
from dynaphos_lgn.build import build_simulator                      # noqa: E402
from dynaphos_lgn.params import require                             # noqa: E402
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
    'K_x0.5': {'current_spread.k_ua_per_mm2': 337.5},
    'K_x2': {'current_spread.k_ua_per_mm2': 1350.0},
    'bosking_spread': {'current_spread.spread_model': 'bosking'},
    # The four magnification models (baseline is 'jacobian').
    'mag_atlas_gradient': {'magnification.model': 'atlas_gradient'},
    'mag_anisotropic': {'magnification.model': 'anisotropic_gradient'},
    'mag_malpeli_density': {'magnification.model': 'malpeli_density'},
    # Threshold form.
    'hard_threshold': {'thresholding.soft_threshold': False},
    'soft_no_baseline': {'thresholding.soft_threshold_baseline': 'none'},
    # Encoder: nearest pixel instead of the receptive-field centre.
    'center_sampling': {'sampling.sampling_method': 'center'},
    # The shipped operating point. Saturates: see --max-current-ua.
    'stim_100ua': {'sampling.stimulus_scale': 1.0e-4},
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

# Analysis scale. Percepts and stimuli are blurred by ANALYSIS_BLUR_DEG
# before being compared, and pooled to ANALYSIS_PIXEL_DEG for decoding
# and image metrics. Fixed across conditions, never derived from them.
ANALYSIS_BLUR_DEG = 0.5
ANALYSIS_PIXEL_DEG = 0.25
SSIM_WINDOW_SIGMA_PX = 1.5
CHANCE = 0.1

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
    params['sampling']['stimulus_scale'] = (
        args.max_current_ua / require(params, 'run.current_unit_to_ua'))
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


def placed(digits: torch.Tensor, X, Y, digit_deg, device=None, chunk=256):
    """Yield placed (H, W) digit images one at a time."""
    for part in digits.split(chunk):
        yield from place_digits(part, X, Y, digit_deg, device)


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
# Rendering
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


@torch.no_grad()
def render(simulator, image: torch.Tensor, thresholds: list):
    """Percepts of one static image, one frame from rest, for each
    threshold draw -> ((R, H, W) percepts, (n,) amplitudes).

    The thresholds only gate intensity, so the costly spatial maps are
    rendered once and reused across draws. This is `simulator(...)`
    written out; `check_render` confirms the two agree.
    """
    simulator.reset()
    amplitude = simulator.sample_stimulus(image, rescale=True)
    simulator.update(amplitude)
    field = simulator.brightness.get() * simulator.spatial_activation()
    percepts = []
    for threshold in thresholds:
        simulator.threshold.state = threshold
        percepts.append(torch.sum(simulator.detection_probability() * field,
                                  dim=0).clamp(0, 1))
    return torch.stack(percepts), amplitude.flatten()


@torch.no_grad()
def check_render(simulator, image, thresholds):
    fast, _ = render(simulator, image, thresholds[:1])
    simulator.reset()
    simulator.threshold.state = thresholds[0]
    slow = simulator(simulator.sample_stimulus(image, rescale=True))
    if not torch.allclose(fast[0], slow, atol=1e-5):
        raise RuntimeError("render() no longer matches simulator(); the "
                           "simulator's forward pass has changed.")


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
# Decoders
# -------------------------------------------------------------------------
def standardise(train: torch.Tensor, test: torch.Tensor, per_feature: bool):
    dims = 0 if per_feature else None
    mean = train.mean(dim=dims) if dims is not None else train.mean()
    std = train.std(dim=dims) if dims is not None else train.std()
    std = torch.clamp(std, min=1e-6)
    return (train - mean) / std, (test - mean) / std


def linear_probe(x_train, y_train, x_test, y_test, steps=300) -> float:
    """Multinomial logistic regression, full batch, zero init."""
    x_train, x_test = standardise(x_train.flatten(1), x_test.flatten(1),
                                  per_feature=True)
    model = nn.Linear(x_train.shape[1], 10).to(x_train.device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.02,
                                 weight_decay=1e-4)
    for _ in range(steps):
        optimiser.zero_grad()
        F.cross_entropy(model(x_train), y_train).backward()
        optimiser.step()
    with torch.no_grad():
        return float((model(x_test).argmax(1) == y_test).float().mean())


class SmallCNN(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Linear(32 * (h // 4) * (w // 4), 10)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def cnn(x_train, y_train, x_test, y_test, seed, epochs) -> float:
    torch.manual_seed(seed)
    x_train, x_test = standardise(x_train, x_test, per_feature=False)
    x_train, x_test = x_train[:, None], x_test[:, None]
    model = SmallCNN(*x_train.shape[-2:]).to(x_train.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator).to(
            x_train.device)
        for batch in order.split(64):
            optimiser.zero_grad()
            F.cross_entropy(model(x_train[batch]), y_train[batch]).backward()
            optimiser.step()
    model.eval()
    with torch.no_grad():
        return float((model(x_test).argmax(1) == y_test).float().mean())


# -------------------------------------------------------------------------
# One condition
# -------------------------------------------------------------------------
def run_condition(name, params, data, images, grid, X, Y, args,
                  reference_xy):
    t0 = time.time()
    simulator, _ = build_simulator(
        params, atlas_dir=str(REPO / require(params, 'atlas.directory')))
    thresholds = threshold_draws(simulator, args.repeats)
    check_render(simulator, images['shapes'], thresholds)

    xy = simulator.array.vf_xy
    same = (reference_xy is None
            or (xy.shape == reference_xy.shape
                and np.allclose(xy, reference_xy, equal_nan=True)))
    unsampled = int((simulator.sampling_mask.flatten(1).sum(1) == 0).sum())

    rows = [{'condition': name, 'repeat': r} for r in range(args.repeats)]
    for row, bench in zip(rows, benchmark_metrics(simulator, params,
                                                  thresholds)):
        row.update(bench)

    rendered = {k: render(simulator, v, thresholds)[0]
                for k, v in images.items()}
    for r, row in enumerate(rows):
        row.update(test_image_metrics({k: v[r] for k, v in rendered.items()},
                                      images, grid, X, Y))

    decoded = {}
    for split in ('train', 'test'):
        percepts, amplitudes = [], []
        for image in placed(data[f'{split}_digits'], X, Y, args.digit_deg,
                            args.device):
            p, a = render(simulator, image, thresholds)
            percepts.append(grid.pool_plain(p))
            amplitudes.append(a)
        decoded[split] = (torch.stack(percepts, dim=1),     # (R, N, h, w)
                          torch.stack(amplitudes))          # (N, n)
    y_train, y_test = data['train_labels'], data['test_labels']
    acc_amps = linear_probe(decoded['train'][1], y_train,
                            decoded['test'][1], y_test)
    for r, row in enumerate(rows):
        x_train, x_test = decoded['train'][0][r], decoded['test'][0][r]
        row['acc_linear'] = linear_probe(x_train, y_train, x_test, y_test)
        row['acc_cnn'] = cnn(x_train, y_train, x_test, y_test, seed=r,
                             epochs=args.epochs)
        row['acc_linear_amps'] = acc_amps
        row['n_electrodes'] = simulator.num_phosphenes
        row['n_unsampled_electrodes'] = unsampled
        row['same_electrodes'] = same

    examples = {k: v[0] for k, v in rendered.items()}
    examples['digit'] = render(simulator, data['example_digit'],
                               thresholds[:1])[0][0]
    print(f"  {name:<22} {time.time() - t0:6.1f} s   "
          f"acc_cnn {np.mean([r['acc_cnn'] for r in rows]):.3f}   "
          f"threshold {rows[0]['threshold_ua']:.1f} uA")
    return rows, examples, xy


def ideal_rows(data, grid, args) -> list:
    """The same decoders on the digit images: the ceiling."""
    x_train, x_test = data['train_ideal'], data['test_ideal']
    y_train, y_test = data['train_labels'], data['test_labels']
    linear = linear_probe(x_train, y_train, x_test, y_test)
    return [{'condition': 'ideal_input', 'repeat': r, 'acc_linear': linear,
             'acc_cnn': cnn(x_train, y_train, x_test, y_test, seed=r,
                            epochs=args.epochs)}
            for r in range(args.repeats)]


# -------------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------------
SUMMARY_PANELS = [
    ('acc_cnn', 'MNIST accuracy, CNN'),
    ('acc_linear', 'MNIST accuracy, linear probe'),
    ('acc_linear_amps', 'Accuracy from amplitudes (encoder only)'),
    ('pearson_r', 'Percept vs stimulus, Pearson r'),
    ('ssim', 'Percept vs stimulus, SSIM'),
    ('e_discrim', 'Tumbling-E discriminability'),
    ('bars_contrast_0.6deg', 'Bar contrast, 0.6 deg band'),
    ('threshold_ua', 'Detection threshold (uA)'),
    ('sigma_deg_100ua', 'Phosphene sigma at 100 uA (deg)'),
]


def plot_summary(summary: pd.DataFrame, path: Path):
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
        if metric in ('acc_cnn', 'acc_linear', 'acc_linear_amps'):
            ax.axvline(CHANCE, color=REFERENCE_GREY, lw=1, ls=':')
            if metric != 'acc_linear_amps' and 'ideal_input' in summary.index:
                ceiling = summary.loc['ideal_input', (metric, 'mean')]
                ax.axvline(ceiling, color=REFERENCE_GREY, lw=1, ls=':')
                ax.text(ceiling, y.max() + 0.6, 'ideal input',
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
    fig.suptitle('Parameter comparison: mean ± sd over threshold draws '
                 '(dashed = baseline, dotted = chance / ceiling)',
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
    parser.add_argument('--n-train', type=int, default=2000)
    parser.add_argument('--n-test', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=3,
                        help='threshold draws (and CNN seeds) per condition')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--n-electrodes', type=int, default=100)
    parser.add_argument('--max-ecc', type=float, default=6.0)
    parser.add_argument('--fps', type=float, default=10,
                        help='one frame = 1/fps s of stimulation')
    parser.add_argument('--max-current-ua', type=float, default=60.0,
                        help='current for a white pixel (sets '
                             'sampling.stimulus_scale). The shipped 100 uA '
                             'saturates 100 electrodes into one blob')
    parser.add_argument('--digit-deg', type=float, default=5.0,
                        help='size of the 28 px MNIST frame, degrees')
    parser.add_argument('--quick', action='store_true',
                        help='small smoke-test run: 400/200 digits, '
                             '2 repeats, 8 epochs')
    parser.add_argument('--device', default='auto',
                        help="'auto' (CUDA if available), 'cpu', 'cuda' "
                             "or 'cuda:N'")
    parser.add_argument('--download', action='store_true',
                        help='fetch MNIST into data/MNIST if missing')
    parser.add_argument('--out', default=str(REPO / 'examples' / 'out'
                                             / 'compare_params'))
    args = parser.parse_args(argv)
    if args.quick:
        args.n_train, args.n_test, args.repeats, args.epochs = 400, 200, 2, 8
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(args.device)
    if args.device.type == 'cuda' and not torch.cuda.is_available():
        sys.exit("CUDA requested but torch.cuda.is_available() is False; "
                 "this torch build is " + torch.__version__)

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
    data = {'train_digits': torch.as_tensor(train_x),
            'test_digits': torch.as_tensor(test_x),
            'train_labels': torch.as_tensor(train_y, dtype=torch.long,
                                            device=args.device),
            'test_labels': torch.as_tensor(test_y, dtype=torch.long,
                                           device=args.device)}
    for split in ('train', 'test'):
        data[f'{split}_ideal'] = torch.stack([
            grid.pool_plain(image) for image
            in placed(data[f'{split}_digits'], X, Y, args.digit_deg,
                      args.device)])
    data['example_digit'] = place_digits(test_x[:1], X, Y, args.digit_deg,
                                         args.device)[0]

    print(f"{len(names)} conditions, {args.n_train}/{args.n_test} digits, "
          f"{args.repeats} repeats, on {args.device}")
    rows = ideal_rows(data, grid, args)
    examples, reference_xy = {}, None
    for name in names:
        params = apply_overrides(base, CONDITIONS[name])
        condition_rows, examples[name], xy = run_condition(
            name, params, data, images, grid, X, Y, args, reference_xy)
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
    plot_summary(summary, out / 'summary.png')
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
