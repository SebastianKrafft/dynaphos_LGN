"""What koniocellular (K) layers change in the LGN simulator.

Writes three figures:

1. ``konio_layers.png`` -- a coronal atlas slice with M, P and K1-K6.
2. ``konio_recruitment.png`` -- the K share of recruited cells against
   current, per electrode.
3. ``konio_percepts.png`` -- class-mixture percepts without K, with K,
   and with K split into ventral/dorsal, plus the difference K makes.

With the real atlas, the first run fits the K-inclusive Jacobian
(~90 s, ~4 GB RAM) and caches it.

Usage::

    python examples/demo_konio_lgn.py
    python examples/demo_konio_lgn.py --synthetic
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynaphos.utils import load_params                              # noqa: E402
from dynaphos_lgn.build import build_simulator                      # noqa: E402
from dynaphos_lgn.konio import DV_AXIS                              # noqa: E402
from dynaphos_lgn.params import require                             # noqa: E402

REPO = Path(__file__).resolve().parents[1]
AMPLITUDE_UA = 80.0
PERCEPTS = {
    'M + P': ['magno', 'parvo'],
    'M + P + K': ['magno', 'parvo', 'konio'],
    'M + P + K ventral/dorsal': ['magno', 'parvo', 'konio_ventral',
                                 'konio_dorsal'],
}


def build(params, classes, args):
    params = copy.deepcopy(params)
    params['percept']['classes'] = classes
    simulator, _ = build_simulator(params, atlas_dir=args.atlas_dir,
                                   synthetic=args.synthetic,
                                   rng=np.random.default_rng(0))
    return simulator


def plot_layers(atlas, params, path: Path):
    """The coronal slice holding the most K voxels."""
    k_codes = require(params, 'atlas.layers.classes.konio')
    k_per_slice = np.isin(atlas.layer, k_codes).sum(axis=(0, 1))
    layer = atlas.layer[:, :, int(np.argmax(k_per_slice))]
    names = require(params, 'atlas.layers.names')
    codes = sorted(names)
    colours = (['#ffffff'] + ['#d95f02'] * 2 + ['#1b9e77'] * 2
               + list(plt.get_cmap('Purples')(np.linspace(0.4, 1, 6))))
    image = np.searchsorted([0] + codes, layer)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.moveaxis(image, DV_AXIS, 0), origin='lower',
              cmap=ListedColormap(colours), vmin=0, vmax=len(codes),
              interpolation='nearest')
    ax.set_xlabel('ML (voxels)')
    ax.set_ylabel('DV (voxels, dorsal up)')
    ax.set_title('Magno (orange), parvo (green), K1-K6 (light to dark)')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_recruitment(simulator, path: Path):
    """K share of the recruited cells against current."""
    currents = np.linspace(5, 100, 20)
    shares = []
    for current in currents:
        simulator.reset()
        simulator(torch.full((simulator.num_phosphenes,), current * 1e-6))
        shares.append(simulator.class_fraction_table()['konio'])
    shares = np.array(shares)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(currents, shares, color='0.6', lw=0.8)
    ax.plot(currents, np.median(shares, axis=1), color='k', lw=2,
            label='median electrode')
    ax.set_xlabel('current (uA)')
    ax.set_ylabel('K share of recruited cells')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_percepts(simulators: dict, path: Path):
    images = {}
    for name, simulator in simulators.items():
        simulator.reset()
        amplitude = torch.full((simulator.num_phosphenes,),
                               AMPLITUDE_UA * 1e-6)
        images[name] = simulator(amplitude).detach().numpy()
    images['difference (M+P+K minus M+P)'] = (images['M + P + K']
                                              - images['M + P'])

    half = require(next(iter(simulators.values())).params,
                   'run.view_angle') / 2
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    for ax, (name, image) in zip(axes, images.items()):
        if name.startswith('difference'):
            limit = max(float(np.abs(image).max()), 1e-6)
            style = dict(cmap='RdBu_r', vmin=-limit, vmax=limit)
        else:
            style = dict(cmap='gray', vmin=0, vmax=1)
        ax.imshow(image, origin='lower', extent=[-half, half, -half, half],
                  **style)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('deg')
    fig.suptitle(f'Class-mixture percepts, all electrodes at '
                 f'{AMPLITUDE_UA:.0f} uA')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params',
                        default=str(REPO / 'config' / 'params_lgn.yaml'))
    parser.add_argument('--atlas-dir',
                        default=str(REPO / 'data' / 'Erwin_Atlas'))
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--n-electrodes', type=int, default=24)
    parser.add_argument('--out-dir', default=str(REPO / 'examples' / 'out'))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = load_params(args.params)
    params['run']['gpu'] = None
    params['konio']['enabled'] = True
    params['percept']['model'] = 'class_mixture'
    params['electrodes']['n_electrodes'] = args.n_electrodes
    params['electrodes']['cell_class'] = ['magno', 'parvo']
    params['electrodes']['max_eccentricity_deg'] = min(
        params['electrodes']['max_eccentricity_deg'],
        0.4 * require(params, 'run.view_angle'))

    simulators = {name: build(params, classes, args)
                  for name, classes in PERCEPTS.items()}
    with_k = simulators['M + P + K']

    plot_layers(with_k.array.atlas, params, out_dir / 'konio_layers.png')
    plot_recruitment(with_k, out_dir / 'konio_recruitment.png')
    plot_percepts(simulators, out_dir / 'konio_percepts.png')
    print(f"Figures written to {out_dir}.")


if __name__ == '__main__':
    main()
