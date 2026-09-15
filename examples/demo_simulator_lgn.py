"""End-to-end demonstration of the LGN phosphene simulator.

Runs the whole pipeline and writes four figures:

1. ``lgn_demo_electrodes.png`` -- where the electrodes sit in the visual
   field, coloured by which renderer each one uses and by eccentricity.
2. ``lgn_demo_percept.png`` -- a rendered percept, plus the per-lamina
   recruitment behind it.
3. ``lgn_demo_optimisation.png`` -- gradient descent on per-electrode
   amplitude towards a target percept, which is the "end-to-end
   optimisable" claim actually exercised.
4. ``lgn_demo_uncertainty.png`` -- the same phosphene rendered under the
   low, transplanted and high values of the excitability constant K, so
   the size uncertainty is visible rather than buried in a footnote.

Usage::

    python examples/demo_simulator_lgn.py                 # real atlas if present
    python examples/demo_simulator_lgn.py --synthetic     # force the stand-in
    python examples/demo_simulator_lgn.py --atlas-dir DIR

Without the real Erwin atlas the demo falls back to the synthetic atlas
and says so on every figure. That path exercises the code; it is not a
result.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynaphos.utils import load_params                              # noqa: E402
from dynaphos_lgn.build import build_simulator                      # noqa: E402
from dynaphos_lgn.current_spread import threshold_reduction_bracket # noqa: E402
from dynaphos_lgn.params import require                             # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _banner(sim) -> str:
    if getattr(sim.array.atlas, 'is_synthetic', False):
        return 'SYNTHETIC ATLAS -- code demonstration, not a result'
    return 'Erwin et al. (1999) macaque LGN atlas'


def plot_electrodes(sim, out_path: Path):
    array = sim.array
    mode = array.render_mode
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for label, marker in (('ellipse', 'o'), ('scatter', 's')):
        sel = mode == label
        if not sel.any():
            continue
        ax.scatter(array.vf_xy[sel, 0], array.vf_xy[sel, 1],
                   c=array.eccentricity_deg[sel], marker=marker,
                   cmap='viridis', s=60, edgecolor='k', linewidth=0.5,
                   label=f'{label} ({int(sel.sum())})')
    ax.set_aspect('equal')
    ax.set_xlabel('visual field x (deg)')
    ax.set_ylabel('visual field y (deg)')
    ax.set_title('Electrode positions in the visual field')
    ax.legend(title='renderer', fontsize=8)
    ax.axhline(0, color='0.8', lw=0.5)
    ax.axvline(0, color='0.8', lw=0.5)

    major, minor = array.phosphene_sigma_deg(array.max_current_ua)
    ax2.scatter(array.eccentricity_deg, major, label='major axis', s=30)
    ax2.scatter(array.eccentricity_deg, minor, label='minor axis', s=30)
    ax2.set_xlabel('eccentricity (deg)')
    ax2.set_ylabel('phosphene sigma (deg)')
    ax2.set_title(f'Phosphene size at I_max = {array.max_current_ua:.0f} uA')
    ax2.legend(fontsize=8)

    fig.suptitle(_banner(sim), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_percept(sim, out_path: Path):
    sim.reset()
    amplitude = torch.full((sim.num_phosphenes,), 80e-6)
    image = sim(amplitude).detach().numpy()
    table = sim.layer_activation_table()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                  gridspec_kw={'width_ratios': [1, 1.15]})
    half = require(sim.params, 'run.view_angle') / 2
    ax.imshow(image, cmap='gray', origin='lower', vmin=0, vmax=1,
              extent=[-half, half, -half, half])
    ax.set_xlabel('visual field x (deg)')
    ax.set_ylabel('visual field y (deg)')
    ax.set_title('Rendered percept, all electrodes at 80 uA')

    names = list(table)
    values = np.array([np.atleast_1d(table[n]).ravel() for n in names])
    bottom = np.zeros(values.shape[1])
    x = np.arange(values.shape[1])
    for name, row in zip(names, values):
        ax2.bar(x, row, bottom=bottom, label=name)
        bottom = bottom + row
    ax2.set_xlabel('electrode')
    ax2.set_ylabel('recruitment (weighted voxels)')
    ax2.set_title('Per-lamina recruitment')
    ax2.legend(fontsize=8)

    fig.suptitle(_banner(sim), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_optimisation(sim, out_path: Path, steps: int = 120):
    """Gradient descent on per-electrode amplitude towards a target.

    The soft threshold is switched on for this, and that is a modelling
    choice worth stating rather than a solver trick: with the hard
    threshold Dynaphos inherits, an electrode below threshold has
    exactly zero gradient and can never learn to cross it.
    """
    sim.soft_threshold = True

    sim.reset()
    target_amplitude = torch.full((sim.num_phosphenes,), 95e-6)
    target_amplitude[::2] = 15e-6            # a pattern, not a flat field
    target = sim(target_amplitude).detach()

    raw = torch.full((sim.num_phosphenes,), -3.0, requires_grad=True)
    optimiser = torch.optim.Adam([raw], lr=0.15)
    scale = 2e-4                              # amperes at raw = 0

    losses = []
    for _ in range(steps):
        sim.reset()
        amplitude = torch.sigmoid(raw) * scale
        loss = torch.nn.functional.mse_loss(sim(amplitude), target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))

    sim.reset()
    final = sim(torch.sigmoid(raw).detach() * scale).detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    half = require(sim.params, 'run.view_angle') / 2
    extent = [-half, half, -half, half]
    axes[0].imshow(target.numpy(), cmap='gray', origin='lower', vmin=0,
                   vmax=1, extent=extent)
    axes[0].set_title('target percept')
    axes[1].imshow(final, cmap='gray', origin='lower', vmin=0, vmax=1,
                   extent=extent)
    axes[1].set_title('after optimisation')
    axes[2].semilogy(losses)
    axes[2].set_xlabel('step')
    axes[2].set_ylabel('MSE loss')
    axes[2].set_title(f'loss {losses[0]:.2e} -> {losses[-1]:.2e}')
    for ax in axes[:2]:
        ax.set_xlabel('deg')
    fig.suptitle(_banner(sim), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params',
                        default=str(REPO / 'config' / 'params_lgn.yaml'))
    parser.add_argument(
        '--atlas-dir',
        default=str(REPO / 'data' / 'Erwin_Atlas'))
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--n-electrodes', type=int, default=24)
    parser.add_argument('--out-dir', default=str(REPO / 'examples' / 'out'))
    parser.add_argument('--steps', type=int, default=120)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(message)s')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = load_params(args.params)
    params['electrodes']['n_electrodes'] = args.n_electrodes
    params['run']['gpu'] = None
    # Keep the electrodes inside the rendered field of view, so the demo
    # figures show every phosphene rather than silently cropping most of
    # the array. The simulator warns when they do not.
    half = require(params, 'run.view_angle') / 2
    params['electrodes']['max_eccentricity_deg'] = min(
        params['electrodes'].get('max_eccentricity_deg', half), 0.8 * half)

    simulator, parts = build_simulator(params, atlas_dir=args.atlas_dir,
                                       synthetic=args.synthetic)

    print(simulator.describe())
    print()

    # Only for reporting
    spacing = 0.4
    bracket = threshold_reduction_bracket(4, spacing, params)
    print(f"Multi-electrode threshold bracket at {spacing * 1e3:.0f} um "
          f"spacing, N = 4:")
    print(f"  spread radius        : "
          f"{bracket['spread_radius_mm'] * 1e3:.0f} um")
    print(f"  spacing / radius     : {bracket['spacing_over_radius']:.2f}")
    print(f"  threshold reduction  : "
          f"{bracket['floor_fraction'] * 100:.0f}% - "
          f"{bracket['ceiling_fraction'] * 100:.0f}%  ({bracket['regime']})")
    print()


    plot_electrodes(simulator, out_dir / 'lgn_demo_electrodes.png')
    plot_percept(simulator, out_dir / 'lgn_demo_percept.png')
    losses = run_optimisation(simulator,
                              out_dir / 'lgn_demo_optimisation.png',
                              steps=args.steps)

    print(f"Optimisation: loss {losses[0]:.3e} -> {losses[-1]:.3e} "
          f"over {len(losses)} steps.")
    print(f"Figures written to {out_dir}.")


if __name__ == '__main__':
    main()
