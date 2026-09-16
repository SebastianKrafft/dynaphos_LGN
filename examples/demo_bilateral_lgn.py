"""Whole-field percepts: both LGNs instead of one.

The published Erwin et al. (1999) atlas is a single left LGN, and one
LGN represents the contralateral hemifield -- so with
``atlas.hemispheres: [left]`` the left half of every rendered frame is
empty, no matter what you feed the simulator. Setting

    atlas:
      hemispheres: [left, right]

adds a second nucleus, built by reflecting the first across the midline,
and the percept covers the whole field.

This writes one figure, ``bilateral_lgn_demo.png``:

1. where the electrodes sit, coloured by nucleus;
2. the target image, which spans both hemifields;
3. what one nucleus perceives of it;
4. what two perceive.

Panels 3 and 4 are the comparison worth looking at, and it is a
controlled one: panel 3 renders the left nucleus of the very same
bilateral simulator, on its own. Same image, same electrodes, same
seeds -- the only difference is the second nucleus.

Usage::

    python examples/demo_bilateral_lgn.py                 # real atlas if present
    python examples/demo_bilateral_lgn.py --synthetic     # force the stand-in
    python examples/demo_bilateral_lgn.py --n-electrodes 32

**What the left half of panel 4 is.** The left atlas reflected, not a
second reconstruction. It assumes the two nuclei are mirror images,
which is the textbook first approximation and nothing more; see
`dynaphos_lgn.atlas.MirroredAtlas`. Without the real atlas this falls
back to the synthetic stand-in and says so on the figure -- that path
exercises the code and is not a result either way.
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
from dynaphos_lgn.params import require                             # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def target_image(params) -> np.ndarray:
    """A pattern that deliberately straddles the vertical meridian.

    Two thick rings and a horizontal bar, drawn in degrees of visual
    angle against the same `run.resolution` / `run.view_angle` /
    `run.origin` the simulator renders in, so image and electrode
    coordinates are in register.

    The rings are deliberately wide -- a couple of degrees, several
    phosphenes across. A few dozen electrodes spread over a 32 deg field
    sit some degrees apart, so a ring thinner than that spacing would be
    missed by most of them, and the figure would be measuring how lucky
    the sampling was rather than how much field the array covers.

    Nothing here is centred in one hemifield, which is the point: a
    one-nucleus array should visibly lose half of it.
    """
    res_x, res_y = require(params, 'run.resolution')
    x_origin, y_origin = require(params, 'run.origin')
    half = require(params, 'run.view_angle') / 2
    x = np.linspace(x_origin - half, x_origin + half, res_x)
    y = np.linspace(y_origin - half, y_origin + half, res_y)
    grid_x, grid_y = np.meshgrid(x, y)
    radius = np.hypot(grid_x, grid_y)

    image = np.zeros((res_y, res_x), dtype=np.float32)
    for ring, thickness in ((0.35 * half, 0.11 * half),
                            (0.80 * half, 0.11 * half)):
        image[np.abs(radius - ring) < thickness] = 1.0
    image[np.abs(grid_y) < 0.09 * half] = 1.0
    return image


def percept_of(simulator, image: np.ndarray) -> np.ndarray:
    """Sample an image with the array and render what it produces."""
    simulator.reset()
    amplitude = simulator.sample_stimulus(image, rescale=True)
    return simulator(amplitude).detach().numpy()


def percept_of_one_nucleus(bilateral, name: str,
                           image: np.ndarray) -> np.ndarray:
    """What the bilateral array would perceive with one nucleus silent.

    Renders the sub-simulator directly rather than building a separate
    one-nucleus simulator: a second build would place its electrodes
    from a different draw, and the comparison would then be measuring
    that instead of the missing hemifield.
    """
    return percept_of(bilateral.simulators[name], image)


def banner(simulator, parts) -> str:
    atlases = ([parts['atlas']] if 'atlas' in parts
               else [p['atlas'] for p in parts['hemispheres'].values()])
    if any(getattr(a, 'is_synthetic', False) for a in atlases):
        return 'SYNTHETIC ATLAS -- code demonstration, not a result'
    return 'Erwin et al. (1999) macaque LGN atlas, left nucleus and its mirror'


def plot(bilateral, image, out_path: Path, params, parts):
    half = require(params, 'run.view_angle') / 2
    extent = [-half, half, -half, half]

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.4))

    ax = axes[0]
    colours = {'left': 'tab:blue', 'right': 'tab:red'}
    for name, span in bilateral.electrode_slices.items():
        xy = bilateral.vf_xy[span]
        ax.scatter(xy[:, 0], xy[:, 1], c=colours[name], s=45,
                   edgecolor='k', linewidth=0.5,
                   label=f'{name} LGN ({span.stop - span.start})')
    ax.axvline(0, color='0.6', lw=1.0, ls='--')
    ax.set_aspect('equal')
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_xlabel('visual field x (deg)')
    ax.set_ylabel('visual field y (deg)')
    ax.set_title('Electrodes, by nucleus\n(dashed: vertical meridian)')
    ax.legend(fontsize=8, loc='upper right')

    for ax, picture, title in (
            (axes[1], image, 'Target image'),
            (axes[2], percept_of_one_nucleus(bilateral, 'left', image),
             'Percept, left LGN alone\n(right hemifield only)'),
            (axes[3], percept_of(bilateral, image),
             'Percept, both nuclei\n(whole field)')):
        ax.imshow(picture, cmap='gray', origin='lower', vmin=0, vmax=1,
                  extent=extent)
        ax.axvline(0, color='tab:red', lw=0.8, alpha=0.4)
        ax.set_xlabel('visual field x (deg)')
        ax.set_title(title)

    fig.suptitle(banner(bilateral, parts), fontsize=9)
    # Leave the top strip to the banner: it names the atlas, and a
    # figure that quietly overlaps it with a panel title is a figure
    # that stops saying whether its numbers are real.
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--params',
                        default=str(REPO / 'config' / 'params_lgn.yaml'))
    parser.add_argument('--atlas-dir', default=None)
    parser.add_argument('--synthetic', action='store_true',
                        help='force the synthetic stand-in atlas')
    parser.add_argument('--n-electrodes', type=int, default=40,
                        help='per nucleus')
    parser.add_argument('--view-angle', type=float, default=32.0,
                        help='rendered field of view, degrees. Wider than '
                             'the shipped 16, because there is now twice '
                             'as much field to show')
    parser.add_argument('--out-dir', default=str(REPO / 'examples' / 'out'))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = load_params(args.params)
    params['run']['view_angle'] = args.view_angle
    params['electrodes']['n_electrodes'] = args.n_electrodes
    params['electrodes']['max_eccentricity_deg'] = args.view_angle / 2

    # One build, two readings of it: the whole bilateral array, and its
    # left nucleus on its own. Building a separate one-nucleus simulator
    # would redraw the electrode positions and confound the comparison.
    bilateral, parts = build_simulator(params, atlas_dir=args.atlas_dir,
                                       synthetic=args.synthetic,
                                       hemispheres=['left', 'right'])
    print(bilateral.describe())
    print()

    image = target_image(params)
    out_path = out_dir / 'bilateral_lgn_demo.png'
    plot(bilateral, image, out_path, params, parts)
    print(f"Wrote {out_path}.")

    covered = bilateral.vf_xy[:, 0]
    print(f"\nVisual field reached: x from {covered.min():+.1f} to "
          f"{covered.max():+.1f} deg "
          f"({int((covered < 0).sum())} electrodes left of the meridian, "
          f"{int((covered >= 0).sum())} right).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
