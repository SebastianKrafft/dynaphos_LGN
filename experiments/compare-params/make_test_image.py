"""Build a test image whose structure sits in the right hemifield, the
half one (left) LGN covers, on the simulator's own pixel grid.

Patterns
    bars    three bands of vertical bars at 1.2, 0.6 and 0.3 deg -- a
            resolution ladder. Read off which band still separates.
    e       a tumbling E, 5 deg on a side. Orientation discrimination.
    shapes  disc, open square and a bar: is shape recoverable at all?

Usage::

    python experiments/compare-params/make_test_image.py                       # bars
    python experiments/compare-params/make_test_image.py --pattern e
    python experiments/compare-params/make_test_image.py --pattern bars --simulate

`--simulate` also writes a stimulus/percept comparison. The PNG is
flipped so it looks upright; reload it with
``cv2.imread(path, cv2.IMREAD_GRAYSCALE)[::-1]``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dynaphos.utils import load_params                              # noqa: E402
from dynaphos_lgn.params import require                             # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# The region the patterns are drawn in, degrees
X_MIN, X_MAX = 0.5, 6.0
Y_MAX = 5.5

# (y_lo, y_hi, bar width) in degrees, coarse to fine
BAR_BANDS = [(2.0, Y_MAX, 1.2), (-1.75, 1.75, 0.6), (-Y_MAX, -2.25, 0.3)]


def coordinate_grid(params):
    """(X, Y) pixel-centre coordinates in degrees, row 0 = bottom, as in
    `dynaphos_lgn.rendering.visual_field_grid`."""
    res_x, res_y = require(params, 'run.resolution')
    x_org, y_org = require(params, 'run.origin')
    half = require(params, 'run.view_angle') / 2
    x = np.linspace(x_org - half, x_org + half, res_x)
    y = np.linspace(y_org - half, y_org + half, res_y)
    return np.meshgrid(x, y)


def pattern_bars(X, Y):
    """Three bands of vertical bars, coarse to fine."""
    image = np.zeros(X.shape)
    inside = (X >= X_MIN) & (X <= X_MAX)
    for y_lo, y_hi, width in BAR_BANDS:
        band = inside & (Y >= y_lo) & (Y <= y_hi)
        on = np.floor((X - X_MIN) / width) % 2 == 0
        image[band & on] = 1.0
    return image


def pattern_e(X, Y):
    """A tumbling E: spine on the left, three arms pointing right."""
    image = np.zeros(X.shape)
    unit = 1.0                                  # stroke width, degrees
    x0, y0 = 0.6, -2.5                          # lower-left corner
    image[(X >= x0) & (X <= x0 + unit)
          & (Y >= y0) & (Y <= y0 + 5 * unit)] = 1.0     # spine
    for arm in (0, 2, 4):
        image[(X >= x0) & (X <= x0 + 5 * unit)
              & (Y >= y0 + arm * unit)
              & (Y <= y0 + (arm + 1) * unit)] = 1.0     # arms
    return image


def pattern_shapes(X, Y):
    """A filled disc, an open square and a horizontal bar."""
    image = np.zeros(X.shape)
    image[(X - 2.0) ** 2 + (Y - 3.4) ** 2 <= 1.1 ** 2] = 1.0
    outer = (np.abs(X - 3.6) <= 1.7) & (np.abs(Y - 0.0) <= 1.7)
    inner = (np.abs(X - 3.6) <= 1.1) & (np.abs(Y - 0.0) <= 1.1)
    image[outer & ~inner] = 1.0
    image[(X >= 1.0) & (X <= 5.5) & (Y >= -4.6) & (Y <= -3.4)] = 1.0
    return image


PATTERNS = {'bars': pattern_bars, 'e': pattern_e, 'shapes': pattern_shapes}


def make_test_image(params, pattern='bars'):
    """The pattern as a float array in [0, 1], row 0 = bottom."""
    X, Y = coordinate_grid(params)
    return PATTERNS[pattern](X, Y)


def simulate(params, image, out_path, atlas_dir, n_electrodes):
    """Run the image through the simulator and write a comparison."""
    from matplotlib import pyplot as plt
    from dynaphos_lgn.build import build_simulator

    params['electrodes']['n_electrodes'] = n_electrodes
    params['electrodes']['max_eccentricity_deg'] = X_MAX
    simulator, _ = build_simulator(params, atlas_dir=atlas_dir)

    stimulus = image * require(params, 'sampling.stimulus_scale')
    simulator.reset()
    percept = simulator(simulator.sample_stimulus(stimulus)).detach().numpy()

    half = require(params, 'run.view_angle') / 2
    extent = [-half, half, -half, half]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    axes[0].imshow(image, cmap='gray', origin='lower', vmin=0, vmax=1,
                   extent=extent)
    axes[0].set_title('input')
    axes[1].plot(*simulator.array.visual_field.cartesian, 'r.', markersize=4)
    axes[1].set(xlim=(-half, half), ylim=(-half, half))
    axes[1].set_aspect('equal')
    axes[1].set_title(f'{simulator.num_phosphenes} electrodes')
    axes[2].imshow(percept, cmap='gray', origin='lower', vmin=0, vmax=1,
                   extent=extent)
    axes[2].set_title('percept')
    for ax in axes:
        ax.set_xlabel('visual field x (deg)')
    axes[0].set_ylabel('visual field y (deg)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return percept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', choices=sorted(PATTERNS), default='bars')
    parser.add_argument('--params',
                        default=str(REPO / 'config' / 'params_lgn.yaml'))
    parser.add_argument('--out', default=None,
                        help='output PNG (default: out/right_hemifield_<pattern>.png '
                             'next to this script)')
    parser.add_argument('--simulate', action='store_true',
                        help='also render the percept beside the input')
    parser.add_argument('--atlas-dir',
                        default=str(REPO / 'data' / 'Erwin_Atlas'))
    parser.add_argument('--n-electrodes', type=int, default=100)
    args = parser.parse_args()

    params = load_params(args.params)
    params['run']['gpu'] = None
    image = make_test_image(params, args.pattern)

    out = Path(args.out or Path(__file__).parent / 'out'
               / f'right_hemifield_{args.pattern}.png')
    out.parent.mkdir(exist_ok=True)
    # Flipped on the way out: PNG row 0 is the top, ours is the bottom.
    cv2.imwrite(str(out), (image[::-1] * 255).astype(np.uint8))

    res_x = require(params, 'run.resolution')[0]
    left = image[:, :res_x // 2].sum()
    print(f"Wrote {out} ({args.pattern}).")
    print(f"  ink in the left half : {left:.0f} pixels")
    print(f"  ink in the right half: {image[:, res_x // 2:].sum():.0f} pixels")

    if args.simulate:
        comparison = out.with_name(out.stem + '_percept.png')
        simulate(params, image, comparison, args.atlas_dir,
                 args.n_electrodes)
        print(f"Wrote {comparison}.")


if __name__ == '__main__':
    main()
