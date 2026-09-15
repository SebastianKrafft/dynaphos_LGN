"""Check #2 (open-design-choices item 2): does MalpeliDensityMagnification
agree with the live JacobianMagnification, on the real atlas?

Standalone script, meant to be dropped into `dynaphos_lgn/` (it only uses
the existing, already-tested `build.py` helpers and `magnification.py`
classes -- no reimplemented logic). Run from the repo root:

    python -m dynaphos_lgn.check_magnification_agreement
    # or, if placed elsewhere on the path:
    python check_magnification_agreement.py --config config/params_lgn.yaml

Writes a markdown report (with the numeric comparison table for both
parvo and magno) next to the atlas, and prints it to stdout.

*** A bug this script found, since fixed ***
`MalpeliDensityMagnification.consistency_report()` used to call
`reference.at_eccentricity(e)` with NO `cell_class` argument. Every
`MagnificationModel.at_eccentricity` defaults `cell_class='parvo'`, and
`JacobianMagnification.at_eccentricity` actually uses that argument (it
picks a different set of atlas voxels per class) -- so as written, it
silently compared a *magno* MalpeliDensityMagnification against the
*parvo* Jacobian table. The parvo comparison was unaffected (the wrong
default happened to be the right value); the magno numbers were wrong.

Fixed in `magnification.py` on 12 Sep 2026, with a regression test at
`test_magnification.py::TestMalpeliDensityMagnification::
test_consistency_report_uses_own_cell_class`. This script's manual
magno workaround has been removed accordingly -- both classes now go
through the same built-in path, which is the point of fixing it.

*** What this script no longer needs to guess ***
`T`, the projection-column thickness, was an unmeasured config value
when this comparison was first run, so its disagreement was reported as
an open unknown. It is now measured from the atlas directly -- see
`check_column_thickness.py`, which supersedes the "what does T do to
this?" half of this script's question. Run that one too before drawing
conclusions from the ratios below.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/params_lgn.yaml')
    ap.add_argument('--atlas-dir', default=None,
                     help='Overrides atlas.directory from the config.')
    ap.add_argument('--cache-dir', default=None,
                     help='Where JACOBIAN.npz lives / will be written. '
                          'Defaults to the atlas directory.')
    ap.add_argument('--eccentricities', default='1,3,5,10,20,30',
                     help='Comma-separated eccentricities, deg.')
    ap.add_argument('--out', default=None,
                     help='Markdown report path. Defaults to '
                          'research/magnification_agreement_<date>.md')
    args = ap.parse_args()

    # Import from the repo, not from this script's own directory, so it
    # works whether this file lives inside dynaphos_lgn/ or beside it.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dynaphos_lgn.build import build_atlas, build_jacobian, load_lgn_params
    from dynaphos_lgn.magnification import (JacobianMagnification,
                                             MalpeliDensityMagnification)

    params = load_lgn_params(args.config)

    # fallback_to_synthetic=False: a synthetic-atlas result would be
    # actively misleading for this check, not just uninformative -- see
    # build_atlas's own docstring.
    atlas = build_atlas(params, atlas_dir=args.atlas_dir,
                         fallback_to_synthetic=False)
    cache_dir = args.cache_dir or args.atlas_dir or str(
        Path(params.get('atlas', {}).get('directory', '.')))
    jacobian_atlas = build_jacobian(params, atlas, cache_dir=cache_dir)

    jac_mag = JacobianMagnification(jacobian_atlas, params, erwin_atlas=atlas)
    e = np.array([float(x) for x in args.eccentricities.split(',')])

    lines = [
        f"# Magnification agreement check -- {dt.date.today().isoformat()}",
        "",
        "Item #2 from `open-design-choices-summary-2026-09-11.md`: does "
        "`MalpeliDensityMagnification` agree with the live "
        "`JacobianMagnification` (the simulator's actual "
        "`magnification.model: jacobian` default)? Not run before -- the "
        "original 1.3-2.4x 'Finding 1' figure was measured against the "
        "retired atlas-gradient-sampling script (`M_of_E.csv`), not this.",
        "",
    ]

    for cell_class in ('parvo', 'magno'):
        malpeli = MalpeliDensityMagnification.from_atlas(
            atlas, params, cell_class=cell_class)
        lines.append(f"## {cell_class}")
        lines.append("")
        # Both classes through the built-in path: `consistency_report`
        # now passes its own cell_class to the reference model, so the
        # hand-built magno table this script used to carry is gone.
        lines.append("```")
        lines.append(malpeli.consistency_report(
            reference=jac_mag, eccentricities_deg=e))
        lines.append("```")
        lines.append("")

    lines.append(
        "**Reading this:** a ratio far from 1 is a processing/unit "
        "problem, not corroborating biology -- Malpeli's formula and the "
        "atlas both trace back to the same single macaque. If the ratio "
        "*is* close to 1, that's still worth recording: it would mean "
        "the current, correctly-derived Jacobian magnification lands "
        "close to the Malpeli-density estimate's central column-thickness "
        "assumption, which the earlier (now-retired) atlas-gradient-script "
        "comparison did not show.\n\n"
        "**Since 12 Sep 2026 there is a better question to ask.** The "
        "ratios above are computed at the config's *assumed* column "
        "thickness. `check_column_thickness.py` measures T from the "
        "atlas instead, and at the measured T the linear ratios come "
        "out within about 10% of 1 beyond 7 deg for both classes. That "
        "agreement is partly algebraic rather than corroborating -- the "
        "Jacobian cancels out of the comparison once T is derived from "
        "it -- so read that report's residuals, not its agreements.")

    text = '\n'.join(lines)
    print(text)
    out = Path(args.out or f'research/magnification_agreement_'
                           f'{dt.date.today().isoformat()}.md')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"\nWritten to {out}")


if __name__ == '__main__':
    main()
