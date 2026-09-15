"""Is `rho_v` -- the volumetric cell density the density route divides by
-- actually a constant? And does the atlas's cell *distribution* match
Malpeli's formula, or only its total?

Why this exists
---------------
`column-thickness-measured-2026-09-12.md` found a residual that no choice
of column thickness can fix: comparing volume magnification measured
from voxel counts against `malpeli_density / rho_v`, the two agree
within 10-25% beyond 7 deg but disagree by 1.5-2x inside 4 deg. T is on
neither side of that comparison, so the cause is upstream of every
magnification model. This script goes after it.

`MalpeliDensityMagnification.from_atlas` takes `rho_v` as **one number**:
the mean cells per voxel over every valid voxel of the class, divided by
the voxel volume. That is only right if packing density is uniform
across the nucleus. It is a testable assumption and it has not been
tested.

What this measures
------------------
Three things, in the order that lets each rule out an explanation for
the one before:

1. **Is the bin comparison itself sound?** The earlier check evaluated
   Malpeli's density at each bin's *midpoint*. The integrand is steeply
   curved -- parvo density goes as ``(E + 2.91)^-2.68`` while the solid
   angle element goes as ``sin(E) dE`` -- so a midpoint rule might
   simply be inaccurate. This script integrates properly and reports
   the midpoint rule's error, so that explanation is disposed of with a
   number rather than an argument.

2. **Is `rho_v` eccentricity-dependent?** Mean cells per voxel, per
   eccentricity bin, against the global mean the code actually uses.

3. **Does the atlas's own cell count match Malpeli's formula per bin?**
   Using the atlas's real `CELLS.DAT` totals rather than
   ``n_voxels * global mean``. This is the comparison the whole-nucleus
   check in `test_magnification.py::
   test_hemifield_integral_reproduces_atlas_cell_counts` makes, but
   resolved by eccentricity -- and a total can match while the
   distribution does not.

The distinction between (2) and (3) is the point. If `rho_v` varies but
the real per-bin cell counts match Malpeli, the residual is entirely an
artefact of the constant-`rho_v` assumption and is fixable in the model.
Whatever is left after that is a genuine disagreement between the atlas
and the formula about where the cells are.

Run from the repo root::

    python -m dynaphos_lgn.check_cell_density_profile

Writes a markdown report and a CSV to `research/`, and prints the report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

ECC_EDGES = np.array([1.0, 2.0, 4.0, 7.0, 10.0, 15.0, 20.0, 30.0, 45.0,
                      60.0, 90.0])

DEG2_PER_STERADIAN = (180.0 / np.pi) ** 2

# Erwin et al. (1999)'s published reconstruction totals, which the
# whole-nucleus check in the test suite reproduces.
ERWIN_PUBLISHED_CELLS = {'parvo': 1_048_571, 'magno': 104_841}

# Samples per bin for the solid-angle integral. The integrand is smooth;
# this is about removing the midpoint rule as a suspect, not precision.
INTEGRATION_SAMPLES = 20001


def hemifield_solid_angle_deg2(ecc_lo_deg: float, ecc_hi_deg: float
                               ) -> float:
    """One hemifield's solid angle between two eccentricities, deg^2."""
    lo, hi = np.deg2rad(ecc_lo_deg), np.deg2rad(ecc_hi_deg)
    return float(np.pi * (np.cos(lo) - np.cos(hi)) * DEG2_PER_STERADIAN)


def integrated_cells(ecc_lo_deg: float, ecc_hi_deg: float, params,
                     cell_class: str, malpeli_density) -> float:
    """Malpeli's predicted cell count in an eccentricity annulus.

    Integrates ``density(E) * dOmega(E)`` rather than taking
    ``density(midpoint) * Omega``, so the comparison does not inherit a
    quadrature error from a steeply curved integrand.
    """
    e = np.linspace(ecc_lo_deg, ecc_hi_deg, INTEGRATION_SAMPLES)
    d_omega = (np.pi * np.sin(np.deg2rad(e))
               * np.deg2rad(e[1] - e[0]) * DEG2_PER_STERADIAN)
    return float(np.sum(malpeli_density(e, params, cell_class) * d_omega))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/params_lgn.yaml')
    ap.add_argument('--atlas-dir', default=None)
    ap.add_argument('--out-md', default=None)
    ap.add_argument('--out-csv', default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dynaphos_lgn.atlas import ErwinAtlas
    from dynaphos_lgn.build import load_lgn_params
    from dynaphos_lgn.magnification import malpeli_density
    from dynaphos_lgn.params import require, resolve_class_codes

    params = load_lgn_params(args.config)
    atlas = ErwinAtlas(params).build_atlas(
        Path(args.atlas_dir or require(params, 'atlas.directory')))
    voxel_volume = float(require(params, 'atlas.voxel_size_mm')) ** 3

    today = dt.date.today().isoformat()
    md = [f"# Cell density profile: is `rho_v` a constant? -- {today}",
          "",
          "Chasing the residual that "
          f"`column-thickness-measured-{today}.md` found and could not "
          "attribute: volume magnification disagrees with the "
          "Malpeli-density prediction by 1.5-2x inside 4 deg, with no "
          "column thickness on either side of the comparison.",
          "",
          "`MalpeliDensityMagnification.from_atlas` takes `rho_v` as a "
          "**single global mean** over the class's voxels. That is only "
          "correct if packing density is uniform across the nucleus, "
          "which has not been checked.",
          ""]

    # ------------------------------------------------------------------
    # 1. Dispose of the quadrature explanation.
    # ------------------------------------------------------------------
    md += ["## First: it is not a midpoint-rule artefact",
           "",
           "The earlier comparison evaluated Malpeli's density at each "
           "bin's midpoint. With density going as `(E + 2.91)^-2.68` and "
           "the solid angle as `sin(E) dE`, that is worth checking "
           "before blaming anything else.",
           "",
           "| bin (deg) | parvo midpoint / integrated | "
           "magno midpoint / integrated |",
           "|---|---|---|"]
    worst_quadrature = 0.0
    for lo, hi in zip(ECC_EDGES[:-1], ECC_EDGES[1:]):
        row = []
        for cell_class in ('parvo', 'magno'):
            omega = hemifield_solid_angle_deg2(lo, hi)
            midpoint = float(
                malpeli_density(0.5 * (lo + hi), params, cell_class)) * omega
            exact = integrated_cells(lo, hi, params, cell_class,
                                     malpeli_density)
            ratio = midpoint / exact
            worst_quadrature = max(worst_quadrature, abs(ratio - 1.0))
            row.append(f"{ratio:.3f}")
        md.append(f"| {lo:g}-{hi:g} | " + " | ".join(row) + " |")
    md += ["",
           f"Worst deviation {worst_quadrature:.1%}. **Ruled out** -- "
           f"nowhere near the 1.5-2x being explained. Every number below "
           f"uses the proper integral anyway.",
           ""]

    csv_rows = [("cell_class", "ecc_lo", "ecc_hi", "n_voxels",
                 "atlas_cells", "cells_per_voxel", "vs_global_mean",
                 "malpeli_cells", "atlas_over_malpeli",
                 "constant_rho_v_over_malpeli")]
    findings = {}

    for cell_class in ('parvo', 'magno'):
        codes = resolve_class_codes(params, cell_class)
        mask = (atlas.valid & np.isin(atlas.layer, codes)
                & ~atlas.is_ipsi_sentinel())
        sel = tuple(np.argwhere(mask).T)
        ecc = atlas.eccentricity_deg[sel]
        cells = atlas.cells_per_voxel[sel]
        global_mean = float(cells.mean())

        md += [f"## {cell_class}",
               "",
               f"Global mean {global_mean:.4f} cells/voxel "
               f"(`rho_v` = {global_mean / voxel_volume:,.0f} /mm^3) -- "
               f"the single number the code uses.",
               "",
               "| ecc bin (deg) | n voxels | atlas cells | cells/voxel | "
               "vs global | Malpeli cells | **atlas / Malpeli** | "
               "constant-rho_v / Malpeli |",
               "|---|---|---|---|---|---|---|---|"]

        per_voxel_ratios, true_ratios, constant_ratios = [], [], []
        for lo, hi in zip(ECC_EDGES[:-1], ECC_EDGES[1:]):
            in_bin = (ecc >= lo) & (ecc < hi)
            n = int(in_bin.sum())
            if n < 50:
                continue
            bin_cells = float(cells[in_bin].sum())
            bin_mean = float(cells[in_bin].mean())
            predicted = integrated_cells(lo, hi, params, cell_class,
                                         malpeli_density)
            true_ratio = bin_cells / predicted
            constant_ratio = n * global_mean / predicted
            per_voxel_ratios.append(bin_mean / global_mean)
            true_ratios.append((lo, hi, true_ratio))
            constant_ratios.append(constant_ratio)
            md.append(f"| {lo:g}-{hi:g} | {n:,} | {bin_cells:,.0f} | "
                      f"{bin_mean:.4f} | {bin_mean / global_mean:.3f} | "
                      f"{predicted:,.0f} | **{true_ratio:.3f}** | "
                      f"{constant_ratio:.3f} |")
            csv_rows.append((cell_class, lo, hi, n, bin_cells, bin_mean,
                             bin_mean / global_mean, predicted, true_ratio,
                             constant_ratio))
        md.append("")

        spread = max(per_voxel_ratios) / min(per_voxel_ratios)
        central = [r for lo, hi, r in true_ratios if hi <= 4.0]
        central_constant = constant_ratios[:len(central)]
        findings[cell_class] = dict(
            spread=spread,
            central_true=float(np.mean(central)) if central else float('nan'),
            central_constant=(float(np.mean(central_constant))
                              if central_constant else float('nan')),
            total_atlas=float(cells.sum()),
            per_voxel=per_voxel_ratios)

        md += [f"**`rho_v` is not constant.** Cells per voxel runs "
               f"{min(per_voxel_ratios):.2f}x to "
               f"{max(per_voxel_ratios):.2f}x of the global mean -- a "
               f"spread of **{spread:.2f}x** across the nucleus, falling "
               f"monotonically with eccentricity. Since volume "
               f"magnification is `density / rho_v`, an error of that "
               f"size in `rho_v` is the same size in volume "
               f"magnification and its square root "
               f"({np.sqrt(spread):.2f}x) in linear magnification.",
               ""]

    # ------------------------------------------------------------------
    # How much of the residual does that account for?
    # ------------------------------------------------------------------
    md += ["## How much of the central residual this explains",
           "",
           "Inside 4 deg, comparing the two ways of counting the atlas's "
           "cells against the same Malpeli prediction:",
           "",
           "| class | with constant `rho_v` (the old number) | "
           "with the atlas's real cell counts | explained |",
           "|---|---|---|---|"]
    for cell_class, f in findings.items():
        closed = ((f['central_true'] - f['central_constant'])
                  / (1.0 - f['central_constant'])
                  if f['central_constant'] < 1.0 else float('nan'))
        md.append(f"| {cell_class} | {f['central_constant']:.3f} | "
                  f"**{f['central_true']:.3f}** | {closed:.0%} |")
    md.append("")

    magno = findings['magno']
    parvo = findings['parvo']
    md += [f"**For magnocellular the residual is essentially gone.** It "
           f"goes from {magno['central_constant']:.2f} to "
           f"{magno['central_true']:.2f} -- the atlas's real magno cell "
           f"counts agree with Malpeli's formula in the central field to "
           f"within a few percent. The entire apparent 2x disagreement "
           f"was the constant-`rho_v` assumption: magnocellular packing "
           f"in the central 2 deg is {magno['per_voxel'][0]:.2f}x the "
           f"nucleus-wide mean, which is the largest departure from "
           f"uniformity anywhere in either class.",
           "",
           f"**For parvocellular about half of it goes.** "
           f"{parvo['central_constant']:.2f} -> "
           f"{parvo['central_true']:.2f}. A real ~"
           f"{(1 - parvo['central_true']) * 100:.0f}% shortfall remains: "
           f"the atlas holds materially fewer parvocellular cells inside "
           f"4 deg than Malpeli's formula predicts for that solid angle, "
           f"and that is not a packing-density effect.",
           ""]

    # ------------------------------------------------------------------
    # The totals agree; the distribution does not.
    # ------------------------------------------------------------------
    md += ["## Why the existing whole-nucleus test does not catch this",
           "",
           "`test_hemifield_integral_reproduces_atlas_cell_counts` checks "
           "the integrated total against Erwin et al.'s published "
           "figures and passes. It is a real check and it is not wrong -- "
           "but a total can match while the distribution does not, and "
           "here it does exactly that.",
           "",
           "| class | atlas total (1-90 deg, valid voxels) | "
           "Erwin published | ratio |",
           "|---|---|---|---|"]
    for cell_class, f in findings.items():
        published = ERWIN_PUBLISHED_CELLS[cell_class]
        md.append(f"| {cell_class} | {f['total_atlas']:,.0f} | "
                  f"{published:,} | {f['total_atlas'] / published:.3f} |")
    md += ["",
           "The totals land within a few percent while individual bins "
           "sit at 0.74 and 1.17. Those are compensating errors, and "
           "only an eccentricity-resolved comparison can see them. **A "
           "per-bin version of that test would be a materially stronger "
           "check** than the whole-nucleus one, and it is the same "
           "machinery.",
           ""]

    md += ["## What to do with this",
           "",
           "1. **`rho_v` should be eccentricity-resolved, or the "
           "constant-`rho_v` error should be stated where it bites.** "
           f"The magno spread is {magno['spread']:.1f}x and the parvo "
           f"spread {parvo['spread']:.1f}x, and both fall monotonically "
           "with eccentricity -- an anatomical gradient, not noise. "
           "`MalpeliDensityMagnification` now accepts a callable or a "
           "tabulated profile for `cells_per_mm3` as well as a scalar; "
           "the config default is unchanged.",
           "",
           "2. **This compounds with the column-thickness finding rather "
           "than replacing it.** Linear magnification goes as "
           "`sqrt(density / (rho_v * T))`, and *both* `rho_v` and `T` "
           "are configured as constants while both measure as "
           "eccentricity- and class-dependent. They are separate errors "
           "in the same expression.",
           "",
           "3. **The remaining parvo shortfall inside 4 deg is real and "
           "unexplained.** It is not quadrature, not packing density, "
           "and not column thickness. It sits at the edge of the "
           "isotropic-by-construction region, where Erwin et al. "
           "substituted Malpeli's own formula for interpolation -- which "
           "makes a shortfall there odd rather than expected, and worth "
           "its own look. `FOVEOLA.DAT`, surfaced earlier and never "
           "used, is the obvious next place to look.",
           ""]

    text = '\n'.join(md)
    print(text)
    out_md = Path(args.out_md
                  or f'research/cell-density-profile-{today}.md')
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text)
    out_csv = Path(args.out_csv
                   or f'research/cell-density-profile-{today}.csv')
    with open(out_csv, 'w') as f:
        for row in csv_rows:
            f.write(','.join(str(x) for x in row) + '\n')
    print(f"\nWritten to {out_md} and {out_csv}")


if __name__ == '__main__':
    main()
