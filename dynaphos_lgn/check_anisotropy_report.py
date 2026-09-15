"""SUPERSEDED FOR ORIENTATION -- see `check_anisotropy_local_frame.py`.

This script's *magnitude* results (median sigma1/sigma2 by class,
eccentricity and meridian) stand and are still worth running. Its
*orientation* statistic does not: it measures the ellipse's major axis
against a FIXED horizontal axis, while Connolly & Van Essen state their
claim in a LOCAL frame (isoeccentricity vs. isopolar) that rotates with
visual-field position. The two agree only exactly on the meridians. Its
"~123 deg vs ~115 deg, same direction, not opposite" finding was
therefore comparing a local claim against a global ruler, and is
uninterpretable rather than contradictory.

It also pools all parvo layers against all magno layers, which is
coarser than the layer-6-vs-layer-1 contrast the claim is actually
about.

`check_anisotropy_local_frame.py` (12 Sep 2026) redoes both. Kept here
because the magnitude half is still the cheapest way to get those
numbers, and because the superseded report it produced is referenced by
dated project docs.

Original docstring follows.

---

Check #11 (open-design-choices item 11): pull an actual anisotropy
number -- magnitude and orientation -- out of the atlas's own per-voxel
Jacobian, instead of leaving the Connolly & Van Essen 2-3x figure as a
qualitative checkpoint with no absolute-degree number attached.

This is NOT the population-level, cross-animal, uncertainty-bounded
estimate `lgn-retinotopic-anisotropy-experimental-design-2026-09-10.md`
is designed to eventually produce (Tracks A/B/C) -- it's a single-macaque
point estimate from data already computed for the Jacobian atlas. Useful
as a first real number and as the thing Track A/B/C would ultimately be
checked against, not as a replacement for them.

Standalone script; only uses the existing, already-tested `build.py` /
`magnification.py` / `atlas.py` machinery -- no reimplemented atlas
logic. Run from the repo root:

    python -m dynaphos_lgn.check_anisotropy_report
    # or:
    python check_anisotropy_report.py --config config/params_lgn.yaml

Writes a markdown report + a CSV of the full binned table, and prints
the report to stdout.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np


# Meridian-proximity classification: a voxel's inclination (polar angle
# in the visual field, degrees, per JacobianMagnification's x=E cos(I),
# y=E sin(I) convention) counts as "near-horizontal" if it's within
# HALF_WIDTH_DEG of 0 deg or 180 deg, "near-vertical" if within
# HALF_WIDTH_DEG of +90 or -90, else "oblique".
HALF_WIDTH_DEG = 22.5

# Eccentricity bin edges (deg) -- chosen to bracket plausible electrode
# placements, matching the spacing used in
# lgn-retinotopic-anisotropy-experimental-design-2026-09-10.md's Track A
# sampling proposal (1, 3, 6, 12, 24 deg).
ECC_EDGES = np.array([0, 2, 4.5, 9, 18, 40])


def meridian_class(incl_deg: np.ndarray) -> np.ndarray:
    """'horizontal' / 'vertical' / 'oblique' per voxel, from inclination."""
    d_horiz = np.minimum(np.abs(incl_deg), np.abs(np.abs(incl_deg) - 180))
    d_vert = np.abs(np.abs(incl_deg) - 90)
    out = np.full(incl_deg.shape, 'oblique', dtype=object)
    out[d_horiz <= HALF_WIDTH_DEG] = 'horizontal'
    out[d_vert <= HALF_WIDTH_DEG] = 'vertical'
    return out


def circular_mean_axis_deg(orientation_rad: np.ndarray):
    """Circular mean + resultant length R for an AXIAL angle (mod pi).

    Uses the doubled-angle trick (orientation has no arrowhead -- theta
    and theta+pi describe the same axis), per the statistics section of
    lgn-retinotopic-anisotropy-experimental-design-2026-09-10.md.
    Returns (mean_deg in [0, 180), R in [0, 1]).
    """
    finite = np.isfinite(orientation_rad)
    if not finite.any():
        return float('nan'), float('nan')
    theta2 = 2.0 * orientation_rad[finite]
    c, s = np.cos(theta2).mean(), np.sin(theta2).mean()
    mean2 = np.arctan2(s, c)
    mean_deg = np.degrees(0.5 * mean2) % 180.0
    r = np.hypot(c, s)
    return float(mean_deg), float(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/params_lgn.yaml')
    ap.add_argument('--atlas-dir', default=None)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--out-md', default=None)
    ap.add_argument('--out-csv', default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dynaphos_lgn.build import build_atlas, build_jacobian, load_lgn_params
    from dynaphos_lgn.magnification import JacobianMagnification
    from dynaphos_lgn.params import resolve_class_codes

    params = load_lgn_params(args.config)
    atlas = build_atlas(params, atlas_dir=args.atlas_dir,
                         fallback_to_synthetic=False)
    cache_dir = args.cache_dir or args.atlas_dir or str(
        Path(params.get('atlas', {}).get('directory', '.')))
    jacobian_atlas = build_jacobian(params, atlas, cache_dir=cache_dir)

    today = dt.date.today().isoformat()
    md = [f"# Atlas-derived anisotropy report -- {today}",
          "",
          "Item #11 from `open-design-choices-summary-2026-09-11.md`. "
          "Single macaque (the Erwin/Malpeli atlas); this converts the "
          "qualitative Connolly & Van Essen 2-3x checkpoint into actual "
          "numbers at real eccentricities, using the per-voxel Jacobian "
          "already computed for the simulator. Not a substitute for the "
          "cross-animal experimental design -- no animal-to-animal "
          "variance or independent measurement is represented here, only "
          "this one atlas's own reconstruction + Jacobian-fit residual.",
          "",
          "| cell class | ecc bin (deg) | meridian | n | median "
          "anisotropy (sigma1/sigma2) | axis mean (deg from horiz.) | R |",
          "|---|---|---|---|---|---|---|"]

    csv_rows = [("cell_class", "ecc_lo", "ecc_hi", "meridian", "n",
                 "median_anisotropy", "axis_mean_deg", "resultant_R")]

    overall_lines = []

    for cell_class in ('parvo', 'magno'):
        mask = (atlas.valid & jacobian_atlas.jacobian_valid
                & np.isin(atlas.layer,
                          resolve_class_codes(params, cell_class)))
        idx = np.argwhere(mask)
        jac = jacobian_atlas.jacobian[tuple(idx.T)]
        major, minor, orientation = JacobianMagnification.decompose(jac)
        ecc = atlas.eccentricity_deg[tuple(idx.T)]
        incl = atlas.inclination_deg[tuple(idx.T)]
        anisotropy = np.where(minor > 0, major / minor, np.nan)
        merclass = meridian_class(incl)

        finite = np.isfinite(anisotropy) & np.isfinite(orientation)
        overall_mean_deg, overall_r = circular_mean_axis_deg(
            orientation[finite])
        overall_lines.append(
            f"- **{cell_class}**: n={finite.sum():,} valid voxels; overall "
            f"median anisotropy = {np.nanmedian(anisotropy[finite]):.2f}x; "
            f"axis clusters at {overall_mean_deg:.1f} deg from horizontal "
            f"(R={overall_r:.2f}, 0=unclustered .. 1=perfectly aligned)")

        for lo, hi in zip(ECC_EDGES[:-1], ECC_EDGES[1:]):
            in_bin = finite & (ecc >= lo) & (ecc < hi)
            for mc in ('horizontal', 'vertical', 'oblique'):
                sel = in_bin & (merclass == mc)
                n = int(sel.sum())
                if n < 20:
                    continue
                med_a = float(np.nanmedian(anisotropy[sel]))
                axis_deg, r = circular_mean_axis_deg(orientation[sel])
                md.append(f"| {cell_class} | {lo:g}-{hi:g} | {mc} | {n:,} "
                          f"| {med_a:.2f} | {axis_deg:.1f} | {r:.2f} |")
                csv_rows.append(
                    (cell_class, lo, hi, mc, n, med_a, axis_deg, r))

    md.insert(6, '\n'.join(overall_lines) + '\n')

    md.append("")
    md.append("**Reading this against Connolly & Van Essen's checkpoint** "
              "(2-3x, parvocellular anisotropy peaking near the horizontal "
              "meridian, magnocellular near the vertical meridian, with "
              "opposite axis alignments): compare the parvo-horizontal vs. "
              "parvo-vertical rows, and the magno-horizontal vs. "
              "magno-vertical rows, above. A same-direction pattern for "
              "both classes, or a magnitude far outside 2-3x, is worth a "
              "second look before this number gets used anywhere -- it "
              "would either contradict the literature checkpoint or "
              "suggest a Jacobian-fit artefact (e.g. near the ~2.5% of "
              "fit-ok voxels flagged isotropic-by-construction in the "
              "central 1 deg, or in cells with only a few dozen voxels).")

    text = '\n'.join(md)
    print(text)

    out_md = Path(args.out_md or f'research/anisotropy_report_'
                                 f'{dt.date.today().isoformat()}.md')
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text)

    out_csv = Path(args.out_csv or f'research/anisotropy_report_'
                                   f'{dt.date.today().isoformat()}.csv')
    with open(out_csv, 'w') as f:
        for row in csv_rows:
            f.write(','.join(str(x) for x in row) + '\n')

    print(f"\nWritten to {out_md} and {out_csv}")


if __name__ == '__main__':
    main()
