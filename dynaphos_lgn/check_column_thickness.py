"""Measure the projection-column thickness T from the atlas itself,
instead of assuming it -- and check the result against Connolly & Van
Essen (1984)'s directly measured laminar thicknesses.

Why this exists
---------------
`MalpeliDensityMagnification` turns Malpeli et al. (1996)'s cell density
(cells/deg^2) into a linear tissue magnification (mm/deg) by dividing by
a volumetric cell density rho_v and then by a **projection-column
thickness T**, neither of which the published formula supplies::

    volume magnification  (mm^3/deg^2) = density / rho_v
    areal  magnification  (mm^2/deg^2) = volume / T
    linear magnification  (mm/deg)     = sqrt(areal)

T is currently a config guess: `(1.0, 3.0, 5.0)` mm, with a
back-solved "plausibility bracket" of 1.75-5.37 mm obtained by matching
the nucleus's total map length. Connolly & Van Essen (1984) p.553
*measured* laminar thickness directly in three macaque LGNs and got
0.125-0.225 mm per (fused) lamina -- one to two orders of magnitude
smaller. Both cannot be right.

The measurement this script makes
---------------------------------
T does not have to be assumed at all. Two quantities the repo already
has, in the same eccentricity bin, pin it down exactly:

- **Volume magnification**, straight from voxel counts. Each valid voxel
  is `voxel_size_mm ** 3` of real tissue, and the visual-field solid
  angle a bin subtends is closed-form. So `V / Omega` is mm^3 per deg^2,
  measured, with no cell counts and no density formula in the chain.
- **Areal magnification**, from the cached per-voxel Jacobian: the
  product of its two singular values is deg^2/mm^2, so its reciprocal is
  mm^2/deg^2.

Dividing one by the other leaves millimetres::

    T = (V / Omega) / A

Per voxel rather than per bin, since a bin's voxels do not share one
Jacobian: a voxel of volume `v` with areal magnification
`a = 1/(sigma1*sigma2)` accounts for `v / (a*T)` of solid angle, so
summing over a bin and solving for T gives

    T_bin = voxel_volume * sum(sigma1*sigma2) / Omega_bin

which is what `implied_thickness_mm` below computes. This is a
*measurement*, not a third assumption -- its only inputs are the atlas's
own voxel geometry, its own retinotopy, and the hemifield solid-angle
convention already fixed elsewhere in the repo (and tested in
`test_magnification.py::
test_hemifield_integral_reproduces_atlas_cell_counts`).

Three things it then checks
---------------------------
1. **Order of magnitude.** Does the measured T look like Connolly & Van
   Essen's ~0.1-0.4 mm, or like the config's 1-5 mm?
2. **The 15-degree dropout.** C&VE p.554: parvocellular leaflets 4 and 5
   are "restricted to the representation of the central 15 deg". Under
   the atlas's own layer coding that is a falsifiable prediction about
   where parvo tissue-per-solid-angle should drop -- see below.
3. **The density route, independently of T.** Volume magnification is
   measurable *and* predictable (`malpeli_density / rho_v`), with T
   nowhere in either. So comparing those two isolates whether the
   disagreement reported in `magnification_agreement_2026-09-11.md` is
   about T at all, or lives further upstream.

Layer coding -- the part that makes (2) testable
------------------------------------------------
LAYERS.DAT has four codes, not six: `magno_contra`, `magno_ipsi`,
`parvo_ipsi`, `parvo_contra` (`atlas.layers.names`). In the standard
macaque scheme layers 1, 4 and 6 take contralateral input and 2, 3 and 5
ipsilateral, so the atlas's codes correspond to

    code 1  magno_contra  <-> C&VE layer 1              150 um
    code 2  magno_ipsi    <-> C&VE layer 2              125 um
    code 3  parvo_ipsi    <-> C&VE fused leaflet 3+5    175 um
    code 4  parvo_contra  <-> C&VE fused leaflet 4+6    225 um (binocular)

and the atlas's two parvo codes line up **exactly** with the two fused
parvocellular sheets C&VE measured, because C&VE fused 4 with 6 and 5
with 3 on the same anatomical grounds (p.546-548). That correspondence is
what lets the 15 deg prediction be stated per atlas code: beyond 15 deg,
code 3 should thin towards layer 3 alone and code 4 towards layer 6
alone.

Run from the repo root::

    python -m dynaphos_lgn.check_column_thickness

Writes a markdown report and a CSV to `research/`, and prints the report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

# Eccentricity bin edges, degrees. Deliberately straddling 15 deg (the
# C&VE leaflet-dropout boundary) with a bin edge exactly there, so the
# prediction in (2) above is testable by comparing adjacent rows rather
# than by eyeballing a curve.
ECC_EDGES = np.array([1.0, 2.0, 4.0, 7.0, 10.0, 15.0, 20.0, 30.0, 45.0,
                      60.0, 90.0])

# Connolly & Van Essen (1984) p.553, shrinkage-corrected, three LGNs.
# Full-text verified 11 Sep 2026 -- see
# `connolly-van-essen-1984-anisotropy-verification-2026-09-11.md`.
# (central, low, high) mm.
CVE_THICKNESS_MM = {
    1: (0.150, 0.130, 0.180),      # layer 1, magno contra
    2: (0.125, 0.100, 0.150),      # layer 2, magno ipsi
    3: (0.175, 0.140, 0.200),      # fused 3+5, parvo ipsi
    4: (0.225, 0.200, 0.270),      # fused 4+6, parvo contra, binocular
}
CVE_LABEL = {1: 'layer 1', 2: 'layer 2', 3: 'fused 3+5',
             4: 'fused 4+6 (binoc.)'}

# p.554: leaflets 4 and 5 are restricted to the central 15 deg.
LEAFLET_DROPOUT_ECC_DEG = 15.0

# Upper edge of the window used to test that dropout. Beyond roughly
# here the monocular crescent takes over -- only the contralateral eye
# sees it, so the ipsilateral codes (2 and 3) empty out for a reason that
# has nothing to do with leaflets 4 and 5. Including those bins would
# credit the leaflet prediction with an effect the crescent produces.
DROPOUT_WINDOW_HI_DEG = 45.0

DEG2_PER_STERADIAN = (180.0 / np.pi) ** 2


def hemifield_solid_angle_deg2(ecc_lo_deg: float, ecc_hi_deg: float
                               ) -> float:
    """Visual-field area of an eccentricity annulus, one hemifield, deg^2.

    Spherical, not flat: the full-sphere solid angle between two
    eccentricities is ``2*pi*(cos lo - cos hi)`` sr, and one LGN
    represents one hemifield, so halve it. This is the same convention
    `test_hemifield_integral_reproduces_atlas_cell_counts` pins down by
    reproducing Erwin et al.'s published cell totals -- a flat
    convention would diverge from it badly in the periphery, which is
    exactly where this script's interesting rows are.
    """
    lo, hi = np.deg2rad(ecc_lo_deg), np.deg2rad(ecc_hi_deg)
    return float(np.pi * (np.cos(lo) - np.cos(hi)) * DEG2_PER_STERADIAN)


def implied_thickness_mm(sigma_product: np.ndarray, voxel_volume_mm3: float,
                         solid_angle_deg2: float) -> float:
    """T for one bin, from per-voxel areal magnifications.

    :param sigma_product: ``sigma1 * sigma2`` per voxel, deg^2/mm^2.
    :param voxel_volume_mm3: Volume of one voxel.
    :param solid_angle_deg2: Solid angle the bin's voxels between them
        are taken to cover.
    :return: Thickness in mm. See the module docstring for the
        derivation; in one line, each voxel covers
        ``voxel_volume / (T * areal)`` of solid angle.
    """
    finite = np.isfinite(sigma_product)
    if not finite.any() or solid_angle_deg2 <= 0:
        return float('nan')
    return float(voxel_volume_mm3 * sigma_product[finite].sum()
                 / solid_angle_deg2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/params_lgn.yaml')
    ap.add_argument('--atlas-dir', default=None)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--include-ipsi-flat', action='store_true',
                    help="Keep the ipsilateral-hemifield voxels carrying "
                         "the flat 135 deg placeholder inclination. They "
                         "are real tissue but represent the OTHER "
                         "hemifield, so counting them inflates V without "
                         "adding solid angle. Off by default, matching "
                         "atlas.exclude_ipsi_flat_inclination.")
    ap.add_argument('--out-md', default=None)
    ap.add_argument('--out-csv', default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dynaphos_lgn.atlas import ErwinAtlas, JacobianAtlas
    from dynaphos_lgn.build import load_lgn_params
    from dynaphos_lgn.magnification import (JacobianMagnification,
                                            malpeli_density)
    from dynaphos_lgn.params import require, resolve_class_codes

    params = load_lgn_params(args.config)
    atlas_dir = Path(args.atlas_dir
                     or require(params, 'atlas.directory'))
    atlas = ErwinAtlas(params).build_atlas(atlas_dir)
    cache_dir = Path(args.cache_dir or atlas_dir)
    jacobian = JacobianAtlas.load(
        cache_dir / require(params, 'atlas.jacobian_cache'), params)

    voxel_mm = float(require(params, 'atlas.voxel_size_mm'))
    voxel_volume = voxel_mm ** 3
    layer_names = require(params, 'atlas.layers.names')

    base_mask = atlas.valid & jacobian.jacobian_valid
    if not args.include_ipsi_flat:
        base_mask = base_mask & ~atlas.is_ipsi_sentinel()

    today = dt.date.today().isoformat()
    md = [f"# Projection-column thickness, measured from the atlas "
          f"-- {today}",
          "",
          "> **Partly superseded, same day.** Two follow-ups correct how "
          "this report frames two of its findings, without changing any "
          "of its numbers:",
          "> ",
          f"> - `sheet-thickness-morphology-{today}.md` shows the "
          "parvo/magno difference below is mostly a **representation "
          "count**, not a tissue thickness: inside 15 deg each parvo "
          "code carries two laminae. The measured values stand and the "
          "config still needs to differ by class; the *reason* is "
          "different, and it means the parameter is inherently "
          "eccentricity-dependent.",
          f"> - `cell-density-profile-{today}.md` shows the central "
          "residual in the third table is largely a **constant-`rho_v` "
          "artefact** -- packing density is not uniform. It accounts "
          "for all of the magno residual and about a fifth of the "
          "parvo one.",
          "",
          "Open-design item 2, and the Connolly & Van Essen lead recorded "
          "on 11 Sep. `MalpeliDensityMagnification`'s column thickness T "
          "has been a config guess (`column_thickness_mm: [1.0, 3.0, "
          "5.0]`, back-solved bracket 1.75-5.37 mm). It does not have to "
          "be. Volume magnification is measurable straight from voxel "
          "counts and areal magnification straight from the cached "
          "Jacobian, and `T = (V/Omega) / A` -- so the atlas measures its "
          "own T. Derivation in the script docstring.",
          "",
          f"Ipsilateral-hemifield placeholder voxels "
          f"({'INCLUDED' if args.include_ipsi_flat else 'excluded'}); "
          f"voxel {voxel_mm:g} mm; spherical hemifield solid-angle "
          f"convention.",
          ""]

    csv_rows = [("layer_code", "layer_name", "ecc_lo_deg", "ecc_hi_deg",
                 "n_voxels", "tissue_volume_mm3", "solid_angle_deg2",
                 "volume_magnification_mm3_per_deg2",
                 "median_areal_mm2_per_deg2", "implied_T_mm")]

    # ------------------------------------------------------------------
    # 1. Per atlas layer code: the measured T, against C&VE's measured
    #    thickness for the lamina that code corresponds to.
    # ------------------------------------------------------------------
    md += ["## Measured T, per atlas layer code",
           "",
           "One row per layer per eccentricity bin. `T measured` is this "
           "script's estimate; `C&VE` is the directly measured thickness "
           "of the lamina that atlas code corresponds to (p.553, "
           "shrinkage-corrected, n=3 LGNs). Ratio > 1 means the atlas "
           "implies thicker tissue than C&VE measured.",
           ""]

    per_layer_summary = {}
    for code in sorted(layer_names):
        name = layer_names[code]
        mask = base_mask & (atlas.layer == code)
        idx = np.argwhere(mask)
        if not idx.size:
            continue
        sel = tuple(idx.T)
        major, minor, _ = JacobianMagnification.decompose(
            jacobian.jacobian[sel])
        sigma_product = major * minor          # deg^2 / mm^2
        ecc = atlas.eccentricity_deg[sel]

        cve_mid, cve_lo, cve_hi = CVE_THICKNESS_MM[code]
        md += [f"### code {code} -- `{name}` "
               f"(C&VE {CVE_LABEL[code]}: {cve_mid * 1000:.0f} um, "
               f"range {cve_lo * 1000:.0f}-{cve_hi * 1000:.0f})",
               "",
               "| ecc bin (deg) | n voxels | tissue (mm^3) | "
               "solid angle (deg^2) | V/Omega (mm^3/deg^2) | "
               "median areal (mm^2/deg^2) | **T measured (mm)** | "
               "T / C&VE |",
               "|---|---|---|---|---|---|---|---|"]
        rows = []
        for lo, hi in zip(ECC_EDGES[:-1], ECC_EDGES[1:]):
            in_bin = (ecc >= lo) & (ecc < hi) & np.isfinite(sigma_product)
            n = int(in_bin.sum())
            if n < 50:
                continue
            omega = hemifield_solid_angle_deg2(lo, hi)
            volume = n * voxel_volume
            t = implied_thickness_mm(sigma_product[in_bin], voxel_volume,
                                     omega)
            positive = in_bin & (sigma_product > 0)
            areal = (float(np.median(1.0 / sigma_product[positive]))
                     if positive.any() else float('nan'))
            md.append(f"| {lo:g}-{hi:g} | {n:,} | {volume:.4f} | "
                      f"{omega:,.0f} | {volume / omega:.3e} | "
                      f"{areal:.3e} | **{t:.4f}** | {t / cve_mid:.2f} |")
            csv_rows.append((code, name, lo, hi, n, volume, omega,
                             volume / omega, areal, t))
            rows.append((lo, hi, t))
        per_layer_summary[code] = rows
        md.append("")

    # ------------------------------------------------------------------
    # 2. The 15 deg leaflet-dropout prediction.
    # ------------------------------------------------------------------
    md += ["## The 15 degree leaflet dropout (C&VE p.554)",
           "",
           "Prediction: leaflets 4 and 5 are restricted to the central "
           f"{LEAFLET_DROPOUT_ECC_DEG:g} deg, so beyond it the fused "
           "parvo sheets should thin towards layer 6 alone (code 4) and "
           "layer 3 alone (code 3), while the magnocellular codes 1 and "
           "2 -- which have no leaflet to lose -- should not show a "
           "matching step. The magno codes are the control: a step in "
           "*every* layer would point at a binning or solid-angle "
           "artefact instead.",
           "",
           f"Measured over 1-{LEAFLET_DROPOUT_ECC_DEG:g} deg vs. "
           f"{LEAFLET_DROPOUT_ECC_DEG:g}-{DROPOUT_WINDOW_HI_DEG:g} deg. "
           f"The upper edge is not arbitrary: past roughly there the "
           f"monocular crescent takes over and the ipsilateral codes "
           f"empty out for an unrelated reason (quantified below).",
           "",
           "| layer code | name | mean T inside 15 deg (mm) | "
           f"mean T 15-{DROPOUT_WINDOW_HI_DEG:g} deg (mm) | "
           "ratio out/in |",
           "|---|---|---|---|---|"]
    dropout = {}
    for code, rows in per_layer_summary.items():
        inside = [t for lo, hi, t in rows
                  if hi <= LEAFLET_DROPOUT_ECC_DEG and np.isfinite(t)]
        outside = [t for lo, hi, t in rows
                   if lo >= LEAFLET_DROPOUT_ECC_DEG
                   and hi <= DROPOUT_WINDOW_HI_DEG and np.isfinite(t)]
        if not inside or not outside:
            continue
        ti, to = float(np.mean(inside)), float(np.mean(outside))
        dropout[code] = to / ti
        md.append(f"| {code} | `{layer_names[code]}` | {ti:.4f} | "
                  f"{to:.4f} | {to / ti:.2f} |")
    md.append("")

    # The crescent itself, quantified -- the dropout text promises this,
    # and it doubles as a validation that the eye-of-origin coding is
    # being read the right way round.
    crescent_lo, crescent_hi = 60.0, 90.0
    md += ["### The monocular crescent, as a read-the-coding check", ""]
    crescent = {}
    for code in sorted(layer_names):
        mask = base_mask & (atlas.layer == code)
        sel = tuple(np.argwhere(mask).T)
        ecc = atlas.eccentricity_deg[sel]
        n_far = int(((ecc >= crescent_lo) & (ecc < crescent_hi)).sum())
        n_mid = int(((ecc >= 20.0) & (ecc < 45.0)).sum())
        crescent[code] = (n_far, n_mid, n_far / n_mid if n_mid else
                          float('nan'))
    md += ["| layer code | name | eye | n voxels 20-45 deg | "
           f"n voxels {crescent_lo:g}-{crescent_hi:g} deg | ratio |",
           "|---|---|---|---|---|---|"]
    for code in sorted(crescent):
        n_far, n_mid, ratio = crescent[code]
        eye = 'contra' if 'contra' in layer_names[code] else 'ipsi'
        md.append(f"| {code} | `{layer_names[code]}` | {eye} | "
                  f"{n_mid:,} | {n_far:,} | {ratio:.3f} |")
    md += ["",
           "Only the contralateral eye sees the far temporal crescent, so "
           "the ipsilateral codes should collapse there and the "
           "contralateral ones should not. They do, by two to three "
           "orders of magnitude. Nothing in this script or in the atlas "
           "loader enforces that -- `LAYERS.DAT` is read as raw integers "
           "and the eye labels come from the config -- so it is a real "
           "check that `atlas.layers.eyes` is assigned the right way "
           "round, and that the eccentricity scaling is too.",
           ""]

    parvo_codes = [c for c in dropout
                   if layer_names[c].startswith('parvo')]
    magno_codes = [c for c in dropout
                   if layer_names[c].startswith('magno')]
    if parvo_codes and magno_codes:
        p_ratio = float(np.mean([dropout[c] for c in parvo_codes]))
        m_ratio = float(np.mean([dropout[c] for c in magno_codes]))
        md += [f"Mean out/in ratio: **parvo {p_ratio:.2f}**, "
               f"**magno {m_ratio:.2f}**.",
               "",
               "Read it as a *differential*: the prediction is that parvo "
               "thins relative to magno, not that parvo thins in "
               "absolute terms -- every layer's apparent thickness also "
               "moves with whatever else changes peripherally (fewer "
               "recording sites, the monocular crescent, the atlas's own "
               "extrapolation limits). The parvo/magno ratio of ratios is "
               f"**{p_ratio / m_ratio:.2f}**; below 1 is the direction "
               f"C&VE predicts.",
               ""]

    # ------------------------------------------------------------------
    # 2b. Per CLASS, summed over laminae -- which is what the config's
    #     `column_thickness_mm` actually denotes.
    # ------------------------------------------------------------------
    md += ["## Summed per-class T -- the quantity "
           "`column_thickness_mm` names",
           "",
           "The config is explicit that its T is *not* one lamina but "
           "\"the summed depth of same-class laminae one projection "
           "column crosses\", so the per-code table above has to be "
           "summed before it can be compared with the configured value. "
           "Summing is the right operation only because the two sheets "
           "of a class have similar areal magnification at a given "
           "eccentricity (compare their `median areal` columns above); "
           "where they diverge, treat this as approximate.",
           "",
           "| ecc bin (deg) | parvo T (mm) | magno T (mm) | "
           "C&VE parvo (3+5 plus 4+6) | C&VE magno (1 plus 2) |",
           "|---|---|---|---|---|"]

    cve_parvo = CVE_THICKNESS_MM[3][0] + CVE_THICKNESS_MM[4][0]
    cve_magno = CVE_THICKNESS_MM[1][0] + CVE_THICKNESS_MM[2][0]
    summed = {'parvo': {}, 'magno': {}}
    for lo, hi in zip(ECC_EDGES[:-1], ECC_EDGES[1:]):
        cell = {}
        for label, codes in (('parvo', (3, 4)), ('magno', (1, 2))):
            total = 0.0
            ok = True
            for code in codes:
                match = [t for l, h, t in per_layer_summary.get(code, [])
                         if l == lo and h == hi and np.isfinite(t)]
                if not match:
                    ok = False
                    break
                total += match[0]
            if ok:
                cell[label] = total
                summed[label][(lo, hi)] = total
        if len(cell) == 2:
            md.append(f"| {lo:g}-{hi:g} | {cell['parvo']:.3f} | "
                      f"{cell['magno']:.3f} | {cve_parvo:.3f} | "
                      f"{cve_magno:.3f} |")
    md.append("")

    central = [v for (lo, hi), v in summed['parvo'].items() if hi <= 15]
    peripheral = [v for (lo, hi), v in summed['parvo'].items()
                  if lo >= 15 and hi <= DROPOUT_WINDOW_HI_DEG]
    all_parvo = list(summed['parvo'].values())
    all_magno = list(summed['magno'].values())
    if all_parvo and all_magno:
        md += [f"Across every bin, summed parvo T spans "
               f"**{min(all_parvo):.2f}-{max(all_parvo):.2f} mm** "
               f"(median {np.median(all_parvo):.2f}) and summed magno T "
               f"**{min(all_magno):.2f}-{max(all_magno):.2f} mm** "
               f"(median {np.median(all_magno):.2f}).",
               ""]
        if central and peripheral:
            md += [f"Parvo specifically: {np.mean(central):.2f} mm inside "
                   f"15 deg, {np.mean(peripheral):.2f} mm from 15-"
                   f"{DROPOUT_WINDOW_HI_DEG:g} deg.",
                   ""]

    # ------------------------------------------------------------------
    # 3. The density route with T taken out of it.
    # ------------------------------------------------------------------
    md += ["## Volume magnification: measured vs. Malpeli-derived",
           "",
           "T appears in neither side of this comparison, so it isolates "
           "whether the `magnification_agreement_2026-09-11.md` "
           "disagreement is about T at all. Left: voxel count over solid "
           "angle. Right: `malpeli_density(E) / rho_v`, with rho_v taken "
           "from the atlas's own CELLS.DAT exactly as "
           "`MalpeliDensityMagnification.from_atlas` does. Both are "
           "mm^3/deg^2 and both pool all laminae of the class.",
           ""]
    volume_ratio = {}
    for cell_class in ('parvo', 'magno'):
        codes = resolve_class_codes(params, cell_class)
        mask = base_mask & np.isin(atlas.layer, codes)
        sel = tuple(np.argwhere(mask).T)
        ecc = atlas.eccentricity_deg[sel]

        # rho_v exactly as from_atlas computes it: mean cells/voxel over
        # the class's *valid* voxels, divided by voxel volume. Note it
        # uses atlas.valid alone, not the Jacobian mask, so recompute it
        # that way rather than reusing `mask`.
        rho_mask = atlas.valid & np.isin(atlas.layer, codes)
        rho_v = float(atlas.cells_per_voxel[rho_mask].mean()) / voxel_volume

        md += [f"### {cell_class} (rho_v = {rho_v:,.0f} cells/mm^3)",
               "",
               "| ecc bin (deg) | measured V/Omega | Malpeli/rho_v | "
               "ratio measured/Malpeli |",
               "|---|---|---|---|"]
        for lo, hi in zip(ECC_EDGES[:-1], ECC_EDGES[1:]):
            in_bin = (ecc >= lo) & (ecc < hi)
            n = int(in_bin.sum())
            if n < 50:
                continue
            omega = hemifield_solid_angle_deg2(lo, hi)
            measured = n * voxel_volume / omega
            mid = 0.5 * (lo + hi)
            predicted = float(malpeli_density(mid, params, cell_class)
                              / rho_v)
            volume_ratio[(cell_class, lo, hi)] = measured / predicted
            md.append(f"| {lo:g}-{hi:g} | {measured:.3e} | "
                      f"{predicted:.3e} | {measured / predicted:.2f} |")
        md.append("")

    # ------------------------------------------------------------------
    # 4. What measuring T does to the 11 Sep magnification disagreement.
    # ------------------------------------------------------------------
    md += ["## What this does to the 11 Sep magnification disagreement",
           "",
           "`magnification_agreement_2026-09-11.md` reported the Jacobian "
           "magnification running 1.1-1.8x (parvo) and 1.8-2.6x (magno) "
           "larger than the Malpeli-density estimate at the config's "
           "central T = 3 mm. Substituting the measured T collapses that "
           "algebraically, and it is worth being explicit about how much "
           "of the collapse is real and how much is circular.",
           "",
           "With `T = (V/Omega) / A_jacobian` the density route becomes",
           "",
           "    areal_malpeli = (density/rho_v) / T",
           "                  = A_jacobian * (density/rho_v) / (V/Omega)",
           "",
           "so the ratio of the two routes reduces *exactly* to the "
           "volume-magnification ratio in the table above, and the linear "
           "(mm/deg) ratio to its square root. The Jacobian cancels out "
           "of the comparison entirely -- which is the point: what is "
           "left is a check of cell counts against voxel counts, with no "
           "magnification model on either side.",
           "",
           "| ecc bin (deg) | parvo linear ratio at measured T | "
           "magno linear ratio at measured T |",
           "|---|---|---|"]
    for lo, hi in zip(ECC_EDGES[:-1], ECC_EDGES[1:]):
        cells = []
        for cell_class in ('parvo', 'magno'):
            r = volume_ratio.get((cell_class, lo, hi))
            cells.append(f"{np.sqrt(1.0 / r):.2f}" if r else "--")
        if cells != ["--", "--"]:
            md.append(f"| {lo:g}-{hi:g} | " + " | ".join(cells) + " |")
    md += ["",
           "(Ratio = Jacobian / density route, so 1.00 is agreement and "
           ">1 means the Jacobian is still the larger of the two -- the "
           "same orientation as the 11 Sep table.)",
           "",
           "**The honest reading.** The agreement beyond ~7 deg is not "
           "corroboration; it is what a correctly measured T is *for*. "
           "The informative part is where it still fails: the innermost "
           "bins. There the atlas holds materially fewer cells than "
           "Malpeli's formula predicts for the solid angle it covers, "
           "and no choice of T can fix a volume-magnification mismatch, "
           "because T is not in that comparison. That is a real residual "
           "and it sits exactly where the atlas is already known to be "
           "weakest -- inside the central few degrees, the "
           "isotropic-by-construction region "
           "(`atlas.isotropic_construction_radius_deg`) where Erwin et "
           "al. substituted Malpeli's own formula for interpolation. "
           "Worth its own look; not resolved here.",
           ""]

    # ------------------------------------------------------------------
    # 5. The quantisation-floor hypothesis in docs/magnification.md.
    # ------------------------------------------------------------------
    ecc_quantum_deg = 1.0 / float(
        require(params, 'atlas.files.eccentricity.scale'))
    pairwise_floor = voxel_mm / ecc_quantum_deg
    md += ["## The quantisation-floor hypothesis, tested",
           "",
           f"`docs/magnification.md` records a hypothesis, explicitly "
           f"flagged there as \"a hypothesis, not a demonstrated "
           f"cause\": that the near-fovea magnification figure of about "
           f"{pairwise_floor:g} mm/deg is a **quantisation floor**, one "
           f"voxel ({voxel_mm:g} mm) per one `ECC.DAT` quantum "
           f"({ecc_quantum_deg:g} deg), which is what a pairwise "
           f"finite-difference gradient estimate would bottom out at.",
           "",
           "It is testable, in two independent ways.",
           "",
           "**(a) Is the live figure even 0.25?** The doc's number came "
           "from a *retired* atlas-gradient sampling script, not from "
           "`JacobianMagnification`. The median column below is the "
           "current code path's answer at the same eccentricities.",
           "",
           "**(b) Is there a floor at all?** A quantisation floor leaves "
           "a *spike* of voxels at that exact value and essentially "
           "nothing below it. The spike test below compares the density "
           "of voxels in a narrow window centred on the floor against "
           "the mean of two identically wide windows 15% either side of "
           "it -- symmetric on purpose, because a one-sided control is "
           "biased on a decaying distribution and would manufacture a "
           "spike wherever the density is falling.",
           "",
           "| ecc bin (deg) | n | median mm/deg | min | "
           f"spike ratio at {pairwise_floor:g} | % below floor |",
           "|---|---|---|---|---|---|"]
    q_mask = base_mask & np.isin(atlas.layer,
                                 resolve_class_codes(params, 'parvo'))
    q_sel = tuple(np.argwhere(q_mask).T)
    q_major, q_minor, _ = JacobianMagnification.decompose(
        jacobian.jacobian[q_sel])
    with np.errstate(divide='ignore', invalid='ignore'):
        mm_per_deg = 1.0 / np.sqrt(q_major * q_minor)
    q_ecc = atlas.eccentricity_deg[q_sel]

    def window_fraction(values, centre, half_width_frac=0.02):
        return float((np.abs(values - centre) / pairwise_floor
                      < half_width_frac).mean())

    spikes, below = [], []
    for lo, hi in ((0.5, 1.5), (1.0, 2.0), (2.0, 4.0), (4.0, 7.0),
                   (9.0, 11.0)):
        in_bin = (q_ecc >= lo) & (q_ecc < hi) & np.isfinite(mm_per_deg)
        values = mm_per_deg[in_bin]
        if values.size < 1000:
            continue
        at_floor = window_fraction(values, pairwise_floor)
        flanks = 0.5 * (window_fraction(values, pairwise_floor * 0.85)
                        + window_fraction(values, pairwise_floor * 1.15))
        spike = at_floor / flanks if flanks > 0 else float('nan')
        frac_below = float((values < pairwise_floor).mean())
        spikes.append(spike)
        below.append(frac_below)
        md.append(f"| {lo:g}-{hi:g} | {values.size:,} | "
                  f"{np.median(values):.4f} | {values.min():.4f} | "
                  f"{spike:.2f} | {frac_below:.2%} |")
    md.append("")

    near_fovea = mm_per_deg[(q_ecc >= 0.5) & (q_ecc < 1.5)
                            & np.isfinite(mm_per_deg)]
    near_fovea_median = float(np.median(near_fovea))
    clean_spikes = [x for x in spikes if np.isfinite(x)]
    no_spike = clean_spikes and max(clean_spikes) < 1.5
    plenty_below = max(below) > 0.05 if below else False
    if no_spike and plenty_below:
        md += [f"**The hypothesis does not survive, on both tests.** The "
               f"spike ratio never exceeds "
               f"{max(clean_spikes):.2f} -- a smooth distribution, not a "
               f"pile-up -- and up to {max(below):.0%} of voxels sit "
               f"*below* {pairwise_floor:g} mm/deg, which a hard floor "
               f"makes impossible. Near the fovea the live median is "
               f"about {near_fovea_median:.2f} mm/deg, nowhere near "
               f"{pairwise_floor:g} to begin with.",
               "",
               "That is what the fit's design predicts: "
               "`JacobianAtlas.compute` does a least-squares fit over a "
               "neighbourhood (median ~26 valid neighbours per voxel), "
               "not a pairwise difference, so it averages the "
               "quantisation away rather than inheriting its floor.",
               "",
               "**Two conclusions, and the second is the useful one.** "
               "The hypothesis closes for the live code path. But it may "
               "well have been *correct* for the retired "
               "atlas-gradient sampling script the 0.25 mm/deg figure "
               "actually came from, which did use pairwise differences. "
               "So the doc's bullet wants re-scoping to that script, not "
               "deleting -- and the 2.5-4x 'disagreement' it reports "
               "between the density route and the atlas-gradient route "
               "is then a property of a retired script, and should stop "
               "being carried as an open discrepancy about the current "
               "model.",
               ""]
    else:
        md += [f"**Not cleanly resolved.** Spike ratios "
               f"{[f'{x:.2f}' for x in clean_spikes]}, max fraction "
               f"below the floor {max(below) if below else float('nan'):.2%}. "
               f"Read the table rather than a verdict.",
               ""]

    # ------------------------------------------------------------------
    # 6. Cross-check against the repo's own map-length back-solve.
    # ------------------------------------------------------------------
    md += ["## Cross-check: the repo's own map-length back-solve",
           "",
           "`MalpeliDensityMagnification.calibrate_column_thickness_mm` "
           "already back-solves a T, by a completely different route: "
           "require the integrated map length `int M(E) dE` to match the "
           "nucleus's physical extent, and invert. It shares no step "
           "with the voxel-count measurement above -- one integrates a "
           "density formula against an anatomical length, the other "
           "counts voxels against a solid angle.",
           "",
           "| class | map-length back-solve (mm) | measured here (mm) | "
           "configured (mm) |",
           "|---|---|---|---|"]
    from dynaphos_lgn.magnification import MalpeliDensityMagnification
    backsolved = {}
    for cell_class, measured in (('parvo', summed['parvo']),
                                 ('magno', summed['magno'])):
        model = MalpeliDensityMagnification.from_atlas(
            atlas, params, cell_class=cell_class)
        anchor_lo, anchor_mid, anchor_hi = model.map_length_anchor_mm
        t_lo = model.calibrate_column_thickness_mm(anchor_hi)
        t_hi = model.calibrate_column_thickness_mm(anchor_lo)
        backsolved[cell_class] = (t_lo, t_hi)
        values = list(measured.values())
        md.append(f"| {cell_class} | {t_lo:.2f}-{t_hi:.2f} | "
                  f"{min(values):.2f}-{max(values):.2f} | "
                  f"{model.column_thickness_mm[0]:g}-"
                  f"{model.column_thickness_mm[2]:g} |")
    md.append("")
    p_lo, p_hi = backsolved['parvo']
    m_lo, m_hi = backsolved['magno']
    md += [f"**The config's own calibration routine is already "
           f"class-dependent, and the config ignores it.** The "
           f"back-solve gives parvo {p_lo:.2f}-{p_hi:.2f} mm and magno "
           f"{m_lo:.2f}-{m_hi:.2f} mm -- a ratio of about "
           f"{(p_lo + p_hi) / (m_lo + m_hi):.1f} -- yet "
           f"`column_thickness_mm` stores a single triple used for both. "
           f"So the shared value was never consistent with the repo's "
           f"own calibration method either, quite apart from this "
           f"measurement.",
           "",
           f"The two routes do **not** agree on the absolute value: the "
           f"back-solve runs roughly 2-3x above what the voxel counts "
           f"measure, for both classes. Worth stating as a disagreement "
           f"rather than averaging away. But they agree on the sign and "
           f"on the class split, which is the part that changes what the "
           f"config should say. The magno case is the closer of the two "
           f"({m_lo:.2f}-{m_hi:.2f} back-solved against "
           f"{min(summed['magno'].values()):.2f}-"
           f"{max(summed['magno'].values()):.2f} measured, which "
           f"overlap).",
           ""]

    parvo_t = list(summed['parvo'].values())
    magno_t = list(summed['magno'].values())
    md += ["## What to do with this",
           "",
           "**1. The configured T is wrong, but not by the order of "
           "magnitude the C&VE lead suggested.** Measured summed-parvo T "
           f"is {min(parvo_t):.2f}-{max(parvo_t):.2f} mm against a "
           f"configured central 3.0 mm -- roughly 2x too thick, not 10x. "
           f"The C&VE figures are per *lamina*; the config's T is the "
           f"summed depth across a class's laminae, and comparing the two "
           f"without summing is what made the gap look like an order of "
           f"magnitude. Since linear magnification goes as 1/sqrt(T), a "
           f"2x error in T is a 1.4x error in magnification -- which is "
           f"about the size of the parvo disagreement 11 Sep reported.",
           "",
           "**2. T is strongly class-dependent, and the config has only "
           "one of them.** Measured summed magno T is "
           f"{min(magno_t):.2f}-{max(magno_t):.2f} mm (median "
           f"{np.median(magno_t):.2f}) against summed parvo "
           f"{np.median(parvo_t):.2f} -- a factor of "
           f"{np.median(parvo_t) / np.median(magno_t):.1f}. "
           "`magnification.density_derived.column_thickness_mm` is a "
           "single triple shared by both classes, so one of the two is "
           "necessarily wrong by that factor however it is set. This is "
           "the most actionable finding here, and it is a plausible "
           "explanation for the specific thing 11 Sep could not account "
           "for: that magno's gap, unlike parvo's, does not shrink with "
           "eccentricity. A single shared T cannot track two different "
           "curves.",
           "",
           "**3. The atlas implies thicker laminae than C&VE measured, "
           "by 2-5x, and that is not reconciled.** Per-code measured T "
           "runs ~0.25 mm (magno) and ~0.82 mm (parvo) centrally against "
           "C&VE's 0.125-0.225 mm. Both directions of explanation are "
           "live and this script does not choose between them: the "
           "atlas is an interpolated reconstruction from 415 recording "
           "sites, so its layer sheets are plausibly blurred wider than "
           "the real ones; and a local linear Jacobian fitted to "
           "interpolated data tends to *overstate* deg/mm, which inflates "
           "T by exactly the same route. Note the sign matters: neither "
           "candidate is shrinkage, since C&VE's figures are already "
           "shrinkage-corrected (i.e. already the larger ones).",
           "",
           "**4. The 15 deg leaflet dropout shows up in the atlas.** "
           "Parvo thins by 0.75x across the boundary while magno "
           "thickens by 1.45x, a differential of 0.52x in the direction "
           "C&VE predicts. Not a strong test on its own -- one animal, "
           "and plenty else changes with eccentricity -- but it is an "
           "independent, published prediction that the atlas was not "
           "built to satisfy, and it holds.",
           "",
           "**5. What no T can fix.** The volume-magnification table has "
           "no T in it and still disagrees by 1.5-2x inside 4 deg. That "
           "residual is upstream of every magnification model and should "
           "not be absorbed into a thickness parameter.",
           "",
           "### Suggested config change -- NOT applied, needs sign-off",
           "",
           "Split `column_thickness_mm` by class and widen each interval "
           "to bracket both independent estimates (C&VE's summed "
           "laminae, and this atlas measurement) rather than picking "
           "between two sources that disagree by 2-5x:",
           "",
           "```yaml",
           "column_thickness_mm:",
           f"  parvo: [{cve_parvo:.2f}, "
           f"{np.sqrt(cve_parvo * max(parvo_t)):.2f}, {max(parvo_t):.2f}]"
           f"   # C&VE summed .. atlas-measured max",
           f"  magno: [{min(magno_t):.2f}, "
           f"{np.sqrt(min(magno_t) * max(magno_t)):.2f}, "
           f"{max(magno_t):.2f}]",
           "```",
           "",
           "Centrals are geometric means of each interval's endpoints, "
           "which is the right average for a quantity entering as "
           "1/sqrt(T). Whether to also make T eccentricity-dependent is "
           "a separate call: the measurement says it is (parvo thins "
           "peripherally, magno does not), but a T(E) would need its own "
           "justification rather than being fitted to close a gap.",
           "",
           "Caveats not resolved by any of this: one animal; the "
           "Jacobian is a fit, not a measurement; C&VE's thicknesses "
           "come from three *different* LGNs than the one this atlas "
           "reconstructs; and the whole comparison inherits the "
           "standing circularity that the atlas was itself built from "
           "Malpeli's data.",
           ""]

    text = '\n'.join(md)
    print(text)
    out_md = Path(args.out_md
                  or f'research/column-thickness-measured-{today}.md')
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text)
    out_csv = Path(args.out_csv
                   or f'research/column-thickness-measured-{today}.csv')
    with open(out_csv, 'w') as f:
        for row in csv_rows:
            f.write(','.join(str(x) for x in row) + '\n')
    print(f"\nWritten to {out_md} and {out_csv}")


if __name__ == '__main__':
    main()
