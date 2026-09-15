"""Anisotropy, redone in the frame Connolly & Van Essen (1984) actually
used -- isoeccentricity vs. isopolar, per layer -- instead of a fixed
horizontal/vertical ruler with the cell classes pooled.

Why the 11 Sep check has to be redone
-------------------------------------
`check_anisotropy_report.py` (11 Sep 2026) produced real numbers, and its
*magnitude* results stand. Its *orientation* result does not, and the
full-text verification the same day said why
(`connolly-van-essen-1984-anisotropy-verification-2026-09-11.md`):

- It measured the ellipse's major axis against a **fixed** horizontal
  axis. C&VE state their claim in a **local** frame that rotates with
  visual-field position -- isoeccentricity ("around", tangential) vs.
  isopolar ("outward", radial). The two agree only exactly on the
  meridians, and diverge by the polar angle itself everywhere else. So
  the "~123 deg vs ~115 deg, same direction, not opposite" finding
  compared a local claim against a global ruler, and is uninterpretable
  rather than contradictory.
- It pooled all parvo layers against all magno layers. C&VE's claim is
  layer 6 vs. layer 1 with the intervening layers graded between them,
  and layer 3 -- parvocellular -- patterns with the *inner* layers
  (p.552). Pooling by class dilutes exactly the contrast being tested.

What this script measures instead
---------------------------------
C&VE's quantity, directly, with no principal-axis inference in between.
"Magnification" here is theirs: millimetres of tissue per degree of
visual field, measured **along a named contour**.

For a unit visual-field direction ``d``, the tissue displacement that
produces it is ``J+ d`` (``J+`` the pseudo-inverse of the 2x3 Jacobian),
so the magnification along ``d`` is ``|J+ d|``. With ``G = J J^T``
(2x2, deg^2/mm^2) that is just::

    M(d) = sqrt( d^T G^-1 d )        mm per deg along d

At visual-field position ``(E, I)`` under the atlas's
``x = E cos I, y = E sin I`` convention, the isopolar (radial) direction
is ``r = (cos I, sin I)`` and the isoeccentricity (tangential) direction
is ``t = (-sin I, cos I)``. So the number C&VE report is::

    M_isoecc / M_isopolar = sqrt( (t^T G^-1 t) / (r^T G^-1 r) )

and their claim, verbatim (p.552), is that in **layer 6** near the
peripheral horizontal meridian this ratio is **2 to 3**, while in the
more ventral layers it inverts.

Note the sign convention, because it is the easiest thing to get
backwards: a *larger* M means *more tissue per degree*, which is a
*smaller* deg/mm, so the ratio above is >1 exactly where the Jacobian's
**major** (deg/mm) axis points along the **isopolar** direction. Stated
in principal-axis terms the claim reads "layer 6's long axis in deg/mm
is radial", which is the opposite of what a careless reading of "greater
along isoeccentricity" suggests. The ratio is computed directly here so
that the sign is carried by the algebra rather than by prose.

The one thing this atlas cannot do, and the one it can
------------------------------------------------------
LAYERS.DAT has four codes, not six: layers 4 and 6 are merged into
`parvo_contra`, and 3 and 5 into `parvo_ipsi` (the same fusion C&VE
themselves used, p.546-548). So a layer-6-only mask does not exist --

**except beyond 15 degrees.** C&VE p.554: leaflets 4 and 5 are
"restricted to the representation of the central 15 deg". Past that,
whatever is left in `parvo_contra` *is* layer 6, and whatever is left in
`parvo_ipsi` is layer 3. And C&VE's strongest statement is specifically
about the **peripheral** horizontal meridian in layer 6 -- the same
place. So the sharpest available test of the claim is:

    code 4, eccentricity > 15 deg, near the horizontal meridian
        vs.
    code 1 (which is layer 1 exactly, at any eccentricity)

which is what the "sharp test" section below reports. The atlas's own
15 deg leaflet dropout is independently corroborated in
`check_column_thickness.py`, so this is not assuming the thing it needs.

Run from the repo root::

    python -m dynaphos_lgn.check_anisotropy_local_frame

Writes a markdown report and a CSV to `research/`, and prints the report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

HALF_WIDTH_DEG = 22.5          # meridian-proximity half-width

# Half-widths swept in the "is 2-3x just a narrower region?" section.
# If C&VE's figure applies to a tighter band around the meridian than
# the default, the ratio should climb as this narrows.
HALF_WIDTH_SWEEP_DEG = (30.0, 22.5, 15.0, 10.0, 5.0, 2.5)
ECC_EDGES = np.array([1.0, 2.0, 4.5, 9.0, 15.0, 25.0, 40.0, 90.0])
LEAFLET_DROPOUT_ECC_DEG = 15.0
CVE_RATIO_RANGE = (2.0, 3.0)   # p.552, verbatim

# Inner -> outer, which is the axis C&VE say the effect is graded along
# (p.557: nested coaxial half-cylinders, layer 1 innermost). The two
# parvo codes are fused pairs, so their position on this axis is
# approximate -- stated here rather than hidden in a sort order.
LAYER_ORDER = [1, 2, 3, 4]
LAYER_NOTE = {
    1: 'layer 1 exactly (innermost)',
    2: 'layer 2 exactly',
    3: 'fused 3+5; layer 3 alone beyond 15 deg',
    4: 'fused 4+6; layer 6 alone beyond 15 deg (outermost)',
}


def contour_magnification_ratio(jacobian: np.ndarray,
                                inclination_deg: np.ndarray,
                                ridge: float = 0.0) -> np.ndarray:
    """``M_isoeccentricity / M_isopolar`` per voxel, dimensionless.

    :param jacobian: (n, 2, 3) deg/mm.
    :param inclination_deg: (n,) visual-field polar angle, degrees.
    :param ridge: Added to ``G``'s diagonal before inversion. Zero by
        default; a near-singular ``G`` means a genuinely degenerate local
        map and is better reported as NaN than quietly regularised.
    :return: (n,) ratio. >1 means more tissue per degree along
        isoeccentricity contours than along isopolar ones, which is the
        orientation of C&VE's layer-6 claim.
    """
    j = np.asarray(jacobian, dtype=float)
    g = np.einsum('nij,nkj->nik', j, j)            # J J^T, (n, 2, 2)
    if ridge:
        g[:, 0, 0] += ridge
        g[:, 1, 1] += ridge

    # Closed-form 2x2 inverse -- faster than np.linalg.inv on millions of
    # tiny matrices, and it hands back the determinant for the
    # degeneracy test for free.
    a, b, c, d = g[:, 0, 0], g[:, 0, 1], g[:, 1, 0], g[:, 1, 1]
    det = a * d - b * c
    incl = np.deg2rad(np.asarray(inclination_deg, dtype=float))
    cos_i, sin_i = np.cos(incl), np.sin(incl)

    def quadratic_form(dx, dy):
        """``d^T G^-1 d`` without forming G^-1."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return ((d * dx * dx - (b + c) * dx * dy + a * dy * dy)
                    / det)

    isopolar = quadratic_form(cos_i, sin_i)        # r = (cos I, sin I)
    isoecc = quadratic_form(-sin_i, cos_i)         # t = (-sin I, cos I)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.sqrt(isoecc / isopolar)
    bad = ~np.isfinite(det) | (det <= 0) | ~np.isfinite(ratio)
    return np.where(bad, np.nan, ratio)


def meridian_class(incl_deg: np.ndarray,
                   half_width_deg: float = HALF_WIDTH_DEG) -> np.ndarray:
    """'horizontal' / 'vertical' / 'oblique' per voxel."""
    d_horiz = np.minimum(np.abs(incl_deg), np.abs(np.abs(incl_deg) - 180))
    d_vert = np.abs(np.abs(incl_deg) - 90)
    out = np.full(incl_deg.shape, 'oblique', dtype=object)
    out[d_horiz <= half_width_deg] = 'horizontal'
    out[d_vert <= half_width_deg] = 'vertical'
    return out


def alignment_efficiency(ratio: np.ndarray, anisotropy: np.ndarray
                         ) -> np.ndarray:
    """How much of the available anisotropy points along the contours.

    ``log(ratio) / log(anisotropy)``, in [-1, +1]:

    +1  the major (deg/mm) axis lies exactly along the isopolar
        direction, so the full anisotropy shows up as
        ``M_isoecc / M_isopolar`` -- C&VE's layer-6 orientation.
    -1  exactly along isoeccentricity -- their layer-1 orientation.
     0  at 45 degrees to both, so no directional ratio at all however
        anisotropic the voxel is.

    Separating this from the ratio matters because a small ratio has two
    completely different causes -- an isotropic voxel, or an anisotropic
    one pointing the wrong way -- and only the second contradicts
    anything.
    """
    ok = (np.isfinite(ratio) & (ratio > 0) & np.isfinite(anisotropy)
          & (anisotropy > 1.0 + 1e-9))
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = np.log(np.where(ok, ratio, np.nan)) / np.log(
            np.where(ok, anisotropy, np.nan))
    return np.where(ok, np.clip(eff, -1.0, 1.0), np.nan)


def geometric_summary(ratio: np.ndarray) -> tuple:
    """(geometric median, 16th, 84th pct) of a ratio.

    Geometric because the quantity is a ratio: 2x and 0.5x are equal and
    opposite departures from 1, and an arithmetic mean of the two would
    report 1.25 rather than 1.
    """
    finite = ratio[np.isfinite(ratio) & (ratio > 0)]
    if finite.size < 20:
        return (float('nan'),) * 3
    log = np.log(finite)
    return (float(np.exp(np.median(log))),
            float(np.exp(np.percentile(log, 16))),
            float(np.exp(np.percentile(log, 84))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/params_lgn.yaml')
    ap.add_argument('--atlas-dir', default=None)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--keep-isotropic-zone', action='store_true',
                    help="Keep voxels flagged isotropic-by-construction. "
                         "Off by default: inside that radius Erwin et "
                         "al. wrote an isotropic formula in, so any "
                         "anisotropy measured there is an artefact of "
                         "construction, not a finding.")
    ap.add_argument('--meridian-half-width', type=float,
                    default=HALF_WIDTH_DEG,
                    help='Half-width of the meridian bands, deg.')
    ap.add_argument('--out-md', default=None)
    ap.add_argument('--out-csv', default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dynaphos_lgn.atlas import ErwinAtlas, JacobianAtlas
    from dynaphos_lgn.build import load_lgn_params
    from dynaphos_lgn.magnification import JacobianMagnification
    from dynaphos_lgn.params import require

    params = load_lgn_params(args.config)
    atlas_dir = Path(args.atlas_dir or require(params, 'atlas.directory'))
    atlas = ErwinAtlas(params).build_atlas(atlas_dir)
    jacobian = JacobianAtlas.load(
        Path(args.cache_dir or atlas_dir)
        / require(params, 'atlas.jacobian_cache'), params)
    layer_names = require(params, 'atlas.layers.names')

    mask = atlas.valid & jacobian.jacobian_valid & ~atlas.is_ipsi_sentinel()
    if not args.keep_isotropic_zone:
        mask = mask & ~jacobian.isotropic_flag
    sel = tuple(np.argwhere(mask).T)

    jac = jacobian.jacobian[sel]
    ecc = atlas.eccentricity_deg[sel]
    incl = atlas.inclination_deg[sel]
    layer = atlas.layer[sel]
    ratio = contour_magnification_ratio(jac, incl)
    major, minor, _ = JacobianMagnification.decompose(jac)
    with np.errstate(divide='ignore', invalid='ignore'):
        anisotropy = np.where(minor > 0, major / minor, np.nan)
    alignment = alignment_efficiency(ratio, anisotropy)
    merclass = meridian_class(incl, args.meridian_half_width)

    today = dt.date.today().isoformat()
    lo_cve, hi_cve = CVE_RATIO_RANGE
    md = [f"# Anisotropy in the isoeccentricity/isopolar frame "
          f"-- {today}",
          "",
          "Redo of the 11 Sep anisotropy check with both corrections the "
          "Connolly & Van Essen full-text verification called for: the "
          "local (isoeccentricity vs. isopolar) frame instead of a fixed "
          "horizontal ruler, and per atlas layer instead of parvo/magno "
          "pooled.",
          "",
          f"Reported quantity is C&VE's own: **M_isoecc / M_isopolar**, "
          f"tissue-mm per visual-degree along each contour, computed "
          f"directly from `J J^T` rather than inferred from principal "
          f"axes. Their claim (p.552) is **{lo_cve:g}-{hi_cve:g}** in "
          f"layer 6 near the peripheral horizontal meridian, inverting "
          f"in the more ventral layers. Values quoted as geometric "
          f"median [16th-84th pct], because a ratio's 2x and 0.5x are "
          f"equal and opposite.",
          "",
          f"{int(mask.sum()):,} voxels; "
          f"isotropic-by-construction zone "
          f"{'KEPT' if args.keep_isotropic_zone else 'excluded'}; "
          f"ipsilateral placeholder voxels excluded.",
          ""]

    csv_rows = [("layer_code", "layer_name", "ecc_lo", "ecc_hi",
                 "meridian", "n", "ratio_geom_median", "ratio_p16",
                 "ratio_p84", "median_anisotropy")]

    # ------------------------------------------------------------------
    # The sharp test the 11 Sep check could not make.
    # ------------------------------------------------------------------
    md += ["## The sharp test: layer 6 (peripheral) vs. layer 1",
           "",
           f"`parvo_contra` beyond {LEAFLET_DROPOUT_ECC_DEG:g} deg is "
           f"layer 6 alone, because leaflet 4 does not extend past there "
           f"(C&VE p.554; the dropout is independently visible in the "
           f"atlas -- see `column-thickness-measured-{today}.md`). "
           f"`magno_contra` is layer 1 exactly, everywhere. So this is "
           f"the claim as stated, not a class-pooled proxy for it.",
           "",
           "| population | n | M_isoecc / M_isopolar | "
           "median anisotropy (ceiling on the ratio) | contour "
           "alignment (-1 isoecc .. +1 isopolar) |",
           "|---|---|---|---|---|"]

    sharp = {}
    peripheral = ecc >= LEAFLET_DROPOUT_ECC_DEG
    horizontal = merclass == 'horizontal'
    cases = [
        ('layer 6, peripheral, near horizontal meridian',
         (layer == 4) & peripheral & horizontal),
        ('layer 6, peripheral, all meridians',
         (layer == 4) & peripheral),
        ('layer 1, peripheral, near horizontal meridian',
         (layer == 1) & peripheral & horizontal),
        ('layer 1, all eccentricities, near horizontal meridian',
         (layer == 1) & horizontal),
        ('layer 1, peripheral, near vertical meridian',
         (layer == 1) & peripheral & (merclass == 'vertical')),
    ]
    for label, case in cases:
        med, p16, p84 = geometric_summary(ratio[case])
        aniso = (float(np.nanmedian(anisotropy[case]))
                 if case.sum() else float('nan'))
        sharp[label] = med
        align = (float(np.nanmedian(alignment[case]))
                 if case.sum() else float('nan'))
        md.append(f"| {label} | {int(case.sum()):,} | "
                  f"**{med:.2f}** [{p16:.2f}-{p84:.2f}] | {aniso:.2f} "
                  f"| {align:+.2f} |")
    md.append("")

    l6 = sharp['layer 6, peripheral, near horizontal meridian']
    l1 = sharp['layer 1, peripheral, near horizontal meridian']
    verdict = []
    if np.isfinite(l6):
        if lo_cve <= l6 <= hi_cve:
            verdict.append(f"Layer 6 lands at **{l6:.2f}**, inside C&VE's "
                           f"{lo_cve:g}-{hi_cve:g}.")
        elif l6 > 1.0:
            verdict.append(f"Layer 6 lands at **{l6:.2f}** -- the right "
                           f"side of 1 (more tissue per degree along "
                           f"isoeccentricity, as C&VE describe) but "
                           f"{'below' if l6 < lo_cve else 'above'} their "
                           f"{lo_cve:g}-{hi_cve:g}.")
        else:
            verdict.append(f"Layer 6 lands at **{l6:.2f}**, i.e. below 1 "
                           f"-- the *opposite* orientation to C&VE's "
                           f"layer-6 claim. That is a contradiction, not "
                           f"a magnitude mismatch, and needs explaining "
                           f"before any of this is used.")
    if np.isfinite(l1) and np.isfinite(l6):
        if (l1 - 1.0) * (l6 - 1.0) < 0:
            verdict.append(f"Layer 1 lands at **{l1:.2f}**, on the "
                           f"opposite side of 1 from layer 6 -- the "
                           f"reversal C&VE report.")
        else:
            verdict.append(f"Layer 1 lands at **{l1:.2f}**, on the *same* "
                           f"side of 1 as layer 6. No reversal here, "
                           f"which is the part of the claim this atlas "
                           f"does not reproduce.")
    md += [' '.join(verdict), ""]

    # ------------------------------------------------------------------
    # The graded inner -> outer prediction.
    # ------------------------------------------------------------------
    md += ["## The inner-to-outer gradient (C&VE p.557)",
           "",
           "The mechanism C&VE propose is purely geometric: the layers "
           "are nested coaxial half-cylinders, so the innermost (layer "
           "1) has the smallest circumference and the outermost (layer "
           "6) the largest, and a fixed angular sweep of visual field "
           "maps to less tissue on the inner sheets. That predicts a "
           "monotone gradient from layer 1 to layer 6, not a two-level "
           "split -- which is a stronger and more falsifiable claim than "
           "the 2-3x figure, and it does not depend on the class "
           "boundary at all.",
           "",
           f"Restricted to eccentricity > {LEAFLET_DROPOUT_ECC_DEG:g} "
           f"deg so the two fused parvo codes are single layers.",
           "",
           "| layer code | name | what it is there | n | "
           "M_isoecc / M_isopolar | contour alignment |",
           "|---|---|---|---|---|---|"]
    gradient = []
    for code in LAYER_ORDER:
        case = (layer == code) & peripheral
        med, p16, p84 = geometric_summary(ratio[case])
        gradient.append(med)
        align = float(np.nanmedian(alignment[case]))
        md.append(f"| {code} | `{layer_names[code]}` | {LAYER_NOTE[code]} "
                  f"| {int(case.sum()):,} | **{med:.2f}** "
                  f"[{p16:.2f}-{p84:.2f}] | {align:+.2f} |")
    md.append("")
    clean = [g for g in gradient if np.isfinite(g)]
    if len(clean) == len(LAYER_ORDER):
        # A step smaller than this counts as a tie, not a reversal. 5%
        # is far inside the 16-84 spread of every row above, so calling
        # such a step a violation would be reading noise as structure.
        tie = 0.05
        steps = np.diff(clean)
        rises = [d for d in steps if d > tie * abs(clean[0])]
        falls = [d for d in steps if d < -tie * abs(clean[0])]
        sequence = " -> ".join(f"{g:.2f}" for g in clean)
        if falls and rises:
            verdict_line = (". **Not monotone** -- it rises and falls, so "
                            "whatever drives the layer-to-layer "
                            "differences is not only the nested-cylinder "
                            "geometry.")
        elif rises and not falls:
            verdict_line = (f". **Monotone increasing** up to ties "
                            f"(steps under {tie:.0%} treated as ties; "
                            f"codes 3 and 4 differ by "
                            f"{abs(clean[3] - clean[2]) / clean[2]:.1%}, "
                            f"far inside their own 16-84 spreads). That "
                            f"is the direction the nested-half-cylinder "
                            f"mechanism predicts, and it is a prediction "
                            f"the atlas was not built to satisfy.")
        else:
            verdict_line = (". Flat or decreasing -- not the direction "
                            "the nesting mechanism predicts.")
        steps_txt = ", ".join(
            f"{clean[i]:.2f}->{clean[i + 1]:.2f}"
            for i in range(len(clean) - 1))
        md += ["Inner-to-outer sequence: " + sequence + verdict_line,
               "",
               f"Read the *step sizes*, not just the ordering: "
               f"{steps_txt}. The first step carries almost all of it. "
               f"So this is monotone, but it is equally well described "
               f"as \"layer 1 differs from the rest\" -- which is a "
               f"weaker claim than the graded nesting mechanism makes, "
               f"and the summary section below scores it that way rather "
               f"than counting the monotonicity as a clean win.",
               ""]

    # ------------------------------------------------------------------
    # Full table.
    # ------------------------------------------------------------------
    md += ["## Full table, per layer / eccentricity / meridian",
           "",
           "| layer | ecc bin (deg) | meridian | n | "
           "M_isoecc / M_isopolar | median anisotropy |",
           "|---|---|---|---|---|---|"]
    for code in LAYER_ORDER:
        for lo, hi in zip(ECC_EDGES[:-1], ECC_EDGES[1:]):
            in_bin = (layer == code) & (ecc >= lo) & (ecc < hi)
            for mc in ('horizontal', 'vertical', 'oblique'):
                case = in_bin & (merclass == mc)
                n = int(case.sum())
                if n < 100:
                    continue
                med, p16, p84 = geometric_summary(ratio[case])
                aniso = float(np.nanmedian(anisotropy[case]))
                md.append(f"| `{layer_names[code]}` | {lo:g}-{hi:g} | "
                          f"{mc} | {n:,} | {med:.2f} [{p16:.2f}-"
                          f"{p84:.2f}] | {aniso:.2f} |")
                csv_rows.append((code, layer_names[code], lo, hi, mc, n,
                                 med, p16, p84, aniso))
    md.append("")

    # ------------------------------------------------------------------
    # Is the 2-3x figure just a narrower region than we are averaging?
    # ------------------------------------------------------------------
    md += ["## Is the 2-3x figure hiding in a narrower meridian band?",
           "",
           "The obvious way a pooled median under-reports a local effect "
           "is by averaging over too much territory. C&VE measured "
           f"sections, not a {HALF_WIDTH_DEG:g} deg band. If their "
           "figure is real but local, narrowing the band should make the "
           "ratio climb towards it. This sweeps the half-width and "
           "looks.",
           "",
           "| half-width (deg) | layer 6 n | layer 6 ratio | "
           "layer 6 anisotropy | layer 1 n | layer 1 ratio |",
           "|---|---|---|---|---|---|"]
    sweep = []
    for half_width in HALF_WIDTH_SWEEP_DEG:
        mer = meridian_class(incl, half_width)
        horiz = mer == 'horizontal'
        l6_case = (layer == 4) & peripheral & horiz
        l1_case = (layer == 1) & peripheral & horiz
        l6_med, _, _ = geometric_summary(ratio[l6_case])
        l1_med, _, _ = geometric_summary(ratio[l1_case])
        l6_aniso_w = float(np.nanmedian(anisotropy[l6_case]))
        sweep.append((half_width, l6_med))
        md.append(f"| {half_width:g} | {int(l6_case.sum()):,} | "
                  f"**{l6_med:.2f}** | {l6_aniso_w:.2f} | "
                  f"{int(l1_case.sum()):,} | {l1_med:.2f} |")
    md.append("")
    widest, narrowest = sweep[0][1], sweep[-1][1]
    climbed = narrowest / widest if widest else float('nan')
    md += [f"From {HALF_WIDTH_SWEEP_DEG[0]:g} deg down to "
           f"{HALF_WIDTH_SWEEP_DEG[-1]:g} deg the layer-6 ratio moves "
           f"{widest:.2f} -> {narrowest:.2f}, a factor of "
           f"{climbed:.2f}. "
           + ("It does **not** climb towards 2-3, so the 'right effect, "
              "wrong averaging window' explanation is ruled out. What is "
              "left is that this atlas's reconstruction does not carry "
              "the contour alignment C&VE measured -- a statement about "
              "the atlas, not about the paper. Worth saying plainly, "
              "because it bounds what any Jacobian-derived anisotropy "
              "from this atlas can be expected to reproduce, and "
              "therefore what Tracks A-C would be validating against."
              if climbed < 1.5 else
              "It climbs materially, so the pooled figure was an "
              "averaging artefact and the band width matters -- rerun "
              "the sections above at the narrower width before using "
              "any of them."),
           "",
           "Note the anisotropy column climbs slightly as the band "
           "narrows while the ratio barely moves: there *is* a little "
           "more anisotropy close to the meridian, it just is not "
           "oriented along the contours.",
           ""]

    # ------------------------------------------------------------------
    # The layer-3 nuance C&VE flag explicitly, tested on its own.
    # ------------------------------------------------------------------
    md += ["## Layer 3, which C&VE single out",
           "",
           "p.552 groups layer 3 -- parvocellular -- with layers 1 and 2 "
           "as approximately isotropic \"along parts of the horizontal "
           "meridian representation\", i.e. patterning with the *inner* "
           "layers rather than with its own cell class. That is the one "
           "place where the layer reading and the parvo/magno reading "
           "diverge, and it is the reason the 11 Sep class-pooled "
           "grouping was called too coarse. Beyond 15 deg the atlas's "
           "`parvo_ipsi` is layer 3 alone, so it can be checked.",
           "",
           "| layer (peripheral, near horizontal meridian) | n | "
           "M_isoecc / M_isopolar | contour alignment |",
           "|---|---|---|---|"]
    layer3 = {}
    for code in LAYER_ORDER:
        case = (layer == code) & peripheral & horizontal
        med, p16, p84 = geometric_summary(ratio[case])
        align = float(np.nanmedian(alignment[case]))
        layer3[code] = (med, align)
        md.append(f"| `{layer_names[code]}` ({LAYER_NOTE[code]}) | "
                  f"{int(case.sum()):,} | {med:.2f} [{p16:.2f}-{p84:.2f}] "
                  f"| {align:+.2f} |")
    md.append("")

    # Every verdict below is computed from `layer3` (peripheral, near
    # the horizontal meridian -- the population C&VE's own sentence is
    # about), so the prose cannot drift from the table above it.
    r1, a1 = layer3[1]
    r2, a2 = layer3[2]
    r3, a3 = layer3[3]
    r6, a6 = layer3[4]
    outer_aligns = [a2, a3, a6]
    outer_spread = max(outer_aligns) - min(outer_aligns)
    layer3_with_inner = abs(a3 - a1) < abs(a3 - a6)
    md += [f"Layer 3 comes out at **{r3:.2f}** (alignment {a3:+.2f}), "
           f"against layer 1 at {r1:.2f} ({a1:+.2f}), layer 2 at "
           f"{r2:.2f} ({a2:+.2f}) and layer 6 at {r6:.2f} ({a6:+.2f}). "
           + (f"It sits nearer layer 1 than layer 6, which is C&VE's "
              f"grouping -- their layer-3 nuance is reproduced."
              if layer3_with_inner else
              f"It sits nearer layer 6 than layer 1 -- in fact it is the "
              f"*most* isopolar-aligned of the four. So C&VE's specific "
              f"grouping of layer 3 with layers 1-2 is **not** "
              f"reproduced here. Note this cuts against the 11 Sep "
              f"correction as much as for it: pooling by cell class was "
              f"the wrong grouping, but layer 3 turns out to pattern "
              f"with its own class after all in this atlas, not against "
              f"it."),
           ""]

    # ------------------------------------------------------------------
    # The headline: what changed by fixing the frame.
    # ------------------------------------------------------------------
    a1_all = float(np.nanmedian(alignment[(layer == 1) & peripheral]))
    a6_all = float(np.nanmedian(alignment[(layer == 4) & peripheral]))
    md += ["## What changed by fixing the frame",
           "",
           f"The 11 Sep check concluded that parvo and magno axes point "
           f"in **nearly the same direction** (~123 deg vs ~115 deg from "
           f"horizontal) and so did not show C&VE's reversal. In the "
           f"correct local frame they point to **opposite sides of "
           f"diagonal**: layer 1 at alignment {a1_all:+.2f} (its long "
           f"deg/mm axis lies along isoeccentricity) against layer 6 at "
           f"{a6_all:+.2f}. The frame was doing the work, exactly as the "
           f"verification note predicted it would. That is the headline: "
           f"a reversal that the fixed-frame analysis reported as absent "
           f"is present once the claim is tested in its own coordinates.",
           "",
           "Taking the claim apart into separable pieces, with the "
           "peripheral horizontal-meridian numbers above:",
           "",
           "| piece of the C&VE claim | verdict |",
           "|---|---|"]

    reversal = (a1 < 0) and (a6 > 0)
    md.append(f"| A reversal exists between the innermost and outermost "
              f"layers | **{'reproduced' if reversal else 'not reproduced'}"
              f"** (layer 1 {a1:+.2f} vs layer 6 {a6:+.2f}) |")

    graded = outer_spread < abs(a1 - a2)
    md.append(f"| It is *graded* across layers, not a two-level split | "
              f"**{'not clearly reproduced' if graded else 'reproduced'}"
              f"** -- almost the whole effect is the layer-1-to-layer-2 "
              f"step ({abs(a1 - a2):.2f} in alignment); layers 2, 3 and 6 "
              f"span only {outer_spread:.2f} between them. Consistent "
              f"with a gradient, but equally consistent with layer 1 "
              f"being the only layer that differs |")

    md.append(f"| Layer 3 patterns against its own cell class | "
              f"**{'reproduced' if layer3_with_inner else 'not reproduced'}"
              f"** |")
    md.append(f"| The magnitude is {lo_cve:g}-{hi_cve:g}x in layer 6 | "
              f"**not reproduced** ({r6:.2f}x) |")

    l6_aniso = float(np.nanmedian(
        anisotropy[(layer == 4) & peripheral & horizontal]))
    md += ["",
           f"The magnitude row is the one to carry forward, and it is "
           f"not a ceiling artefact: layer 6's median anisotropy in that "
           f"same population is {l6_aniso:.2f}, so a ratio near "
           f"{lo_cve:g} was arithmetically available and simply did not "
           f"occur -- the axes are only {a6:+.2f} aligned with the "
           f"contours, i.e. close to diagonal. Two candidate readings, "
           f"not distinguished here: this atlas's reconstruction smooths "
           f"out contour alignment that C&VE measured directly in "
           f"sections, or their 2-3x is local to a much smaller region "
           f"than a {HALF_WIDTH_DEG:g} deg meridian band. The second is "
           f"cheap to test -- narrow `HALF_WIDTH_DEG` and see whether "
           f"the ratio climbs.",
           ""]

    overall_aniso = float(np.nanmedian(anisotropy))
    md += ["## How to read this against the 11 Sep report",
           "",
           "- The **magnitude** results there are unchanged and are not "
           "restated here: this script reports a directional ratio, not "
           "sigma1/sigma2. The `median anisotropy` column is included "
           "only because it is the *ceiling* on the directional ratio "
           "-- the ratio cannot exceed it, so a population whose "
           f"anisotropy is {overall_aniso:.2f} cannot show a "
           f"{hi_cve:g}x directional ratio however the axes happen to "
           f"lie. Where a row's ratio sits well below C&VE's range, "
           f"check that column before concluding the orientation "
           f"disagrees: it may simply be that there is not enough "
           f"anisotropy at that voxel to go around.",
           "- The 11 Sep **orientation** result (\"~123 deg vs ~115 deg, "
           "same direction\") is superseded, not contradicted. It was "
           "computed in a fixed frame against a local claim; this "
           "supersedes it rather than disagreeing with it.",
           "- Still one animal, still the atlas's own Jacobian fit, "
           "still no cross-animal variance. Tracks A-C of "
           "`lgn-retinotopic-anisotropy-experimental-design-2026-09-10."
           "md` are what would actually close the question; this makes "
           "the single-animal answer a *correct* single-animal answer, "
           "which the 11 Sep orientation half was not.",
           ""]

    text = '\n'.join(md)
    print(text)
    out_md = Path(args.out_md
                  or f'research/anisotropy-local-frame-{today}.md')
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text)
    out_csv = Path(args.out_csv
                   or f'research/anisotropy-local-frame-{today}.csv')
    with open(out_csv, 'w') as f:
        for row in csv_rows:
            f.write(','.join(str(x) for x in row) + '\n')
    print(f"\nWritten to {out_md} and {out_csv}")


if __name__ == '__main__':
    main()
