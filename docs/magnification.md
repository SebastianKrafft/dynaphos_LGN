# Magnification

Code: [`dynaphos_lgn/magnification.py`](../dynaphos_lgn/magnification.py).
Config: the `magnification:` section of `config/params_lgn.yaml`.

## The quantity everything downstream wants

Given a patch of LGN tissue of some physical size (mm), how much visual
field (deg) does it cover? That — **degrees of visual angle per
millimetre of tissue** — is what turns a current-spread radius into a
phosphene size, so it is the common currency every model here returns.

`LocalMagnification.linear_mm_per_deg` gives back the reciprocal
(mm/deg) that the magnification literature usually tabulates. It is
defined as `1/sqrt(areal)`, the side of the square patch of tissue
covering one square degree. For an isotropic model that is exactly the
reciprocal of `major_deg_per_mm`; for an anisotropic one it is a
deliberate scalar summary that **discards the anisotropy**, so it must
not be used where the ellipse matters.

## Three models, deliberately not collapsed into one

| Model | Shape | Source |
| --- | --- | --- |
| `JacobianMagnification` (default) | per-voxel, anisotropic | cached `JacobianAtlas` |
| `AtlasGradientMagnification` | isotropic scalar M(E) by eccentricity bin | same atlas gradient, tabulated |
| `MalpeliDensityMagnification` | isotropic, interval-valued | Malpeli et al. (1996) closed-form density |

**None of the three is an independent measurement of the others.** Erwin
et al. (1999) reconstructed the atlas *starting from* Malpeli et al.
(1996)'s data, and for the central 1 deg wrote Malpeli's own isotropic
formula directly into `ECC.DAT`. All three therefore trace back to one
macaque's 415 recording sites and share the same ~211 um (SD 217 um)
accuracy floor. Differences between them are processing noise, not
corroboration — which is why the comparison utilities are framed as
*validation checkpoints* rather than cross-checks between independent
sources.

`AtlasGradientMagnification` sets `isotropic_by_construction`
unconditionally, so choosing the cheap fallback path stays visible
downstream rather than silently producing circular phosphenes that look
like a finding.

## Why the Jacobian is a 2×3 matrix

The Jacobian maps a physical displacement (mm, in ML/DV/AP) to a
visual-field displacement (deg, in a Cartesian `x = E·cos(I)`,
`y = E·sin(I)` frame). Its two non-zero singular values are the
principal magnifications in deg/mm; the corresponding left singular
vector gives the major axis's direction *in the visual field*, which is
the orientation a rendered ellipse needs.

Because LGN layers are thin curved sheets, only two of the three
physical directions carry retinotopic gradient — the third, through the
sheet, is close to a null direction. That is why a 2×3 matrix has
generically only two meaningful singular values, and why no third
magnification is reported.

`at_eccentricity` averages over inclination, which is exactly the step
that discards the anisotropy the Jacobian exists to capture. It is for
validation against inclination-blind literature curves, not for
rendering.

## The Malpeli density formulas

`magnification.malpeli` holds Eqs. 1–2 of Malpeli, Lee & Baker (1996),
verified against the full text:

- parvocellular: `1_011_688 · (E + 2.9144)^−2.6798` cells/deg²
- magnocellular: `2620.2 · ((E − 1.8322)² + 5.5638)^−0.8012` cells/deg²

The magnocellular formula is non-monotonic near the fovea. That is the
formula's own documented behaviour — the quadratic-inside-power-law term
exists precisely because magnocellular density, unlike parvocellular,
does not fall monotonically with eccentricity — not a numerical
artefact.

Both rest on 415 recording sites from 38 microelectrode passes in **one**
macaque (Malpeli & Baker 1975). The Erwin et al. (1999) atlas is not an
independent second source; it was reconstructed from the same data.

## Density → magnification: a conversion that is not free

Malpeli's functions give cell **density** in cells/deg². Getting a
tissue magnification in mm/deg needs two further quantities the formula
does not supply:

1. a volumetric cell density `rho_v` (cells/mm³), turning cells/deg²
   into a *volume* magnification mm³/deg², and
2. a projection-column thickness `T` (mm) — the total depth of
   same-class laminae a single visual-field location's projection column
   passes through — turning that into an *areal* magnification mm²/deg².

Then `M_linear = sqrt(areal)`, and that square root is a third
assumption (isotropy), not a unit conversion.

Neither (1) nor (2) is fixed by the formula, so `MalpeliDensityMagnification`
returns an **interval** by default. Per the project's standing rule, a
derivation chain with more than one arbitrary choice in it reports a
range, not a point estimate. The interval's width comes from the
layer-thickness term alone; `rho_v` shifts all three bounds together and
is not an independent source of width.

`from_atlas` takes `rho_v` from the atlas's own mean cells per voxel
(`CELLS.DAT` counts real cells in real 25 um voxels), which is preferred
over supplying an external packing figure.

### T does not have to be assumed — it is measurable

Superseded 12 Sep 2026. This section previously reported an
order-of-magnitude standoff about `T` and told the reader to treat the
density route as the *magnitude* estimate and the Jacobian route as the
*shape* estimate, without combining them. That advice is withdrawn; what
follows replaces it. Script:
[`check_column_thickness.py`](../dynaphos_lgn/check_column_thickness.py),
report `research/column-thickness-measured-2026-09-12.md`.

The atlas measures its own `T`. Volume magnification comes straight from
voxel counts (each valid voxel is `voxel_size_mm³` of real tissue, and
an eccentricity annulus's solid angle is closed-form), areal
magnification comes straight from the cached Jacobian, and dividing one
by the other leaves millimetres:

    T = (V / Ω) / A,  computed per voxel as
    T_bin = voxel_volume · Σ(σ₁σ₂) / Ω_bin

**What it says.** Summed over a class's laminae — which is what this
parameter denotes — measured `T` is **0.56–1.77 mm for parvo** and
**0.20–0.74 mm for magno**. So:

- the configured central `T = 3 mm` is about **2× too thick for parvo**
  and 5–7× for magno. Since `M_linear ∝ 1/√T`, a 2× error in `T` is a
  1.4× error in magnification — about the size of the parvo
  disagreement previously attributed to an unknown;
- **`T` is class-dependent by a factor of ~2.4**, and
  `column_thickness_mm` is a single triple shared by both classes, so
  one of the two is necessarily wrong whatever it is set to. This is
  the most actionable item, and it is a candidate explanation for the
  otherwise-unexplained observation that magno's gap does not shrink
  with eccentricity the way parvo's does: one shared `T` cannot track
  two different curves. `MalpeliDensityMagnification` now accepts a
  per-class mapping as well as a single triple; the config still ships
  the single triple, with the measured split written out in a comment
  awaiting sign-off;
- the atlas implies laminae **2–5× thicker than Connolly & Van Essen
  (1984) p.553 measured directly** (their 0.125–0.225 mm per fused
  lamina). Now largely reconciled — see the next section.

**What no `T` can fix.** Volume magnification can be compared without
any `T` on either side: measured voxel-volume-per-solid-angle against
`malpeli_density / rho_v`. Those agree to within ~10–25% beyond 7 deg
but disagree by 1.5–2× inside 4 deg. That residual is upstream of every
magnification model and must not be absorbed into a thickness
parameter. Most of it turns out to be the constant-`rho_v` assumption
rather than a real cell shortfall — see "`rho_v` is not a constant
either" below.

**Beware the circularity.** Substituting the measured `T` back into the
density route makes the two routes agree by construction beyond ~7 deg
— the Jacobian cancels out of the comparison algebraically. The
agreement is therefore not corroboration; the *residual* is the finding.

### What `T` actually counts, and the C&VE discrepancy decomposed

Script:
[`check_sheet_thickness_morphology.py`](../dynaphos_lgn/check_sheet_thickness_morphology.py).

The sheets can be measured *geometrically* — a distance transform over
`LAYERS.DAT`, with no retinotopy, no solid angle and no Jacobian
anywhere in the chain. That separates the two candidate causes above,
because a reconstruction that fattens its sheets shows up in a
morphological measurement and a Jacobian that overstates deg/mm cannot.

The 2–5× decomposes into three factors, and the largest is not an error:

| factor | size | what it is |
| --- | --- | --- |
| representation count | ~2.2× | inside 15° each parvo code carries **two** laminae — its principal layer plus its continuous leaflet |
| reconstruction thickening | ~1.7× | the atlas's sheets really are fatter than C&VE's sections, about equally in all four codes |
| residual | ~1.2× | what is actually left for Jacobian inflation to explain |

The leaflet doubling is a falsifiable prediction and it holds: the ratio
of implied to morphological thickness is 2.8 for the parvo codes inside
15°, falls to 1.3 beyond the leaflet boundary, and stays flat at ~1.3
for the magno codes, which have no leaflet to lose. The magno codes are
the control and they behave like one.

So **`column_thickness_mm` is not a lamina thickness.** It is tissue
depth per unit map area *summed over representations*, which makes it
inherently eccentricity-dependent — it should fall beyond 15°, and the
measurement shows it doing so. The class split the previous section
calls for is still needed and still the right size; the reason is that
parvo represents each central point twice, not that parvocellular tissue
is thicker.

The ~1.7× reconstruction thickening is real, is not shrinkage (C&VE
corrected for that, so their figures are already the larger reading) and
is a reason to treat any absolute tissue-depth number from this atlas as
soft.

### `rho_v` is not a constant either

Script:
[`check_cell_density_profile.py`](../dynaphos_lgn/check_cell_density_profile.py).

`from_atlas` takes `rho_v` as a **single** mean over the class's voxels.
Cells per voxel actually falls monotonically with eccentricity, by
**1.5× for parvo and 2.6× for magno** between the centre and the far
periphery. Volume magnification is `rho_v`'s direct reciprocal, so the
constant is wrong by up to 2× for magno in the central field, and by
√2 in linear magnification.

That accounts for the central residual above almost entirely for magno
(0.55 → 1.00 using the atlas's real cell counts) and for about a fifth
of it for parvo (0.67 → 0.74). `from_atlas(..., eccentricity_resolved=True)`
builds a tabulated profile; the config default is unchanged.

Note this **compounds with** the `T` finding rather than replacing it:
linear magnification goes as `sqrt(density / (rho_v · T))`, and both
`rho_v` and `T` are configured as constants while both measure as
eccentricity- and class-dependent. Two separate errors in one
expression.

A genuine ~26% parvocellular shortfall inside 4° survives all of this.
It is not quadrature (the midpoint rule is accurate to 1–5%), not
packing density and not column thickness. `FOVEOLA.DAT` — surfaced
earlier and still unused — is the obvious next place to look.

### The whole-nucleus cell-count check is weaker than it looks

`test_hemifield_integral_reproduces_atlas_cell_counts` compares
integrated totals against Erwin et al.'s published figures and passes.
Resolved by eccentricity the same comparison runs ~25% short in the
central bins and ~15% long in the mid-periphery — compensating errors
that a total cannot see. `TestRealAtlasCellDistribution` now pins the
per-bin behaviour alongside it.

### The quantisation-floor hypothesis: closed for the live code path

This document previously hypothesised that a near-fovea figure of
~0.25 mm/deg was a quantisation floor — one voxel per one `ECC.DAT`
quantum (0.025 mm / 0.1 deg) — as a pairwise finite-difference estimate
would produce. Tested 12 Sep 2026 and **not supported for
`JacobianMagnification`**: there is no spike of voxels at 0.25 (spike
ratio never above 1.4 against symmetric flanking windows), up to 84% of
voxels sit *below* it, and the live near-fovea median is ~0.64 mm/deg,
not 0.25. `JacobianAtlas.compute` least-squares-fits over a
neighbourhood of ~26 valid neighbours rather than differencing a pair,
so it averages quantisation away instead of inheriting a floor.

The hypothesis may still have been right about the **retired**
atlas-gradient sampling script the 0.25 figure actually came from, which
did use pairwise differences. Either way, the 2.5–4× "disagreement"
between the density route and the atlas-gradient route was a property of
that retired script and should no longer be carried as an open
discrepancy about the current model.

`calibrate_column_thickness_mm` inverts the map length to back-solve `T`.
`M_linear` scales as `1/sqrt(T)`, so the map length does too and the
inversion is closed-form: `T = T_ref · (L_ref / L_target)²`. The
plausibility bracket is applied afterwards purely as a sanity check, and
an out-of-bracket result is logged and returned unclamped — an
implausible value is itself the finding.

## Validation checkpoints

These run in preprocessing only and never in the forward pass.

**Vurro, Crowell & Pezaris (2014)**, `sigma = 0.043·rho + 0.083` deg. A
human-acuity-derived phosphene-size curve, cross-checked by its authors
against a single unpublished LGN monkey data point (~0.5 deg at 10 deg
eccentricity). Structurally it is the isotropic special case of this
module's scalar fallback, reached via a different proxy — so use it as
an order-of-magnitude cross-reference at specific eccentricities, never
as a source formula or a simulator input.

**Connolly & Van Essen anisotropy** (1984, p.552, full text verified):
in **layer 6** near the peripheral horizontal meridian the magnification
is 2–3× greater along isoeccentricity contours than along isopolar ones,
and this reverses in the more ventral layers. Note the sign, which is
easy to invert: a larger magnification means more *tissue per degree*,
i.e. fewer deg/mm, so the ratio exceeds 1 exactly where the Jacobian's
**major** (deg/mm) axis points **radially**.

Tested properly on 12 Sep 2026 by
[`check_anisotropy_local_frame.py`](../dynaphos_lgn/check_anisotropy_local_frame.py),
which computes C&VE's own quantity directly from `J Jᵀ` rather than
inferring it from principal axes, per layer and in the local
isoeccentricity/isopolar frame. Results (report
`research/anisotropy-local-frame-2026-09-12.md`):

| piece of the claim | verdict |
| --- | --- |
| a reversal exists between innermost and outermost layers | **reproduced** — layer 1 sits at contour alignment −0.72, layer 6 at +0.17 |
| it is graded across layers rather than a two-level split | **not clearly** — almost all of it is the layer-1→2 step; layers 2, 3 and 6 span only 0.15 between them |
| layer 3 patterns against its own cell class | **not reproduced** — layer 3 is the *most* isopolar-aligned of the four |
| the magnitude is 2–3× in layer 6 | **not reproduced** — 1.08× |

The magnitude failure is not a ceiling artefact: layer 6's median
anisotropy in that population is 1.85, so ~2× was arithmetically
available. Nor is it an averaging-window artefact — narrowing the
meridian band from 30° to 2.5° moves the ratio only 1.05 → 1.15. What is
left is that **this atlas's reconstruction does not carry the contour
alignment C&VE measured in sections**, which is a statement about the
atlas rather than about the paper, and it bounds what any
Jacobian-derived anisotropy from this atlas can be expected to
reproduce.

An earlier check (11 Sep) measured orientation against a *fixed*
horizontal axis and concluded parvo and magno pointed the same way. That
is superseded, not contradicted: a local claim was being compared
against a global ruler. Its magnitude results (median anisotropy
1.75 parvo, 1.94 magno) stand.

Compare against `LocalMagnification.anisotropy` for magnitude, but use
the local-frame script for anything directional.

**`compare_against_malpeli`** deliberately returns no pass/fail.
Agreement is partly by construction and disagreement is informative
about processing, not about biology.

## Flags on `LocalMagnification`

- `isotropic_by_construction` — the underlying retinotopy is isotropic
  because Erwin et al. wrote Malpeli's isotropic formula in, not because
  anisotropy was measured and found absent.
- `unreliable_beyond_neighborhood` — extrapolating this single local
  linear map across a whole current-spread footprint is not trustworthy;
  the atlas's own values a few hundred microns away differ from what
  this Jacobian predicts by more than the flag's threshold. It cannot
  separate a genuine anatomical tear from ordinary fast curvature of the
  retinotopic map, and does not try to. Consumers route flagged
  electrodes to per-voxel scatter rendering instead of the
  single-Jacobian ellipse shortcut.
