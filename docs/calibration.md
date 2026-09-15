# Calibration

Code: [`dynaphos_lgn/calibration.py`](../dynaphos_lgn/calibration.py).

## The only two quantitative anchors that exist

Neither is a measurement of the thing being modelled.

**Pezaris & Reid (2007)** — a 40 ± 12 uA detection threshold, plus
2.3 ± 1.2 deg percept accuracy and 0.8 ± 0.6 deg repeatability.
Behavioural, single-electrode, one study. Note that the 40 uA titration
sub-study (n=6) and the main-task scatter figures (n=14) are
non-overlapping experiments.

**Vurro, Crowell & Pezaris (2014)** — `sigma = 0.043·rho + 0.083` deg,
derived from **human** visual acuity as a magnification proxy (anchored
via Cowey & Rolls 1974 and Anderson et al. 1991) and cross-checked by
its own authors against a single unpublished LGN monkey point (~0.5 deg
at 10 deg eccentricity).

## The implied-K diagnostic

`implied_k` back-solves what excitability constant would reconcile the
simulator's rendered sizes with the Vurro curve. The rendered sigma is
proportional to the recruitment sigma in tissue, which under Stoney
scales as `1/sqrt(K)`, so matching a target sigma means

    K_implied = K_current · (sigma_model / sigma_target)²

evaluated per electrode.

**Read this as a diagnostic, not as a calibrated K.** In order of
severity:

1. The target curve is human psychophysics, not macaque LGN tissue, and
   its only LGN contact is one unpublished data point.
2. Vurro et al. model no electrode, no current and no current spread at
   all — phosphene size is an experimenter-chosen input there, not the
   output of a forward model. So this is not comparing two estimates of
   the same quantity.
3. The comparison depends on the current chosen, on the magnification
   model, and on the contour convention linking the Stoney radius to a
   Gaussian sigma — three choices, none of them measurements.

The two curves have different **shapes** as well as different
magnitudes: Vurro's is linear in eccentricity because it is an acuity
proxy, while the simulator's grows with the local magnification. No
single `K` makes them agree everywhere, which is why the spread of
per-eccentricity answers is reported alongside the median rather than
averaged away — the spread *is* the useful output.

`size_comparison` compares the geometric mean of the ellipse's two axes,
i.e. the sigma of the circle with the same area, which is the
like-for-like comparison against an isotropic reference curve.

### A coincidence worth noticing but not leaning on

The project's earlier attempt to back-derive `K` from Pezaris & Reid's
behavioural scatter produced ~6,000 uA/mm² before being rejected on four
independent grounds — radius/diameter conflation, eccentricity-anchor
sensitivity, mismatched experiment subsets, and a scatter signal
statistically indistinguishable from oculomotor noise. Properly
accounted for, that back-derivation spans roughly 200 to 60,000 uA/mm²,
two to two and a half orders of magnitude.

If `implied_k` lands in the same region, that is two flawed methods
sharing a bias as easily as it is two methods converging on the truth —
both rest on percept-level data with no tissue-level measurement
anywhere in the chain.

`kernel_with_implied_k` exists so the alternative can be *rendered and
looked at* rather than argued about, not because the implied value
should be adopted. It copies the array's own kernel and swaps only `K`,
so the comparison is about `K` alone.
