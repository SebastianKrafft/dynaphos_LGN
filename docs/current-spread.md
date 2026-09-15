# Current spread and recruitment

Code: [`dynaphos_lgn/current_spread.py`](../dynaphos_lgn/current_spread.py).
Config: the `current_spread:` and `benchmarks:` sections of
`config/params_lgn.yaml`.

## Stoney's law, and the radius/diameter trap

The spread model is Stoney/Tehovnik:

    D = 2 * sqrt(I / K)

`D` is the **diameter** of activated tissue and `K` an excitability
constant in uA/mm^2. The radius the geometry actually wants is therefore
`R = sqrt(I / K)` — half of `D`.

Conflating the two is a factor-of-2 error in every rendered size and a
factor-of-4 error in any `K` back-derived from published data. It is one
of the reasons this project's earlier attempt to derive `K` from
behavioural scatter was abandoned. `stoney_radius_mm` and
`stoney_diameter_mm` are kept as separate, unambiguously named functions
for that reason.

## Where K comes from

`current_spread.k_ua_per_mm2` is 675 uA/mm^2, from Tehovnik et al.
(2006), measured in **macaque V1**. It is unvalidated for LGN: no LGN-
or thalamus-specific current-spread constant exists in the literature at
any confidence level. Everything the simulator reports about phosphene
size inherits that.

`bosking_diameter_mm` offers a saturating sigmoid as an alternative
functional *shape*. Bosking et al. (2017) measured human ECoG **surface**
stimulation at milliampere currents, which does not transfer to
microelectrode microamperes, so the configured slope and half-current are
Dynaphos's own V1 values rather than Bosking's.

## Why the kernel is smooth

`RecruitmentKernel` weights each nearby voxel by a smooth function of
distance and current, never a hard cutoff. This is a hard requirement,
not a preference: a hard in/out test against `R(I)` has zero gradient
with respect to `I` almost everywhere, so an encoder optimised
end-to-end could never learn to move a subthreshold electrode across
threshold.

The kernel is anchored so its weight equals `contour_value` at exactly
`R(I)`. That anchoring is what keeps "the radius the size model uses"
and "the radius per-layer recruitment uses" the same number instead of
two independently tuned ones.

`contour_value` defaults to 1/e², which makes the derived
`radius_to_sigma` equal V1 Dynaphos's shipped constant of 0.5. The other
common convention is 0.5 (half-maximum), which yields a different
`radius_to_sigma`.

Because a smooth kernel has non-zero weight past its nominal radius,
setup code sizing a candidate neighbourhood must use
`effective_radius_mm`, not `radius_mm`, or it clips the tail.

## Cell-class scaling of K

Larger cells are more excitable, so a class-scaled model would give
magnocellular tissue a smaller `K` — wider spread per unit current —
than parvocellular. **No measurement supports any particular exponent.**
Three options are therefore offered explicitly rather than one being
buried as a default:

| `current_spread.class_scaling` | Meaning |
| --- | --- |
| `none` (default) | every class shares the reference `K` |
| `diameter` | `K ~ 1/d`; threshold current falls linearly with soma size |
| `diameter_squared` | `K ~ 1/d²`; threshold current falls with membrane area |

The nucleolus diameters the scaling uses (parvo 2.25 um, magno 3.00 um)
are an adopted project design parameter, and the mapping from diameter
to a per-class `K` is a modelling choice, not a measurement.

## Multi-electrode interaction

**There is no LGN multi-electrode threshold data.** The three
`aggregate_activation` modes bracket a real disagreement in the non-LGN
literature rather than approximating one known answer, so results should
be reported across modes, not from one:

- `independent` (default) — plain linear summation, inherited from
  Dynaphos. Each electrode recruits its own tissue and the totals add.
- `field_superposition` — sum current-density contributions before the
  recruitment non-linearity, i.e. treat the extracellular field as
  additive in a quasi-static volume conductor, per Mazurek et al. (2022)
  and Kim et al. (2017). Because the kernel is already a monotone
  function of current, superposing the *weights* is a first-order stand-in
  for superposing the fields: the two agree in the linear regime and
  differ where the kernel saturates.
- `probability_summation` — combine electrodes as independent noisy
  detectors, `1 - prod(1 - w)`, reproducing Callier et al. (2015)'s
  near-zero, spacing-independent threshold reduction.

`threshold_reduction_bracket` reports the two mechanistic extremes plus
the geometric quantity that decides which should dominate — electrode
spacing relative to the current-spread radius:

| Source | Tissue | Reduction at N≈4 | Spacing dependence |
| --- | --- | --- | --- |
| Callier et al. (2015) | macaque S1 | <10% | none |
| Kunigk et al. (2022) | rat S1 | up to ~66–76% | vanishes beyond ~600 um |
| Fernandez et al. (2021) | human V1 | 28% at ~400 um pitch | nearest empirical anchor |

At 40 uA under the transplanted `K` the spread radius is ~244 um, so any
array with more than a handful of contacts inside the (small) macaque
LGN is close to forced into the tight-spacing, field-summation regime.
That is a geometric plausibility argument, not a measurement.

## Detection threshold

`benchmarks.pezaris_2007.threshold_ua` is 40 ± 12 uA (n = 6), from the
single load-bearing macaque LGN microstimulation paper. Note that the
40 uA titration sub-study (n=6) and the main-task scatter figures (n=14)
are non-overlapping experiments — nothing confirms the main task ran at
~40 uA.
