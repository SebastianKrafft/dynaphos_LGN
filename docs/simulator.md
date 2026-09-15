# The forward pass

Code: [`dynaphos_lgn/simulator.py`](../dynaphos_lgn/simulator.py).
Config: the `run:`, `sampling:`, `thresholding:` and `default_stim:`
sections of `config/params_lgn.yaml`.

## Same shape as the V1 simulator

The forward pass deliberately mirrors Dynaphos's V1 simulator —
stimulation parameters in, a rendered percept out, with a
leaky-integrator tissue state carried between frames — so an encoder
trained against one can be pointed at the other. The temporal-dynamics,
thresholding and brightness-saturation machinery is imported from
`dynaphos.simulator` unchanged rather than reimplemented.

What is new here is everything spatial:

- recruitment is computed over **real atlas voxels** around each
  electrode, not from an analytic cortical map;
- those voxels are grouped by **layer**, so one electrode produces one
  activation total per lamina it touches rather than a single scalar;
- phosphene geometry comes from the **local Jacobian** of the atlas's
  retinotopy, so phosphenes can be elliptical where the map is
  anisotropic;
- electrodes whose Jacobian should not be extrapolated across their own
  footprint are rendered from the recruited **point cloud** instead.

## Gradient flow

The loss reaches amplitude, pulse width and frequency through the
rendered image, the recruitment weights and the activation state. It
does **not** need to flow through voxel positions, layer labels or
cached Jacobians — none of those depend on the stimulation parameters,
so they stay fixed buffers.

That is also why the recruitment kernel must be smooth: a hard in/out
distance test would have zero gradient with respect to current
everywhere, and the optimiser would never move. The same rule is why
`rendering._smooth_clamp` exists instead of `torch.clamp` on the splat
bandwidth.

## The soft threshold (on by default since 12 Sep 2026)

Dynaphos gates phosphenes with a hard comparison. That is correct as a
model of "the percept is either there or it is not", but it has zero
gradient with respect to amplitude on both sides of the boundary, so a
subthreshold electrode can never learn to cross it.

The soft version replaces the step with a logistic psychometric curve,
which is what a detection experiment actually measures anyway. It
**changes the model, not just the optimiser**, so it is reported as a
flag — but it is now the default, because this simulator exists to be
optimised end-to-end and the hard gate makes that impossible for exactly
the electrodes an optimiser most needs to move. Measured on the real
atlas (64 electrodes, seed 0, `check_soft_threshold.py`) at 30–40 µA —
above the rheobase, below the detection threshold, which is the window
an optimiser has to climb out of — the soft gate's gradient with respect
to amplitude is **5–8× larger** than the hard gate's. The two converge
at high amplitude, as they should.

Its slope defaults to 1/SD of the threshold distribution, so the curve's
width matches the spread already in the model rather than introducing a
new free parameter.

### Why the curve is baseline-corrected

The raw logistic does not return zero at zero activation, and not by a
small margin. The threshold sits only

    activation_threshold / activation_threshold_sd = 1.36

slope-widths above zero, so at the default slope an **unstimulated**
electrode gets `p(detect) = 0.204`. Multiplied by the brightness curve's
own 0.117 floor at zero activation, a 64-electrode array on the real
atlas renders a zero-amplitude frame with a peak pixel of 0.041 and 2.0%
of pixels above 0.01 — both reproducible with
`python -m dynaphos_lgn.check_soft_threshold`. Under the hard gate that
frame is exactly black — and the suite's own
`test_zero_current_gives_an_empty_percept` had silently encoded that
invariant since before the soft threshold existed.

`thresholding.soft_threshold_baseline: zero_activation` (the default)
subtracts the zero-activation value and renormalises,

    p(a) = (S(a) − S(0)) / (1 − S(0))

so `p(0) = 0` exactly and `p(∞) = 1`. This is the standard guess-rate
correction from psychophysics, not an ad-hoc clamp, and it leaves
`dp/da` strictly positive everywhere — so it costs nothing that motivated
turning the soft threshold on. Set it to `none` only to reproduce a
result from before 12 Sep 2026.

### Near-threshold percepts come out dimmer

The gate change is not brightness-neutral. At 40 µA the peak rendered
pixel falls from 0.171 (hard) to 0.028 (soft, baselined) — a factor of
6. That is the model doing what it now says: the hard gate multiplies
brightness by exactly 1 the instant threshold is crossed, whereas a
psychometric curve says a just-threshold electrode is detected about
half the time.

It is still worth knowing about, because `brightness_saturation` is
inherited from Dynaphos's V1 fit and was never re-anchored for the LGN —
[`calibration.md`](calibration.md) lists the only two quantitative
anchors that exist and neither is a brightness one, so nothing in the
evidence base adjudicates which scale is right. Any perceptual loss
whose magnitude was tuned against hard-gate renders needs retuning.

### The rheobase ReLU is a second, separate gradient wall

Worth knowing before reading too much into a gradient measurement:
`get_current` computes

    charge_per_s = relu((amplitude − trace − rheobase) · pw · freq)

so **below the rheobase (23.9 µA) the activation is identically zero and
its gradient with respect to amplitude is zero too**, whatever the
detection gate does. The soft threshold does not and cannot fix that
regime; it is a different hard nonlinearity in the same forward pass.

The raw (unbaselined) logistic *appears* to pass gradient below rheobase
— but that gradient flows entirely through the rendered **size** of an
electrode that produces no percept, via the pedestal above. It is an
artefact of the glow, not a usable learning signal, and an optimiser
would exploit it. If gradient below rheobase is actually wanted, the
thing to soften is that `relu`, as its own explicit decision.

## Out-of-view electrodes are kept, not dropped

The V1 simulator silently drops out-of-view phosphenes, which quietly
changes how many electrodes a result is actually about. Here they are
kept — an electrode that recruits tissue is still stimulating it, and
its per-layer recruitment is still meaningful — but the mismatch is
logged. A rendered percept that omits half the array without saying so
is the kind of thing that gets noticed only after it has been plotted in
a report.

## Units

Dynaphos's own parameters are in **amperes**, because the
temporal-dynamics parameters were fitted in those units. The
current-spread literature works in **microamperes**. The conversion
happens once, at the simulator boundary, via
`run.current_unit_to_ua`.

## Strength–duration: transplanted, not measured

`get_current` reproduces Dynaphos's leak current (memory trace +
rheobase) that produces habituation. Both the rheobase (23.9 uA) and the
pulse-width dependence were fitted to **human intracortical V1**
(Fernandez et al. 2021, Utah array). No LGN- or thalamus-specific
strength-duration data exists anywhere, so the *shape* of this term is
inherited, not measured here.

Chronaxie is configured as a **range** (axonal, 30–200 us, per Ranck
1975) rather than a point estimate, because nothing in the evidence base
picks a value for LGN. Two things to keep in mind:

- Tehovnik & Slocum (2009) measured 0.11–0.24 ms (mean 0.19 ms) for
  macaque V1 ICMS detection — the same V1/Tehovnik lineage as `K`, if a
  matched anchor is wanted.
- Somatic chronaxie is ~1–10 ms, one to two orders of magnitude longer.
  Short ICMS pulses therefore bias towards **axonal** activation,
  including axons of passage (optic tract, geniculocortical and
  corticogeniculate fibres) rather than relay-cell somata. If direct
  relay-cell activation is the intended model, the somatic range is the
  one to use.

## Stimulus sampling

In `receptive_fields` mode the sampling region's radius is the LGN
receptive-field centre radius at that electrode's eccentricity, for the
configured cell class — not a fixed number of millimetres of tissue as
in the V1 model, since LGN receptive fields are specified directly in
degrees. See [`receptive-fields.md`](receptive-fields.md).
