# Receptive fields

Code: [`dynaphos_lgn/receptive_fields.py`](../dynaphos_lgn/receptive_fields.py).
Config: the `receptive_fields:` section of `config/params_lgn.yaml`.

## What these are used for

LGN receptive fields are circularly symmetric centre-surround
differences-of-Gaussians — not orientation-tuned like V1's. That is the
main structural reason the V1 simulator's phosphene primitive (an
isotropic Gaussian blob) needs *less* adaptation for LGN than one might
expect: Dynaphos's blob was justified by human verbal reports of round
flashes, not by orientation columns, so it transfers cleanly.

This module supplies the *size* of those fields as a function of
eccentricity and cell class. The simulator uses it for stimulus sampling
— which patch of image an electrode "sees" — not for phosphene
appearance.

## Verification status, which is unusually uneven

**M and P (macaque).** Croner & Kaplan (1995) is the right species and
the right structure, but publishes no closed-form `rc(E)`. The
configured coefficients are an in-project fit to digitised Fig. 4 points
(`dynaphos_lgn/research/croner_kaplan_fig4_extracted.csv`), refittable
with `fit_croner_kaplan`. As a cross-check, the fit reproduces Croner &
Kaplan's own qualitative "M ≈ 2× P" statement (the fit gives 2–3×
depending on eccentricity).

**K (koniocellular).** No macaque K receptive-field dataset exists in
any form. The defaults are owl monkey (Xu, Bonds & Casagrande 2002):
`rc = 0.063·E − 0.31` (r² = 0.59) and `rs = 0.067·E + 0.28` (not
significant). Owl monkey absolute sizes run about 2× macaque and 0.5×
bush baby at matched eccentricity. They are here for the robust
cross-species *ordering* K > M > P, not as transplantable absolute
sizes. The marmoset-vs-macaque P comparison at 35 deg is off by more
than an order of magnitude — a concrete demonstration that these
equations do not substitute for one another.

`class_ordering_check` reports all three classes at one eccentricity.
Because the K numbers are owl monkey while M and P are macaque, the
magnitude of any gap it shows mixes a real class difference with a
species difference.

## Conventions that are easy to get wrong

`rc` is Croner & Kaplan's own convention: the radius at which
sensitivity falls to 1/e of its peak. It is **not** a standard deviation
and not a half-width at half-maximum. They parameterise the DoG as
`Kc·exp(−(r/rc)²)`, so `rc = sqrt(2)·sigma`. Collapsing the two readings
is a factor of 1.41 in every derived size, which is why
`center_sigma_deg` exists as a separate property rather than the two
being used interchangeably.

`surround_center_volume_ratio` is the **integrated** surround/centre
sensitivity ratio, `(Ks·rs²)/(Kc·rc²)`. It is stored in that form
because that is the form the literature reports; the peak ratio `Ks/Kc`
that the DoG equation actually needs is derived from it in
`surround_center_peak_ratio`.

The configured value is 0.55 (macaque M+P pooled, Croner & Kaplan 1995).
This is not a universal constant: bush baby K is ~0.87 and owl monkey K
is ~0.77, so any single value is a choice among these.

In `ReceptiveField.profile` the centre mechanism is normalised to peak
at 1, so the net DoG peaks at `1 − Ks/Kc` (slightly below 1) and crosses
zero near the centre/surround boundary. That is the shape, not an error.

## Flagged assumptions

**The surround radius for M and P is `rs/rc` × `rc(E)`.** Croner &
Kaplan report a broad `rs/rc` across their sample; the configured value
is their commonly quoted median, used as a scalar multiplier because no
separate `rs(E)` fit was digitised in-project. This forces `rs` to have
exactly the same eccentricity dependence as `rc`, which the data does
not establish. The koniocellular class ignores it and uses Xu et al.'s
own separate `rs(E)` fit.

**The koniocellular linear fit goes negative below ~4.9 deg.** The same
paper reports a measured mean of 0.31 deg below 15 deg, so `rc` is
clamped to that floor rather than to zero.

**Refitting has a bias/variance trade-off.** `fit_croner_kaplan`'s
`confidence` filter exists because the digitisation marks points
recovered from overlapping clusters as `approx`. Excluding them shrinks
the P sample from 80 points to 16, so filtering is not a free
improvement.
