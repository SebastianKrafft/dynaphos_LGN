"""Check: what does turning `thresholding.soft_threshold` on actually do?

Item "detection threshold: soft (logistic), not hard" from
`open-design-choices-summary-2026-09-11.md` was signed off on 11 Sep 2026
and the config default flipped on 12 Sep. This script is the measurement
behind that flip, and behind the `soft_threshold_baseline` parameter it
turned out to need.

It answers three separate questions, because they have different answers:

1. **Does the soft gate actually buy gradient?** Yes, in the regime that
   matters -- above the rheobase, below the detection threshold -- by
   roughly an order of magnitude.
2. **What does it cost?** With the raw logistic, a *zero-amplitude* frame
   stops being black. The threshold sits only `mu/sd = 1.36` slope-widths
   above zero activation, so a silent electrode gets `p(detect) ~ 0.2`.
3. **Does fixing that cost the gradient back?** No. The guess-rate
   correction `(S(a) - S(0)) / (1 - S(0))` is an affine rescaling of the
   same logistic, so `dp/da` stays strictly positive.

It also makes visible a *second*, independent gradient wall that the soft
threshold neither causes nor fixes: `get_current` applies a hard
`relu(amplitude - trace - rheobase)`, so below 23.9 uA the activation --
and its gradient -- is identically zero however the detection gate is
configured.

Run from the repo root::

    python -m dynaphos_lgn.check_soft_threshold

Writes a markdown report to `research/` and prints it.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import sys
from pathlib import Path

import numpy as np

# Amplitudes, microamperes. Chosen to straddle both nonlinearities: the
# rheobase (23.9 uA) and the detection threshold.
AMPLITUDES_UA = (0.0, 10.0, 20.0, 23.9, 30.0, 40.0, 60.0, 100.0, 200.0)

GATES = (('hard', False, 'none'),
         ('soft, baseline=none', True, 'none'),
         ('soft, baseline=zero_activation', True, 'zero_activation'))


def analytic_table(params) -> list:
    """The gate curves on their own, with no simulator around them.

    Independent of the atlas, so a reader can check the pedestal claim
    without the 145 MB download.
    """
    th = params['thresholding']
    br = params['brightness_saturation']
    mu = float(th['activation_threshold'])
    sd = float(th['activation_threshold_sd'])
    slope = th.get('soft_threshold_slope') or 1.0 / sd

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    floor = sigmoid(slope * -mu)
    lines = [
        f"- `activation_threshold` mu = {mu:.6g}",
        f"- `activation_threshold_sd` sd = {sd:.6g}",
        f"- default soft slope 1/sd = {slope:.6g}",
        f"- **mu/sd = {mu / sd:.4f}** -- how many slope-widths zero "
        f"activation sits below threshold. This one ratio is the whole "
        f"problem: the logistic is nowhere near saturated at a = 0.",
        f"- raw logistic at zero activation: p(detect) = **{floor:.4f}**",
        f"- brightness curve at zero activation: **"
        f"{sigmoid(br['slope_brightness'] * (0 - br['cps_half'])):.4f}**",
        "",
        "| activation / mu | hard | soft (raw) | soft (baselined) "
        "| brightness |",
        "|---|---|---|---|---|",
    ]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        a = frac * mu
        soft = sigmoid(slope * (a - mu))
        lines.append(
            f"| {frac:.2f} | {float(a > mu):.3f} | {soft:.3f} "
            f"| {(soft - floor) / (1 - floor):.3f} "
            f"| {sigmoid(br['slope_brightness'] * (a - br['cps_half'])):.3f} "
            f"|")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/params_lgn.yaml')
    ap.add_argument('--atlas-dir', default=None)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--n-electrodes', type=int, default=64)
    ap.add_argument('--resolution', type=int, default=128)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch

    from dynaphos_lgn.build import (build_atlas, build_electrode_array,
                                    build_jacobian, build_kernel,
                                    build_magnification_model,
                                    load_lgn_params)
    from dynaphos_lgn.simulator import LGNPhospheneSimulator

    base = load_lgn_params(args.config)
    base['run']['resolution'] = [args.resolution, args.resolution]
    base['run']['gpu'] = None
    base['electrodes']['n_electrodes'] = args.n_electrodes

    # fallback_to_synthetic=False: the pedestal's *size* depends on how
    # much tissue each electrode recruits, so a synthetic-atlas number
    # would be a different claim wearing the same units.
    atlas = build_atlas(base, atlas_dir=args.atlas_dir,
                        fallback_to_synthetic=False)
    cache_dir = args.cache_dir or args.atlas_dir or str(
        Path(base.get('atlas', {}).get('directory', '.')))
    jacobian = build_jacobian(base, atlas, cache_dir=cache_dir)

    def make(soft, baseline):
        params = copy.deepcopy(base)
        params['thresholding']['soft_threshold'] = soft
        params['thresholding']['soft_threshold_baseline'] = baseline
        rng = np.random.default_rng(0)
        array = build_electrode_array(params, atlas, jacobian,
                                      build_kernel(params), rng)
        array.magnification_model = build_magnification_model(
            params, atlas, jacobian)
        # A fresh generator, so every gate samples the SAME per-electrode
        # thresholds and the comparison is about the gate alone.
        return LGNPhospheneSimulator(params, array,
                                     rng=np.random.default_rng(0))

    sims = {label: make(soft, baseline) for label, soft, baseline in GATES}

    today = dt.date.today().isoformat()
    md = [f"# Soft detection threshold: what flipping it on does "
          f"-- {today}",
          "",
          f"Real Erwin atlas, {args.n_electrodes} electrodes, "
          f"{args.resolution}x{args.resolution} render, seed 0, identical "
          f"per-electrode thresholds across gates. Reproduce with "
          f"`python -m dynaphos_lgn.check_soft_threshold`.",
          "",
          "## The gate curves alone (no atlas needed)",
          ""]
    md += analytic_table(base)

    md += ["",
           "## Rendered frame vs. amplitude",
           "",
           "Peak pixel of the rendered percept, and the fraction of "
           "pixels above 0.01, with every electrode driven at the same "
           "amplitude. Both are quoted in `config/params_lgn.yaml` and "
           "`docs/simulator.md`, so they are produced here rather than "
           "by an ad-hoc script -- a figure in the docs that this "
           "script does not reproduce is exactly the drift worth not "
           "having.",
           "",
           "| amplitude (uA) | "
           + " | ".join(f"{g[0]} — peak / % > 0.01" for g in GATES) + " |",
           "|---|" + "---|" * len(GATES)]
    zero_frame = {}
    for ua in AMPLITUDES_UA:
        row = []
        for label, _, _ in GATES:
            sim = sims[label]
            sim.reset()
            img = sim(torch.full((sim.num_phosphenes,),
                                 ua * 1e-6)).detach()
            peak = float(img.max())
            above = float((img > 0.01).to(img.dtype).mean())
            if ua == 0.0:
                zero_frame[label] = (peak, above)
            row.append(f"{peak:.5f} / {above:.2%}")
        md.append(f"| {ua:g} | " + " | ".join(row) + " |")

    md += ["",
           "## Gradient of the rendered percept w.r.t. amplitude",
           "",
           "`sum(percept).backward()`, mean |d/d amplitude| over "
           "electrodes. Larger is better for an end-to-end optimiser.",
           "",
           "| amplitude (uA) | " + " | ".join(g[0] for g in GATES) + " |",
           "|---|" + "---|" * len(GATES)]
    grads = {label: {} for label, _, _ in GATES}
    for ua in AMPLITUDES_UA:
        row = []
        for label, _, _ in GATES:
            sim = sims[label]
            sim.reset()
            amp = torch.full((sim.num_phosphenes,), ua * 1e-6,
                             requires_grad=True)
            sim(amp).sum().backward()
            g = float(amp.grad.abs().mean())
            grads[label][ua] = g
            row.append(f"{g:.4g}")
        md.append(f"| {ua:g} | " + " | ".join(row) + " |")

    rheobase_ua = float(base['thresholding']['rheobase']) * 1e6
    supra = [ua for ua in AMPLITUDES_UA if ua > rheobase_ua]
    ratios = []
    for ua in supra:
        hard = grads['hard'][ua]
        soft = grads['soft, baseline=zero_activation'][ua]
        if hard > 0:
            ratios.append((ua, soft / hard))

    # Peak-pixel dimming at a near-threshold amplitude: the soft gate's
    # other consequence, and the one most likely to surprise.
    dim_ua = 40.0
    peaks = {}
    for label, _, _ in GATES:
        sim = sims[label]
        sim.reset()
        peaks[label] = float(sim(torch.full((sim.num_phosphenes,),
                                            dim_ua * 1e-6)).detach().max())
    hard_peak = peaks['hard']
    soft_peak = peaks['soft, baseline=zero_activation']
    dim_factor = hard_peak / soft_peak if soft_peak > 0 else float('inf')

    md += ["",
           "## The consequence most likely to surprise: near-threshold "
           "percepts get dimmer",
           "",
           f"At {dim_ua:g} uA the peak rendered pixel falls from "
           f"**{hard_peak:.4f}** (hard) to **{soft_peak:.4f}** "
           f"(soft, baselined) -- a factor of {dim_factor:.1f}. That is "
           f"not a bug: the hard gate multiplies brightness by exactly "
           f"1 the instant threshold is crossed, while the psychometric "
           f"curve says a just-threshold electrode is detected about "
           f"half the time. The soft gate is the more defensible model "
           f"of a detection experiment, but it is a different brightness "
           f"scale, so:",
           "",
           "- any perceptual loss whose magnitude was tuned against "
           "hard-gate renders needs retuning;",
           "- `brightness_saturation` is inherited from Dynaphos's V1 "
           "fit and was never re-anchored for the LGN "
           "(`docs/calibration.md` lists the only two quantitative "
           "anchors that exist, and neither is a brightness one), so "
           "there is no measurement that adjudicates which scale is "
           "right;",
           "- the two gates converge at high amplitude (see the 200 uA "
           "row), so this is specifically a near-threshold effect.",
           "",
           "## What the three questions come out as",
           "",
           f"**1. Does the soft gate buy gradient?** Yes, above the "
           f"rheobase ({rheobase_ua:.1f} uA): "
           + ", ".join(f"{r:.0f}x at {ua:g} uA" for ua, r in ratios)
           + ". That is the regime an optimiser has to climb out of, and "
             "it is the case for the flip.",
           "",
           f"**2. What does it cost?** With the raw logistic, a "
           f"zero-amplitude frame stops being black: peak pixel "
           f"{zero_frame['soft, baseline=none'][0]:.3f} with "
           f"{zero_frame['soft, baseline=none'][1]:.1%} of pixels above "
           f"0.01, against exactly 0 for both the hard gate and the "
           f"baselined soft gate. The repo's own "
           "`test_zero_current_gives_an_empty_percept` had encoded that "
           "invariant since before the soft threshold existed, and fails "
           "under `baseline: none`.",
           "",
           "**3. Does the correction cost the gradient back?** No. "
           "`(S(a) - S(0)) / (1 - S(0))` is an affine rescaling of the "
           "same logistic by `1/(1 - S(0)) > 1`, so above the rheobase "
           "the corrected gradient is if anything larger.",
           "",
           "## The other gradient wall, which this does not fix",
           "",
           f"`get_current` applies `relu(amplitude - trace - rheobase)`. "
           f"Below {rheobase_ua:.1f} uA the activation is identically "
           f"zero and so is its gradient -- in the `hard` and "
           f"`baseline=zero_activation` columns alike. The `raw` column's "
           f"nonzero sub-rheobase gradient is **not** a counterexample: it "
           f"flows entirely through the rendered *size* of an electrode "
           f"that produces no percept, via the pedestal. An optimiser "
           f"would exploit it.",
           "",
           "If gradient below the rheobase is actually wanted, the thing "
           "to soften is that `relu` -- as its own explicit decision, not "
           "as a side effect of the detection gate's baseline. Not "
           "currently on the open-choices list; surfaced 12 Sep 2026.",
           ""]

    text = '\n'.join(md)
    print(text)
    out = Path(args.out or f'research/soft-threshold-baseline-{today}.md')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"\nWritten to {out}")


if __name__ == '__main__':
    main()
