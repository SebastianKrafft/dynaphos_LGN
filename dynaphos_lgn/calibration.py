"""Calibration checks against the few LGN benchmarks that exist.

Compares rendered phosphene sizes against the Vurro et al. (2014) curve
and back-solves what excitability constant K would reconcile them. That
back-solved value is a DIAGNOSTIC, not a replacement for K: the two
curves differ in shape as well as magnitude, so no single K makes them
agree everywhere and the spread of per-eccentricity answers is itself
the output. See `docs/calibration.md`.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np

from dynaphos_lgn.current_spread import (RecruitmentKernel,
                                         excitability_constant_ua_per_mm2)
from dynaphos_lgn.magnification import vurro_phosphene_sigma_deg


def size_comparison(array, current_ua: float,
                    eccentricities_deg: Optional[Sequence[float]] = None
                    ) -> dict:
    """Rendered phosphene sigma versus the Vurro cross-reference.

    :param array: A built `LGNElectrodeArray`.
    :param current_ua: Stimulation current to evaluate at.
    :return: dict of parallel arrays, one row per electrode.
    """
    major, minor = array.phosphene_sigma_deg(current_ua)
    ecc = (np.asarray(eccentricities_deg, dtype=float)
           if eccentricities_deg is not None else array.eccentricity_deg)
    reference = vurro_phosphene_sigma_deg(ecc, array.params)
    # Geometric mean of the two axes: the sigma of the equal-area
    # circle, the like-for-like comparison against an isotropic curve.
    equivalent = np.sqrt(np.maximum(major, 0) * np.maximum(minor, 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = equivalent / reference
    return {'eccentricity_deg': ecc,
            'sigma_major_deg': major,
            'sigma_minor_deg': minor,
            'sigma_equivalent_deg': equivalent,
            'vurro_sigma_deg': reference,
            'ratio': ratio}


def implied_k(array, current_ua: float,
              eccentricities_deg: Optional[Sequence[float]] = None
              ) -> dict:
    """Back-solve the K that would match the Vurro cross-reference.

    The rendered sigma is proportional to the recruitment sigma in
    tissue, which under Stoney scales as ``1/sqrt(K)``. So matching a
    target sigma means

        K_implied = K_current * (sigma_model / sigma_target)^2

    evaluated per electrode. The spread across electrodes measures how
    badly the two curves' SHAPES disagree, and is reported alongside the
    median rather than averaged away.

    Read this as a diagnostic, not a calibrated K: the target is human
    psychophysics, it models no electrode or current spread at all, and
    the comparison depends on three further choices that are not
    measurements. `docs/calibration.md` has the detail.
    """
    comparison = size_comparison(array, current_ua, eccentricities_deg)
    k_current = array.kernel.k
    ratio = np.asarray(comparison['ratio'], dtype=float)
    finite = np.isfinite(ratio) & (ratio > 0)
    per_electrode = np.where(finite, k_current * ratio ** 2, np.nan)

    values = per_electrode[finite]
    out = dict(comparison)
    out.update({
        'k_current_ua_per_mm2': k_current,
        'k_implied_per_electrode': per_electrode,
        'k_implied_median': float(np.median(values)) if values.size else
        float('nan'),
        'k_implied_range': (float(np.min(values)), float(np.max(values)))
        if values.size else (float('nan'), float('nan')),
        'current_ua': current_ua,
    })
    return out


def report(array, current_ua: float) -> str:
    """Human-readable calibration summary, caveats included."""
    r = implied_k(array, current_ua)
    ratio = np.asarray(r['ratio'], dtype=float)
    finite = ratio[np.isfinite(ratio)]
    lo, hi = r['k_implied_range']

    lines = [
        f"Size calibration against Vurro et al. (2014), at "
        f"{current_ua:.0f} uA",
        "-" * 62,
        f"  model / reference sigma ratio : median "
        f"{np.median(finite):.2f}, range {finite.min():.2f}-"
        f"{finite.max():.2f}" if finite.size else
        "  no finite comparisons available",
        f"  K in use                      : "
        f"{r['k_current_ua_per_mm2']:.0f} uA/mm^2 "
        f"(transplanted from macaque V1, unvalidated for LGN)",
        f"  K implied by the reference    : median "
        f"{r['k_implied_median']:.0f} uA/mm^2, range {lo:.0f}-{hi:.0f}",
        "",
        "  The implied value is a DIAGNOSTIC, not a calibrated K: the",
        "  reference is human acuity and models no current spread at",
        "  all. The width of the range measures how differently the two",
        "  curves grow with eccentricity. See docs/calibration.md.",
    ]
    return '\n'.join(lines)


def kernel_with_implied_k(array, current_ua: float) -> RecruitmentKernel:
    """A kernel using the back-solved K, for a what-if comparison.

    So the alternative can be rendered and looked at -- not because the
    implied value should be adopted.
    """
    r = implied_k(array, current_ua)
    k = r['k_implied_median']
    if not np.isfinite(k) or k <= 0:
        k = excitability_constant_ua_per_mm2(array.params)
    # Swap only K; every other kernel setting stays as configured, so
    # the comparison is about K alone.
    return dataclasses.replace(array.kernel, k=float(k))
