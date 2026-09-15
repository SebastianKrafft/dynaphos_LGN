"""LGN adaptation of the Dynaphos phosphene simulator.

The sibling package `dynaphos` (V1 / cortex) is left untouched and is
imported from here for the genuinely generic pieces (image processing,
differentiable utilities, the leaky-integrator state classes). Everything
that is physiologically LGN-specific -- retinotopic position lookup,
magnification, receptive-field size, current spread, per-layer
recruitment and phosphene rendering -- lives in this package.

See `claude/repo-structure-decision-2026-09-04.md` in the project vault
for why this is a sibling package rather than a subclass/strategy inside
`dynaphos.cortex_models`.

Every number the package uses comes from the parameter file; see
`params.py` for how it is read. Where those numbers come from, and
how far they had to travel to be usable here, is tracked in the
project documentation rather than in code.
"""
