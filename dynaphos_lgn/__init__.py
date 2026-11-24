"""LGN adaptation of the Dynaphos phosphene simulator.

Reuses `dynaphos` for image processing, utilities and the temporal
state classes; everything LGN-specific (atlas lookup, magnification,
current spread, recruitment, rendering) lives here. All parameters come
from `dynaphos_lgn/config/params_lgn.yaml`.
"""
