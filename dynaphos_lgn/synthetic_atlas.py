"""A small, closed-form stand-in for the Erwin et al. (1999) atlas.

Same interface as `ErwinAtlas` (attributes, sentinels, layer codes), so
tests and demos run without the real data. Not a model of the LGN: use
it to exercise code, not for results.
"""
from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np

from dynaphos_lgn.atlas import ErwinAtlas
from dynaphos_lgn.params import require


class SyntheticAtlas(ErwinAtlas):
    """An `ErwinAtlas` filled from a closed-form map, configured by the
    `synthetic_atlas` block.

    Eccentricity grows exponentially along AP,
    ``E(s) = E0 * (exp(s / lambda) - 1)``; inclination is linear across
    ML; `n_layers` laminae are stacked along DV with codes cycling
    through 1-4.
    """

    is_synthetic = True

    def __init__(self, params: Mapping):
        super().__init__(params)
        cfg = require(params, 'synthetic_atlas')

        self.atlas_shape = tuple(int(s) for s in cfg['shape_ml_dv_ap'])
        self.max_eccentricity_deg = float(cfg['max_eccentricity_deg'])
        self.eccentricity_scale_deg = float(cfg['eccentricity_scale_deg'])
        self.n_layers = int(cfg['n_layers'])
        self.inclination_span_deg = float(cfg['inclination_span_deg'])
        self.margin_vox = int(cfg['margin_vox'])
        self._density_half_ecc = float(cfg['density_half_eccentricity_deg'])
        self._density_jitter_sd = float(cfg['density_jitter_sd'])
        self._rng = np.random.default_rng(cfg['seed'])
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        n_ml, n_dv, n_ap = self.atlas_shape
        ml = np.arange(n_ml)[:, None, None]
        dv = np.arange(n_dv)[None, :, None]
        ap = np.arange(n_ap)[None, None, :]

        m = self.margin_vox
        inside = ((ml >= m) & (ml < n_ml - m)
                  & (dv >= m) & (dv < n_dv - m)
                  & (ap >= m) & (ap < n_ap - m))

        # --- laminae along DV, with one-voxel interlaminar gaps --------
        usable_dv = n_dv - 2 * m
        band = np.maximum(usable_dv // self.n_layers, 2)
        layer_index = np.clip((dv - m) // band, 0, self.n_layers - 1)
        in_gap = ((dv - m) % band == 0) & (dv > m)
        layer_code = (layer_index % 4) + 1
        layer = np.where(inside & ~in_gap, layer_code, self._missing_layer)
        layer = np.broadcast_to(layer, self.atlas_shape).astype(np.int8)

        # --- eccentricity: exponential along AP ------------------------
        # dE/ds = (E + E0)/lam, i.e. magnification M(E) = lam / (E + E0)
        span_vox = max(n_ap - 2 * m - 1, 1)
        s = (ap - m) / span_vox                       # 0..1 along AP
        lam = 1.0 / np.log1p(self.max_eccentricity_deg
                             / self.eccentricity_scale_deg)
        ecc_deg = self.eccentricity_scale_deg * (np.expm1(s / lam))
        ecc_deg = np.clip(ecc_deg, 0.0, self.max_eccentricity_deg)

        # --- inclination: linear along ML ------------------------------
        u = (ml - m) / max(n_ml - 2 * m - 1, 1)
        incl_deg = (u - 0.5) * self.inclination_span_deg

        ecc_deg = np.broadcast_to(ecc_deg, self.atlas_shape)
        incl_deg = np.broadcast_to(incl_deg, self.atlas_shape)
        valid = np.broadcast_to(inside & ~in_gap, self.atlas_shape)

        # The gaps carry retinotopy but no layer, like the real atlas's
        # interlaminar zones.
        self.eccentricity = np.where(
            inside, np.round(ecc_deg * self._ecc_scale),
            self._missing_ecc_incl).astype(np.int16)
        self.inclination = np.where(
            inside, np.round(incl_deg * self._incl_scale),
            self._missing_ecc_incl).astype(np.int16)
        self.layer = layer

        # --- cell counts: falling with eccentricity, lightly jittered ---
        density = 1.0 / (1.0 + ecc_deg / self._density_half_ecc)
        jitter = self._rng.normal(1.0, self._density_jitter_sd,
                                  size=self.atlas_shape)
        cells = np.where(valid, np.clip(density * jitter, 0, None), 0.0)
        self.cells = np.round(cells * self._cells_scale).astype(np.int16)

        self.update_valid()
        self.eccentricity_deg = self.eccentricity / self._ecc_scale
        self.inclination_deg = self.inclination / self._incl_scale
        self.cells_per_voxel = self.cells / self._cells_scale

    # ------------------------------------------------------------------
    def analytic_magnification_deg_per_mm(self, eccentricity_deg
                                          ) -> Tuple[np.ndarray, np.ndarray]:
        """The map's exact (along-eccentricity, along-inclination)
        magnifications in deg/mm, for checking the Jacobian fit."""
        e = np.asarray(eccentricity_deg, dtype=float)
        n_ml, _, n_ap = self.atlas_shape
        m = self.margin_vox
        span_mm = max(n_ap - 2 * m - 1, 1) * self._voxel_size_mm
        lam_mm = span_mm / np.log1p(self.max_eccentricity_deg
                                    / self.eccentricity_scale_deg)
        d_ecc = (e + self.eccentricity_scale_deg) / lam_mm

        ml_span_mm = max(n_ml - 2 * m - 1, 1) * self._voxel_size_mm
        d_incl_deg_per_mm = self.inclination_span_deg / ml_span_mm
        # Visual-field displacement per degree of inclination is
        # E * dI(rad), so the tangential magnification scales with E.
        d_incl = e * np.deg2rad(d_incl_deg_per_mm)
        return d_ecc, d_incl

    def build_atlas(self, atlas_dir=None) -> 'SyntheticAtlas':
        """No-op: a synthetic atlas is built in `__init__`."""
        return self
