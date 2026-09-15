"""A small, analytic stand-in for the Erwin et al. (1999) atlas.

The real atlas is 21.5M voxels of separately-downloaded binary data, so
this builds a much smaller volume with the SAME INTERFACE -- attribute
names, sentinels, layer codes, index/mm convention -- from a closed-form
retinotopic map, for tests and a first look at the pipeline.

**It is not a model of the LGN and must never be used for results.** Its
retinotopy is a clean exponential by construction: no tears, no
ipsilateral sentinel sparsity, no reconstruction noise, no central-1 deg
isotropy artefact. Anything it validates is code correctness, not
biology. Every instance carries ``is_synthetic = True``.

It does reproduce laminar structure, a foveally-magnified eccentricity
map, genuine local anisotropy, and out-of-nucleus placeholder voxels --
enough to exercise every code path.
"""
from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np

from dynaphos_lgn.atlas import ErwinAtlas
from dynaphos_lgn.params import require


class SyntheticAtlas(ErwinAtlas):
    """An `ErwinAtlas` filled from a closed-form map instead of a file.

    Every knob lives in the config's `synthetic_atlas` block:

    - ``shape_ml_dv_ap``: (ML, DV, AP) voxel counts. The shipped values
      give roughly 2.4 x 1.6 x 5.0 mm, sized so the magnification lands
      in the real atlas's order of magnitude -- a smaller volume crams
      the whole eccentricity range into a millimetre and renders
      phosphenes tens of degrees across.
    - ``max_eccentricity_deg``: Eccentricity at the anterior end.
    - ``eccentricity_scale_deg``: E0 in
      ``E(s) = E0 * (exp(s / lambda) - 1)``, the eccentricity at which
      the map switches from roughly linear to roughly exponential.
      Smaller values concentrate more tissue on the fovea.
    - ``n_layers``: How many laminae to stack along DV. Codes cycle
      through the real LAYERS.DAT values 1-4.
    - ``inclination_span_deg``: Inclination range mapped across ML.
    - ``seed``: Reproducibility for the cell-count jitter.
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
        # E(s) = E0 * (exp(s / lam) - 1), so dE/ds = (E + E0)/lam, i.e.
        # linear magnification M(E) = lam / (E + E0) -- the same
        # functional shape as the monopole cortical model, which is what
        # gives a realistically steep foveal magnification.
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

        self.eccentricity = np.where(
            valid, np.round(ecc_deg * self._ecc_scale),
            self._missing_ecc_incl).astype(np.int16)
        self.inclination = np.where(
            valid, np.round(incl_deg),
            self._missing_ecc_incl).astype(np.int16)
        self.layer = layer

        # --- cell counts: falling with eccentricity, lightly jittered ---
        density = 1.0 / (1.0 + ecc_deg / self._density_half_ecc)
        jitter = self._rng.normal(1.0, self._density_jitter_sd,
                                  size=self.atlas_shape)
        cells = np.where(valid, np.clip(density * jitter, 0, None), 0.0)
        self.cells = np.round(cells * self._cells_scale).astype(np.int16)

        self.valid = ((self.eccentricity != self._missing_ecc_incl)
                      & (self.inclination != self._missing_ecc_incl)
                      & (self.layer != self._missing_layer))
        self.eccentricity_deg = self.eccentricity / self._ecc_scale
        self.inclination_deg = self.inclination.astype(float)
        self.cells_per_voxel = self.cells / self._cells_scale

    # ------------------------------------------------------------------
    def analytic_magnification_deg_per_mm(self, eccentricity_deg
                                          ) -> Tuple[np.ndarray, np.ndarray]:
        """The map's true (AP, ML) magnifications, for checking the fit.

        Because the synthetic map is closed-form, the Jacobian machinery
        can be validated against the exact answer instead of against
        another estimate. Returns (along-eccentricity, along-inclination)
        magnification in deg/mm.
        """
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
