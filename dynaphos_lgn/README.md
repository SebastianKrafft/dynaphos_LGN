# `dynaphos_lgn` — an LGN adaptation of the Dynaphos phosphene simulator

An end-to-end differentiable simulator of the percepts produced by
electrical microstimulation of the **lateral geniculate nucleus**. It is
a sibling package to the original `dynaphos` (V1 / cortex), which it
reuses for image processing, utilities and the temporal dynamics.

## What it does

```
electrode position (mm, Horsley-Clarke)
        |  atlas lookup, once, at setup
        v
atlas voxel  ->  retinotopic position (ecc, incl)  +  layer id  +  cell count
        |  fixed candidate neighbourhood, sized to I_max
        v
per-voxel offsets (mm), layer one-hot, visual-field positions
        |
        |  ---- every forward pass, differentiable in the stimulation params ----
        v
amplitude, pulse width, frequency
        -> charge/s -> leaky-integrator activation, memory trace, brightness
        -> smooth recruitment kernel over the candidate voxels
              -> per-lamina activation totals
              -> phosphene size, via the local Jacobian of the retinotopy
        -> rendered percept (elliptical Gaussian, or per-voxel scatter)
        -> loss -> backprop to amplitude / pulse width / frequency
```

## Modules

| module | what it owns |
|---|---|
| `params.py` | config access: `require`, `optional`, `resolve_class_codes` |
| `atlas.py` | Erwin et al. (1999) atlas loading; the per-voxel Jacobian of its retinotopy; the mirrored right nucleus |
| `synthetic_atlas.py` | a small closed-form stand-in, so tests and demos run without the atlas download |
| `magnification.py` | Malpeli et al. (1996) densities; the magnification models (Jacobian, atlas-gradient, anisotropic-gradient, density-derived) |
| `receptive_fields.py` | difference-of-Gaussians centre and surround sizes by cell class |
| `konio.py` | koniocellular layers K1-K6 labelled in the atlas's interlaminar zones |
| `current_spread.py` | Stoney/Tehovnik law, smooth recruitment kernels, the multi-electrode threshold-reduction bound |
| `electrodes.py` | setup-time geometry: voxel lookup, candidate gather, layer grouping, per-electrode flags |
| `rendering.py` | the elliptical-Gaussian and per-voxel scatter renderers |
| `simulator.py` | the forward pass for one nucleus |
| `bilateral.py` | both nuclei at once: one percept over the whole visual field |
| `build.py` | one call from a YAML config to a working simulator; the Jacobian cache CLI |

## Configuration

All parameters live in `config/params_lgn.yaml`, including the atlas
grid, sentinels, voxel size and layer codes. Code reads them with
`dynaphos_lgn.params.require`, which names the missing key on failure:

```python
from dynaphos_lgn.params import require
k = require(params, 'current_spread.k_ua_per_mm2')
```

The `validation:` block holds Erwin et al.'s published totals, which
`build.py` checks the loaded atlas against.

## Quick start

```python
from dynaphos.utils import load_params
from dynaphos_lgn.build import build_simulator
import torch

params = load_params('config/params_lgn.yaml')
simulator, parts = build_simulator(params)      # synthetic=True skips
                                                # the atlas download
print(simulator.describe())

image = simulator(torch.full((simulator.num_phosphenes,), 80e-6))
```

The demo writes three figures, including an end-to-end optimisation:

```
python examples/demo_simulator_lgn.py --synthetic
```

Build the Jacobian cache once per checkout (~90 s, ~4.2 GB RAM, ~110 MB
file); it also prints a validation report:

```
python -m dynaphos_lgn.build --cache-jacobian
```

## One nucleus or two

The atlas is a single **left** LGN, which covers the **right** half of
the visual field. `atlas.hemispheres` decides how much is built:

```yaml
atlas:
  hemispheres: [left]           # [left] | [right] | [left, right]
```

or per build with `build_simulator(params, hemispheres=['left', 'right'])`
(widen `run.view_angle` too). Two nuclei give a `BilateralLGNSimulator`,
which holds one `LGNPhospheneSimulator` per nucleus and sums their
unclamped percepts before a single clamp. Both classes offer the same
per-electrode views (`vf_xy`, `electrode_slices`,
`hemisphere_of_electrode`, `phosphene_sigma_deg`, `simulators`), and
`parts['hemispheres']` always maps each nucleus to its parts.
Electrodes are numbered left nucleus first, and `n_electrodes` is per
nucleus (`n_electrodes_by_hemisphere` overrides one side).

The right nucleus is the left atlas mirrored (`MirroredAtlas`): the ML
axis is reversed and inclination becomes `180 - I`. The Jacobian becomes
`diag(-1, 1) J diag(-1, 1, 1)`, so phosphene sizes are unchanged and only
orientations are mirrored; the cached fit is reused.

```
python examples/demo_bilateral_lgn.py
```

compares the percept of one nucleus with that of both.
`examples/demo_simulator_lgn.ipynb` is the long walkthrough and
`examples/demo_simulator_lgn_simple.ipynb` the short one.

## Koniocellular layers

The atlas labels only the magno and parvo laminae, but its interlaminar
zones carry retinotopy. With `konio.enabled`, these are labelled K1-K6
(Kn lies just ventral to principal layer n) and given 10% of all cells,
spread uniformly. Their median thickness is 100 um, as reported for K
layers. The Jacobian is then fitted with them included and cached as
`JACOBIAN_KONIO.npz` (`python -m dynaphos_lgn.build --cache-jacobian
--konio`).

Cell classes only change the percept with the class-mixture model:

```yaml
percept:
  model: class_mixture          # single (default) | class_mixture
  classes: [magno, parvo, konio]   # or konio_ventral, konio_dorsal
```

Each phosphene is then a sum of one blob per class, widened by that
class's receptive-field centre and weighted by its share of the
recruited cells. Rendering costs about one pass per class.
`electrodes.cell_class: [magno, parvo]` keeps the electrodes of a run
without K.

```
python examples/demo_konio_lgn.py
```

## Design notes

* **Nothing is a hard cutoff.** The recruitment kernel, the renderer's
  bandwidth clamps and the default detection threshold are smooth, so
  gradients reach every electrode.
* **Two renderers, chosen per electrode.** The ellipse renderer
  stretches the whole blob by the Jacobian at the electrode's centre.
  Where that map should not be extrapolated across the footprint, the
  electrode is rendered by per-voxel scatter instead.
* **Some outputs are ranges.** `MalpeliDensityMagnification.interval`
  and `threshold_reduction_bracket` return low and high values.

## Known limitations

* K, the rheobase and the temporal dynamics come from V1; koniocellular
  receptive fields from owl monkey (Xu et al. 2002) and marmoset (White
  et al. 2001).
* Koniocellular cells are speculative: uniform density, no macaque data,
  retinotopy interpolated by the atlas. K bridges, K-o orientation
  tuning, latency and colour are not modelled.
* The right LGN is a mirror image of the left, not a second atlas.
* Nothing couples the two nuclei.
* Electrodes combine independently at render time; the multi-electrode
  threshold reduction is only reported (`threshold_reduction_bracket`).
* Electrode positions are fixed; they are not optimisable.

## Tests

```
python -m pytest dynaphos_lgn/test -q
```

Runs against the synthetic atlas. Tests that need the real atlas skip
themselves when it is absent.
