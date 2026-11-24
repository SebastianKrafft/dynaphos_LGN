# `dynaphos_lgn`: an LGN adaptation of the Dynaphos phosphene simulator

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
        |  fixed candidate neighborhood, sized to I_max
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
| `atlas.py` | Erwin et al. (1999) atlas loading; the per-voxel Jacobian of its retinotopy; the mirrored right nucleus |
| `fovea.py` | the central 1 deg of eccentricity, rebuilt from the atlas notes' recipe |
| `projection.py` | projection columns: the tissue representing one visual-field point |
| `synthetic_atlas.py` | a small closed-form stand-in, so tests and demos run without the atlas download |
| `magnification.py` | Malpeli et al. (1996) fits; phosphene magnification from the Jacobian |
| `receptive_fields.py` | difference-of-Gaussians center and surround sizes by cell class |
| `current_spread.py` | Stoney/Tehovnik current spread and the Gaussian recruitment kernel |
| `lgn_utils.py` | electrode placement spread over the visual field |
| `electrodes.py` | setup-time geometry: voxel lookup, candidate gather, layer grouping, per-electrode flags |
| `aggregation.py` | multi-electrode aggregation: field superposition or independent |
| `merging.py` | merged phosphenes: connected recruited volumes, each drawn from its own tissue |
| `rendering.py` | the elliptical-Gaussian and per-voxel scatter renderers |
| `simulator.py` | the forward pass for one nucleus |
| `bilateral.py` | both nuclei at once: one percept over the whole visual field |
| `build.py` | from a config to a working simulator; the Jacobian cache command |
| `report.py` | text reports checking the atlas and the Jacobian fit against the paper |

## Configuration

All model parameters live in `dynaphos_lgn/config/params_lgn.yaml`, including the atlas
grid, sentinels, voxel size and layer codes, each with its source. Code
reads them as a nested dictionary, as in Dynaphos:

```python
from dynaphos.utils import load_params
from dynaphos_lgn.params import PARAMS_PATH

params = load_params(str(PARAMS_PATH))
k = params['current_spread']['k_ua_per_mm2']
```

The `validation:` block holds Erwin et al.'s published totals, which
`python -m dynaphos_lgn.build --check-only` checks the loaded atlas
against.

## Atlas data

`data/Erwin_Atlas/` holds the four `.DAT` volumes from
malpeli.psychology.illinois.edu/atlas, plus `foveola.dat`: the projection
column of the center of the fovea, as 0-based (ML, DV, AP) voxel indices.
It is narrower than the ECC.DAT == 0 region, because ECC.DAT rounds
eccentricities below 0.05 deg to 0. It is optional, and
`ErwinAtlas.foveola_column` is None without it.

Within the central 1 deg, eccentricity was not interpolated from
recordings. Erwin et al. ranked voxels between this column and the 1 deg
isoeccentricity surface, then assigned 0.1 deg zones outward so that each
zone's laminar volume matched Malpeli et al. (1996) Eq. 2 (the site's
"Eccentricity Assignments for the Central 1 Degree" notes). ECC.DAT
zone volumes follow V dE times a constant (~1.6) rather than the
hemifield area's pi V E dE, which puts too much tissue near the fovea.
By default
(`atlas.central_eccentricity: rebuilt`) the loader redoes the recipe with
the area factor, unrounded (`fovea.py`); `published` keeps ECC.DAT, with
its own Jacobian cache.

The column crosses only the parvo layers. `fovea.foveal_references`
gives every layer its closest approach to it, and `build.py`'s atlas
report lists them: magno layer 2 comes within 0.12 mm (0.31 deg), magno
layer 1 within 0.31 mm (0.72 deg). Below those eccentricities a layer
has no tissue in this atlas.

`projection.projection_column(atlas, params, E, I)` traces the tissue
representing any visual-field point: in each layer, the voxels within
`atlas.projection_column` tolerance of it, or the layer's closest ones
if it never gets that near. It traces (0, 0) onto the foveola's column,
and columns elsewhere run obliquely across all four laminae, 2-3 mm
long.

At zero eccentricity, inclination is undefined. The loader moves the +-135
deg ipsilateral placeholder found there onto the vertical meridian, so the
fovea is not excluded with the ipsilateral tissue.

## Quick start

```python
import torch

from dynaphos.utils import load_params
from dynaphos_lgn.build import build_simulator
from dynaphos_lgn.params import PARAMS_PATH

params = load_params(str(PARAMS_PATH))
simulator, parts = build_simulator(params)      # synthetic=True skips
                                                # the atlas download
print(simulator.describe())

image = simulator(torch.full((simulator.num_phosphenes,), 80e-6))
```

The demo writes three figures, including an end-to-end optimization:

```
python examples/demo_simulator_lgn.py --synthetic
```

Build the Jacobian cache once per checkout (about 90 s, 4 GB of RAM and a
110 MB file); it also prints a validation report:

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

## Design notes

* **Nothing is a hard cutoff.** The recruitment kernel, the renderer's
  bandwidth clamps and the default detection threshold are smooth, so
  gradients reach every electrode.
* **Two renderers, chosen per electrode.** The ellipse renderer
  stretches the whole blob by the Jacobian at the electrode's center.
  Where that map should not be extrapolated across the footprint, the
  electrode is rendered by per-voxel scatter instead.

## Known limitations

* K, the rheobase and the temporal dynamics come from V1.
* Only the magno and parvo laminae are modeled; koniocellular cells and
  phosphene color are not.
* The right LGN is a mirror image of the left, not a second atlas.
* Nothing couples the two nuclei.
* Electrodes combine by field superposition by default: drives sum
  before the recruitment threshold, from each electrode's distance
  histogram and its neighbors' shell-averaged field. `independent` is
  the alternative, and no LGN data picks between them. Summation assumes
  simultaneous pulses (`aggregation.simultaneous_pulses`).
* A merged phosphene is tissue-level in shape only: electrodes whose
  summed field recruits one connected volume render as one phosphene,
  drawn from that volume's own voxels, with its brightest member's
  brightness and detection probability, since those are calibrated
  per electrode.
* Electrode positions are fixed; they are not optimizable.

## Tests

```
python -m pytest dynaphos_lgn/test -q
```

Runs against the synthetic atlas. Tests that need the real atlas skip
themselves when it is absent.
