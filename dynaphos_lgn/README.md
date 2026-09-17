# `dynaphos_lgn` — an LGN adaptation of the Dynaphos phosphene simulator

A biologically-grounded, end-to-end differentiable simulator of the
percepts produced by electrical microstimulation of the **lateral
geniculate nucleus**, built as a sibling package to the original
`dynaphos` (V1 / cortex) implementation, which is left untouched and
reused for the genuinely generic parts.

The design goal is **modularity and honest uncertainty, not accuracy**.
Almost nothing about LGN stimulation has been measured: there is no
LGN-specific current-spread constant, no LGN strength-duration data, no
macaque koniocellular receptive-field dataset, no direct measurement of
LGN phosphene shape, and no human LGN implant of any kind. Every number
here is therefore transplanted, borrowed across species, or fitted
in-project — and the package is built so that stays visible in the
output rather than disappearing into a config file.

---

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
| `params.py` | config access: `require` (dotted path, loud failure), `optional`, `resolve_class_codes` |
| `atlas.py` | Erwin et al. (1999) file loading; per-voxel local Jacobian of the retinotopy, with caching and confidence flags; the mirrored right nucleus |
| `synthetic_atlas.py` | a small closed-form stand-in, so tests and demos need no 145 MB download. **Never a result** |
| `magnification.py` | Malpeli et al. (1996) closed forms; three magnification models (Jacobian / atlas-gradient / density-derived) |
| `receptive_fields.py` | difference-of-Gaussians centre and surround sizes by cell class |
| `current_spread.py` | Stoney/Tehovnik law, smooth recruitment kernels, multi-electrode aggregation modes and the threshold-reduction bracket |
| `electrodes.py` | all setup-time geometry: voxel lookup, candidate gather, layer grouping, per-electrode flags |
| `rendering.py` | the elliptical-Gaussian shortcut and the per-voxel scatter renderer |
| `simulator.py` | the forward pass, reusing Dynaphos's temporal-dynamics state classes unchanged |
| `bilateral.py` | both nuclei at once: one percept over the whole visual field |
| `build.py` | one call from a YAML config to a working simulator |

## Where the numbers live

**Every number is in `config/params_lgn.yaml`.** There are no
module-level constants and no defaults in function signatures: a value
you cannot change without editing code is not refinable, and refinement
is this project's whole premise. That includes the things that look
structural — the atlas grid shape, the 999/0 sentinels, the 25 µm voxel
size, the Horsley-Clarke origin, the per-file dtypes and the laminar
code table — so pointing the package at a refined or differently-gridded
atlas is a config change.

Code reads the config through `dynaphos_lgn.params.require`, which takes
a dotted path and fails loudly, naming the missing key and what was
available beside it:

```python
from dynaphos_lgn.params import require
k = require(params, 'current_spread.k_ua_per_mm2')
```

Do **not** use `params['a']['b']`, and never `.get(key, fallback)` — a
fallback in code is the module-level constant this rule exists to
abolish, hidden one level deeper. `params.optional` exists for the few
values whose absence is meaningful (a null `cells_per_mm3` means "derive
it from the atlas"), not as a way around adding a key.

Two consequences worth knowing:

- Where each number came from, how far it had to travel, and how far
  it can be trusted is named in the config's own comments, block by
  block, rather than in the code.
- The published validation targets (Erwin et al.'s cell counts, volume
  and densities) live under `validation:` in the config too. That is
  deliberate but worth stating plainly: **editing them edits the check,
  not the model.**

## Quick start

```python
from dynaphos.utils import load_params
from dynaphos_lgn.build import build_simulator
import torch

params = load_params('config/params_lgn.yaml')
simulator, parts = build_simulator(params)      # add synthetic=True to
                                                # skip the atlas download

print(simulator.describe())                     # array geometry + flags

image = simulator(torch.full((simulator.num_phosphenes,), 80e-6))
```

That covers the **right half** of the visual field. For the whole of it,
ask for both nuclei -- in the config, or per build:

```python
simulator, parts = build_simulator(params, hemispheres=['left', 'right'])
```

and widen `run.view_angle`, because there is now twice as much field to
show. See "One nucleus or two" below.

Or run the demo, which writes four figures including an end-to-end
optimisation and a picture of the K uncertainty:

```
python examples/demo_simulator_lgn.py --synthetic
```

Building the Jacobian cache — the one genuinely expensive setup step —
has its own entry point, which also prints a validation report against
the published totals:

```
python -m dynaphos_lgn.build --cache-jacobian
```

Roughly 90 s and 4.2 GB of peak RAM at the atlas's 21.5M voxels, writing
a ~110 MB `JACOBIAN.npz` beside the `.DAT` files.

## One nucleus or two

The published atlas is a single **left** LGN, and one LGN represents the
**contralateral** hemifield. So on its own the package reaches the right
half of the visual field and no more: the left half of every rendered
frame stays black whatever you feed it.

`atlas.hemispheres` decides how much is built.

```yaml
atlas:
  hemispheres: [left]           # [left] | [right] | [left, right]
```

`[left]` is the default and reproduces everything this package did
before the key existed. `[left, right]` adds a second nucleus and
returns a `BilateralLGNSimulator` instead, which holds one
`LGNPhospheneSimulator` per nucleus and adds their percepts:

```
percept = clamp( sum over BOTH arrays of brightness * p(detect) *
                 activation map , 0, 1 )
```

The clamp lands once, over both, so a bilateral run agrees exactly with
what a single array holding the same electrodes would have produced.
Electrodes are numbered left nucleus first; `electrode_slices` and
`hemisphere_of_electrode` say which is which, and `n_electrodes` is
**per nucleus**, with `n_electrodes_by_hemisphere` overriding it on one
side for an asymmetric implant.

**The right nucleus is the left one reflected, not a second dataset.**
`MirroredAtlas` mirrors three things and nothing else: the ML grid axis,
the inclination (`I -> 180 - I`, i.e. visual-field `x -> -x`), and the
ipsilateral placeholder, which sits at ±135° on the left and so reads as
±45° there. Eccentricity, layer identity and cell counts are untouched
by a reflection and are exposed as NumPy views of the left atlas's own
arrays, so the second nucleus costs one extra inclination volume rather
than a second copy of everything. The Jacobian is derived the same way:
reflecting turns `J` into `diag(-1, 1) J diag(-1, 1, 1)`, an orthogonal
map on each side, so **both principal magnifications -- and so every
phosphene size -- are unchanged**, and only the major axis's orientation
flips. The cached fit is reused; nothing is refitted.

What that assumes is in "Known gaps" below, and travels with every
electrode as `hemisphere == 'right'` and in `describe()`.

```
python examples/demo_bilateral_lgn.py
```

writes a four-panel figure: where the electrodes sit by nucleus, a
target image spanning both hemifields, what one nucleus perceives of it,
and what two do. `examples/demo_simulator_lgn_bilateral.ipynb` is the
longer walkthrough -- the bilateral counterpart of
`demo_simulator_lgn.ipynb`, section for section.

## Four design decisions worth knowing about

**1. The atlas gives position; Malpeli's formula gives magnification.**
The two are not independent sources — Erwin et al. reconstructed the
atlas *starting from* Malpeli et al.'s data, and wrote Malpeli's own
isotropic formula directly into the central 1°. Going through the atlas
for magnification buys interpolation noise, not corroboration. So the
atlas is used for the one thing a closed form cannot give: per-voxel
retinotopic position.

**2. Nothing may be a hard cutoff.** The recruitment kernel is smooth
because a hard in/out distance test has zero gradient with respect to
current, and the optimiser would never move. The same rule drives the
soft bandwidth clamps in the renderer and the optional soft detection
threshold.

**3. Two renderers, chosen per electrode.** The cheap path stretches the
whole activation blob by one Jacobian evaluated at the electrode's
centre. That is only valid where the magnification does not change much
across the footprint, so electrodes where it does are routed to
per-voxel scatter instead — with a deliberate exception inside the
central 1°, where the atlas's own positions are isotropic by
construction and scatter would cost more for the same circle.

**4. Uncertainty is an output, not a footnote.**
`MalpeliDensityMagnification.interval` and `threshold_reduction_bracket`
both return ranges. Where a number can be read two ways — radius versus diameter, integrated
versus peak surround ratio, 1/e radius versus standard deviation — both
readings are computed and named.

## Known gaps and open items

* **The right LGN is an assumption, not a measurement.** There is one
  reconstructed nucleus in the world and it is a left one, so the right
  nucleus here is its mirror image: same volume, same laminar order,
  same retinotopy with the hemifield flipped. That is the textbook
  first approximation and nothing more. Real left and right LGNs differ
  in volume and in cell count both between individuals and between
  sides, and there is no second atlas to say by how much — so the left
  half of a bilateral percept carries its counterpart's uncertainty
  **plus** the assumption of symmetry.
* **The vertical meridian is represented twice.** Voxels at
  inclination ±90° reflect onto themselves, so a bilateral simulation
  covers that strip from both nuclei. The nasotemporal overlap it
  stands for is real, but nothing here models it — the duplication is
  inherited from the reflection, not chosen, and its width is the
  atlas's quantisation rather than a measured overlap.
* **Nothing couples the two nuclei.** They are independent simulators
  with independent state, which is right for the tissue, but it also
  means no binocular or interhemispheric interaction is modelled at
  all.
* **K = 675 µA/mm² is transplanted from macaque V1 and unvalidated.** No
  LGN- or thalamus-specific current-spread constant exists in the
  literature at any confidence level. An earlier attempt to back-derive
  K from Pezaris & Reid (2007)'s behavioural scatter was rejected as not
  scientifically plausible; no numeric uncertainty bound is carried in
  its place.
* **The simulator's sizes run ~4–6× larger than the Vurro et al. (2014)
  cross-reference** at typical currents. The K that would reconcile them
  lands in the same region as the project's earlier, rejected
  behavioural back-derivation — which is as easily two flawed methods
  sharing a bias as two methods converging.
* **The density-derived and atlas-gradient magnification models disagree
  by a factor of roughly 1.3–2.4.** Since the two are not independent
  sources, that ratio indicates a processing or unit-convention problem
  rather than a biological disagreement, and is not yet resolved.
* **Electrode positions are fixed.** Optimising them would need the
  atlas sampled continuously and `LAYERS.DAT` re-represented as smooth
  per-layer occupancy volumes, because a categorical layer id cannot be
  interpolated.
* **Koniocellular cells are modelled from owl monkey data**, and only for
  receptive-field size. No macaque K receptive-field dataset, no K-layer
  magnification function, and no literature connecting K cells to
  microstimulation at all.
* **Multi-electrode interaction defaults to independent summation.** The
  field-superposition and probability-summation alternatives are
  implemented; nothing in the evidence picks between them for LGN.

## Tests

```
python -m pytest dynaphos_lgn/test -q
```

Runs against the synthetic atlas, so no download is needed. Tests that
require the real atlas skip themselves and say so.
