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
| `viz3d.py` | the 3-D viewer: the nucleus as a voxel surface, every per-voxel map painted onto it, and the electrodes inside it |
| `slices.py` | the slice viewer: the same volume and the same overlays, cut coronally, sagittally or horizontally |
| `explore.py` | the live viewer: the same volume in a notebook, with a kernel attached so the controls need nothing precomputed |

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

## Looking at the atlas

```
python -m dynaphos_lgn.viz3d
```

draws the nucleus in Horsley-Clarke millimetres as a voxel surface and
writes an interactive HTML file: rotate it, and switch what is painted
on it from a dropdown. `--list-overlays` names everything it can paint.

| overlay | what it is |
|---|---|
| `eccentricity`, `inclination` | the atlas's retinotopy, as published |
| `cell_class`, `layer`, `eye` | magno/parvo, the four laminae, and which eye drives each |
| `cells_per_voxel` | Erwin et al.'s reconstructed counts |
| `magnification_major`, `magnification_minor`, `anisotropy`, `magnification_areal`, `magnification_orientation` | the per-voxel Jacobian of the retinotopy, decomposed |
| `jacobian_residual`, `jacobian_flags` | how far that local linear fit can be trusted, voxel by voxel |

The first six come straight from the four `.DAT` files. The rest need
the cached Jacobian, so they are opt-in: `--overlay all`, or name them.

**Three things the picture will not do.** It does not smooth,
interpolate or fill in: a drawn voxel takes its value from one real
atlas voxel, chosen as the one nearest the centre of the block it
stands for. Tissue the atlas has no value for is grey, in a band
reserved below the colour scale, so "not measured" cannot be read as
"measured, and small". And the two maps the package itself flags as
shaky are overlays you can select rather than something hidden behind
the pretty ones.

### Seeing inside

The laminae are interior structure, and an opaque surface hides them.
Two ways in, and they combine:

```
python -m dynaphos_lgn.viz3d --overlay layer --clip ap:6:9
python -m dynaphos_lgn.viz3d --overlay layer --surface-opacity 0.35
```

`--clip AXIS:LOW:HIGH` cuts the volume open in millimetres. The cut
face is a surface like any other and is painted like any other, so a
clipped run is a cross-section rather than a way of hiding tissue.
`--cell-class`, `--layers`, `--ecc-min` and `--ecc-max` narrow what is
drawn the same way the simulator narrows where it will place an
electrode.

`--stride N` trades detail for speed: one drawn voxel per N atlas
voxels along each axis, `4` by default, `1` for the full 25 µm grid.
A block is drawn whenever ANY atlas voxel in it is tissue, so a
one-voxel lamina survives a stride that would step over it.

### Electrodes

```
python -m dynaphos_lgn.viz3d --electrodes layout --surface-opacity 0.35
python -m dynaphos_lgn.viz3d --electrodes array --colour-by render_mode
python -m dynaphos_lgn.viz3d --probe 'entry=11,5.5,8;target=11,0,8;n=10'
python -m dynaphos_lgn.viz3d --grid 'centre=10.5,2,7.5;plane=dv,ap;n=4,4;pitch=0.5,0.5'
python -m dynaphos_lgn.viz3d --electrodes-file contacts.csv
```

`layout` places the array this config describes, through the same
`get_electrode_layout` the simulator's own constructor uses. `array`
builds the real `LGNElectrodeArray`, which costs a Jacobian load and in
exchange carries what the simulator concluded about each electrode --
its realised retinotopic position, both principal magnifications, which
renderer it was routed to, and every confidence flag it raised -- into
the hover text. `--colour-by` paints the contacts by lamina, cell
class, nucleus, renderer or flag.

The geometric layouts take Horsley-Clarke millimetres, where `ml` is
lateral distance from the midline and so is positive in both nuclei;
`hemisphere=right` inside the spec is what puts a shank in the right
one. A contact that lands off the atlas grid is drawn hollow and says
so when you hover it, rather than quietly reporting the nearest voxel's
tissue as its own.

`--show-spread` adds the tissue each contact recruits, as a sphere of
the recruitment kernel's own radius at
`visualization.electrodes.spread_current_ua` -- from
`RecruitmentKernel`, so the picture and the forward pass cannot
disagree about how far a current reaches.

### Slices

A surface can only show you its outside, and the laminae are on the
inside. `dynaphos_lgn.slices` cuts the same volume into planes and
paints the same overlays on them:

```
python -m dynaphos_lgn.slices                                  # ortho triple
python -m dynaphos_lgn.slices --view slices --plane coronal --count 12
python -m dynaphos_lgn.slices --overlay layer --electrodes layout
python -m dynaphos_lgn.slices --at 11,2.0,7.5 --backend matplotlib
```

Three views. **ortho** (the default) cuts all three planes through one
point and draws a crosshair on each showing where the other two went;
`--at` puts that point somewhere specific, in scene millimetres or at
`electrode:N`, and defaults to the centroid of the drawn tissue.
**slices** steps one plane through the nucleus: matplotlib draws a
montage, plotly gives a slider. **combined** is the two halves of this
package in one figure, and has its own section below.

It shares every option it can with the 3-D viewer -- `--overlay`,
`--clip`, `--cell-class`, `--electrodes`, `--probe`, `--grid`,
`--colour-by`, `--hemispheres`, `--backend` -- from the same builder,
so a flag means the same thing in both. The two it drops are `--stride`
and `--surface-opacity`: a slice is one voxel thick and fully opaque,
so they would do nothing.

**A slice is one voxel thick.** 25 µm, not an average through a slab,
and never interpolated -- one image pixel is one atlas voxel or it is
nothing. Three states stay distinguishable on every panel, and are
meant to: **background** where there is no voxel at all, **grey**
where there is a voxel the atlas has no value for, and **colour**
where there is a value. Orientation letters (L/R, D/V, P/A) sit on
each panel's edges, and dorsal is up in every plane that has a dorsal.

Colour limits come once from a sample of the whole selected volume,
never from the plane on screen. A scale set per slice would make every
slice look the same and none of them comparable, and the slider would
move the colours as well as the tissue.

Electrodes are points and a plane is 25 µm thick, so contacts within
`--slab` millimetres of the plane are drawn on it, faded with distance
and carrying their real offset into the hover. The slab is a drawing
tolerance, and says so.

With both nuclei, a coronal or horizontal panel holds them in the
neurological convention with the real gap between them to scale, and a
sagittal position picks a nucleus by its sign. Nothing is resampled to
do that: a scene column IS an atlas ML index, and the grid only
records which nucleus's index it is.

### The slice and the nucleus together

```
python -m dynaphos_lgn.slices --view combined
python -m dynaphos_lgn.slices --view combined --plane sagittal --count 16
```

The nucleus as a see-through shell on the left, the active slice flat
on the right, and that same slice drawn where it actually cuts, inside
the shell. One slider moves all three, so the flat panel always has a
position and the position always has a picture. Electrodes within the
slab are highlighted in the scene as well as drawn on the panel, which
is the quickest way to see which contacts a given plane passes
through, and the overlay dropdown works as it does everywhere else.

The shell is **context, not data**: one colour, see-through, because
two coloured surfaces in one scene fight each other. To paint the
nucleus's own surface with an overlay, that is what `viz3d` is for.

Three things about it worth knowing:

- The in-scene copy of the slice is coarser than the flat one
  (`combined.plane_stride_vox`). It is there to show where the cut is,
  and a 25 µm texture on a plane seen edge-on costs a lot to say very
  little. Still every nth voxel, never the mean of n.
- There is no Play button, and that is not an oversight. Stepping
  through frames is a plotly *animation*, and plotly cannot animate a
  heatmap: the cut plane would move and the flat panel would simply
  stop being drawn. The slider restyles instead, which redraws every
  kind of trace -- at the cost of writing its numbers into the file as
  JSON rather than as binary, so they are rounded to what the colour
  axis and the 25 µm grid can actually show.
- It spans only planes holding at least
  `combined.min_slice_fraction` of the fullest one, so it opens on a
  cross-section rather than on the nucleus's tapered end.
- One plane per figure. Coronal, sagittal and horizontal each carry
  their own set of slices, and offering all three at once would triple
  a file that is already the largest thing here. Run it three times.

The matplotlib backend draws the same pair of panels as a still, at
the middle slice, for when it has to go in a document.

### Live, in a notebook

```
jupyter lab examples/demo_lgn_explorer.ipynb
python -m dynaphos_lgn.explore          # checks the dependencies
```

Everything above writes a **figure**: an HTML file you can email, in
which every state a control can reach was computed in advance. That is
what makes it shareable, and it is also what caps how interactive it
can be -- a slider there is a lookup table, so an arbitrary slice
position, a different spacing between slices, or a switch from coronal
to sagittal is not something the file can do. It would have to contain
every one of them.

`dynaphos_lgn.explore` is the other half: the same volume, the same
overlays, the same colours and the same electrodes, driven from
ipywidgets with the kernel still attached. Nothing is precomputed, so
the controls are free. Position steps one 25 µm voxel at a time
anywhere in the volume; **count** and **spacing** turn the single cut
into a stack of parallel planes whose separation you set in
millimetres; plane, overlay, slab, shell opacity and stride are all
live; and cell class, laminae and eccentricity rebuild on demand.

The spacing snaps to whole voxels, and the stack is built outwards from
the plane the flat panel is showing -- so a spacing the 25 µm grid
cannot express moves the *others*, never the one you are reading. That
plane is the opaque one; the rest of the stack is not.

Three things can legitimately differ from what you asked for, and the
line under the figure reports all three rather than assuming them: the
snapped spacing, the planes a stack near the end of the nucleus loses,
and a position that has landed where there is no tissue.

It needs `ipywidgets`, `plotly` and `anywidget`, and it writes no file.

### Exporting what you are looking at

```python
ex = explore.Explorer.open(hemispheres=['left'], electrodes='layout')
ex.widget
print(ex.export().text)      # the YAML, and the command that redraws it
ex.export().save()           # ... under visualization.output.directory
```

Every control writes its value into the parameter dictionary at the
dotted path the rest of the package reads it from, so the explorer's
state *is* a configuration rather than a description of one, and
exporting is a diff against `config/params_lgn.yaml` rather than a
second inventory of what the controls happen to say. Two forms come
back, and they are not alternatives:

- a **YAML fragment** of everything that differs from the file on disk,
  nested so it can be merged back by hand. The durable record.
- a **command line** that renders the same picture as a static figure.
  It restates as flags everything that has one, so in the ordinary
  case it needs nothing else.

The two cannot disagree: each flag folds into exactly the config key
the explorer wrote it to. The one part of the state with flags but no
config keys is the tissue selection, which is why `--cell-class`,
`--layers`, `--ecc-min/max` and `--clip` are rendered from the
selection rather than diffed out of the config.

`save(full=True)` writes the whole parameter dictionary instead, for
`--params` to be pointed at. It is also what a position inside the
nucleus's tapered end needs: an explicit range there is trimmed by
`min_slice_fraction`, no flag lifts the trimming, and the export says
so rather than leaving you to discover that the command drew a
different slice.

**What it costs.** plotly's widget ships only one-dimensional arrays
to the browser as binary, and every array a plane or a panel is made
of has two dimensions, so a redraw is a few megabytes of JSON: about
100 ms for one nucleus, nearer 350 ms for both with a stack of nine,
because a coronal panel spanning both then has the real 17 mm midline
gap through the middle of it. The knobs are the plane count, the plane
stride and how many nuclei you open; sliders redraw on release rather
than during a drag, which `visualization.explorer.continuous_update`
controls.

### Both nuclei, and the other backend

```
python -m dynaphos_lgn.viz3d --hemispheres left right --colour-by hemisphere
python -m dynaphos_lgn.viz3d --backend matplotlib --overlay eccentricity layer
```

`--hemispheres left right` draws the mirrored nucleus beside the
published one, on the other side of a signed ML axis (left nucleus at
negative x, in the neurological convention). It is the same reflection
the simulator uses, with the same assumption behind it.

The matplotlib backend needs no extra dependency and writes a PNG with
one panel per overlay; the plotly backend needs `pip install plotly`
and gives the interactive file. Both read the same geometry and the
same colormaps, so a panel and a rotation show the same colours for the
same number.

Every default -- stride, colormaps, colour limits, marker sizes,
lighting, output paths -- is in the config's `visualization` block, and
every one has a flag.

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
