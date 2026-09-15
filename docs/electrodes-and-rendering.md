# Electrodes and rendering

Code: [`dynaphos_lgn/electrodes.py`](../dynaphos_lgn/electrodes.py) and
[`dynaphos_lgn/rendering.py`](../dynaphos_lgn/rendering.py).
Config: the `electrodes:`, `recruitment:` and `rendering:` sections of
`config/params_lgn.yaml`.

## The setup/forward split

Everything expensive and everything categorical happens once, at array
construction: each electrode's atlas voxel and retinotopic position, a
fixed neighbourhood of candidate voxels sized to the largest current the
protocol will ever deliver, each candidate's layer identity and cell
class and visual-field position, the local Jacobian at the electrode
centre, and the confidence flags that decide how the electrode renders.

The forward pass then only multiplies those fixed tensors by a smooth,
differentiable function of the optimised current. That split is what
makes per-layer recruitment differentiable at all: the *indices* (which
voxels, which layer) are constant and only the *weights* move.

**Electrode positions are fixed**, set at implant time rather than
optimised. Making them optimisable is a materially different problem — a
fixed gather around one position cannot be re-sampled at a nearby
fractional position, because a categorical layer id cannot be
interpolated — and is deliberately out of scope.

## The candidate neighbourhood

Sized from the *effective* kernel radius at maximum current, not the
nominal Stoney radius: a smooth kernel still has weight past its nominal
edge, and truncating there would clip the tail that the smoothness
exists to provide.

A voxel stride is chosen so the candidate count fits the configured
budget. The stride turns the recruitment sum into a regular subsample of
the footprint, and `voxels_per_candidate` records how many real voxels
each candidate stands for, so absolute cell counts stay correct.

**Distance histograms.** The recruitment weight depends only on distance
and current, so the sum of weights over one layer's voxels can come from
a histogram of those voxels' distances instead of from the voxels
themselves. That turns the per-layer activation step from
O(n_electrodes × n_candidates) per forward pass into
O(n_electrodes × n_layers × n_bins) — typically a hundred-fold saving —
with no change to what is computed beyond binning. The discretisation
error is bounded by how much the kernel changes across one bin; at the
default bin count the bin width is well under a hundredth of the gather
radius, far below the atlas's own 211 um accuracy floor.

## Placement

`spread_in_visual_field` farthest-point-samples in *retinotopic* rather
than physical space on purpose: physical voxel density is wildly
non-uniform, so a physically even spread badly under-samples the
periphery.

The shipped `electrodes.depth_selection: random` differs from
`get_electrode_layout`'s own backwards-compatible default. Every
(eccentricity, inclination) bin contains many voxels — one projection
column seen in each lamina — and taking the first puts every electrode
in the same layer. That is fine for a single-lamina study and badly
misleading for anything about per-layer recruitment.

`from_visual_field_targets` matches nearest-neighbour in visual-field
Cartesian coordinates, so the realised position can differ from the
requested one wherever the atlas has no tissue representing it.
`position_error_deg` reports by how much.

Where an electrode lands on a placeholder voxel, the position falls back
to a local neighbourhood average, taken in Cartesian visual-field
coordinates and converted back — averaging inclination directly would
break across the ±180 deg wrap.

## Ellipse or scatter?

`render_mode` picks `scatter` wherever the single-point Jacobian should
not be extrapolated across the footprint — exactly what
`unreliable_beyond_neighborhood` answers — and `ellipse` otherwise.

**With one deliberate exception**: electrodes confined to the central
1 deg get `ellipse`, not `scatter`. Inside that patch every voxel's
position already traces back to Malpeli's isotropic formula (see
[`atlas.md`](atlas.md)), so per-voxel scatter recovers no real
anisotropic detail there — it would just cost more for the same circle.
Scatter still earns its place in the wider near-foveal band just
outside, where the Jacobian is a real interpolated measurement that
simply curves too fast for one point evaluation to summarise.

## Scatter-point selection and bandwidth

Scatter mode renders the recruited point cloud directly instead of
assuming an ellipse, but does not need every candidate to do so. Points
are chosen by **distance rank with a stride**, which keeps the radial
profile representative rather than over-weighting the much more numerous
outer shell the way a uniform random draw would.

The splat kernel that smears each point is a rendering technicality —
continuity, so gradients flow — not a biological claim. Its bandwidth is
therefore sized from the point cloud's own spacing rather than from any
magnification factor; otherwise a per-voxel-Jacobian assumption would
sneak back in through the renderer.

Two corrections go into that floor:

1. **Degenerate pairs are excluded.** `ECC.DAT` is quantised to 0.1 deg
   and `INCL.DAT` to 1 deg, so genuinely distinct voxels routinely share
   a retinotopic position and give zero spacing. Those say nothing about
   how far apart the cloud's *resolvable* points are, so the median runs
   over non-degenerate pairs only; with none, the renderer's half-pixel
   floor takes over.
2. **Uniform-equivalent spacing is considered too.** Nearest-neighbour
   spacing alone under-smooths a cloud that is not uniformly spaced, and
   this one is not: it is a regular voxel lattice projected through a
   quantised retinotopic map, so points cluster tightly along one axis
   while leaving visible gaps along another. Rendering at the tight
   spacing leaves those gaps as blocky structure that is a lattice
   artefact, not a phosphene. The floor takes whichever of the two
   spacings is larger.

Masked-out candidates are parked at the electrode's own centre, so they
must be excluded from the spacing computation or they stack a pile of
coincident points there and drive the median to zero.

`scatter_patch_radius_deg` comes from the actual spread of the
electrode's points about its centre plus a margin, rather than from a
magnification estimate — the point cloud already knows how far it
reaches.

## Phosphene shape

No literature directly measures LGN-phosphene shape. Round/punctate is
converging inference from receptive-field architecture (LGN receptive
fields are circularly symmetric centre-surround) and behavioural proxies
(Pezaris & Reid 2007 describe LGN percepts as "focal") — not a
measurement. Dynaphos's own Gaussian blob rests on human verbal reports
of round flashes rather than on orientation-column architecture, so it
is if anything better justified for LGN than for V1.

Rendered phosphenes are therefore isotropic where the local Jacobian is
isotropic, and elliptical where it is not.

`phosphene_sigma_deg` multiplies the recruitment sigma in tissue (mm) by
the two principal magnifications (deg/mm); an isotropic magnification
gives equal sigmas, i.e. a circle. Everything it reports inherits the
uncertainty in `K` — see [`current-spread.md`](current-spread.md).

## Behavioural benchmarks

From Pezaris & Reid (2007), the single load-bearing macaque LGN
microstimulation paper: accuracy 2.3 ± 1.2 deg, repeatability
0.8 ± 0.6 deg. `calibration.py` compares against these and against the
Vurro et al. (2014) size curve.
