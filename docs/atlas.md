# The Erwin atlas and its Jacobian

Code: [`dynaphos_lgn/atlas.py`](../dynaphos_lgn/atlas.py).
Config: the `atlas:` section of `config/params_lgn.yaml`.

## The source data

Erwin, Baker, Busen & Malpeli (1999): a voxelised macaque LGN atlas of
25 um isotropic voxels on a 240 × 280 × 320 (ML, DV, AP) grid, released
as four headerless binary files — `ECC.DAT`, `INCL.DAT`, `LAYERS.DAT`,
`CELLS.DAT`.

Every property of the grid and the file format — shape, voxel size,
ordering, per-file dtype and scale factor, sentinel values, the
Horsley-Clarke origin and the laminar code table — is read from the
config rather than hardcoded, so a refined or differently-gridded atlas
is a config change instead of a code change.

**Accuracy floor.** The reconstruction shows a mean 211 um (SD 217 um)
discrepancy between recorded sites and the reconstructed projection
column. Nothing downstream can be more accurate than that.

### File format details worth knowing

**Fortran order.** The atlas's documentation says the first 240 (ML)
entries in the flat file form one ML line, the next 240 the adjacent ML
line in the same coronal plane (×280), then that repeats per coronal
plane (×320 AP). ML is the fastest-varying axis — Fortran order for
shape (240, 280, 320), not NumPy's default C order.

**Headerless means a wrong dtype fails silently.** It reinterprets the
whole volume rather than raising, which is why `_DTYPES` is a restricted
allowlist and why `build.check_atlas_files` measures each file's size
against the dtype it will be read with.

**Horsley-Clarke origin.** Per the atlas's own documentation: "the
Horsley-Clarke position of the origin of this coordinate system is
8.5 mm lateral, −1.5 mm dorsal, and 3.5 mm anterior, with increasing
array indices corresponding to increasing distance in the lateral,
dorsal and anterior directions". That confirms
`index = (coord − origin) / voxel_size`, direction included, on all
three axes.

**float32, not float64.** At 21.5M voxels each derived view is 172 MB in
float64 and 86 MB in float32, and the underlying data carries far less
precision than either: `ECC.DAT` is quantised to 0.1 deg, `INCL.DAT` to
whole degrees, `CELLS.DAT` to 0.001 cells/voxel.

## Two constructed regions, not measured ones

**The central 1 deg is isotropic by construction.** Erwin et al. (1999),
p. 94: "Magnification changes most dramatically near the fovea;
therefore, for the central 1 deg a simple linear interpolation resulted
in unacceptable deviations from the original electrophysiological data.
Instead, we assigned eccentricity values by using the volume
magnification function of Malpeli et al. (1996)." That function is
isotropic — a function of eccentricity alone — so `ECC.DAT`/`INCL.DAT`
inside this radius encode a perfectly radially symmetric field by
construction. Any anisotropy a local Jacobian fit finds there is an
artefact of that construction, not real retinotopy. The radius is
`atlas.isotropic_construction_radius_deg`.

**The ipsilateral hemifield carries a flat ±135 deg placeholder.** Per
`INCL.DAT`'s own documentation, the two ipsilateral-hemifield quadrants
get a coarse ±135 deg value at *every* eccentricity, not the finely
resolved 15 deg sectors used contralaterally — only 17 ipsilateral
recording sites existed in the original study, too few to do better.

Excluding these voxels removes up to ~21% of near-fovea parvocellular
voxels. They are not missing data; they are a genuinely different part
of the visual field represented at a different resolution. The Jacobian
fit excludes them by default
(`atlas.exclude_ipsi_flat_inclination`) because their inclination does
not vary continuously with position, so including them would corrupt the
local gradient estimate.

## Why the Jacobian differentiates (x, y), not (E, I)

`JacobianAtlas.compute` fits a local linear map from physical mm offset
to visual-field offset in the Cartesian frame
`(x, y) = E·(cos I, sin I)`, not in raw (eccentricity, inclination).

Polar coordinates have two properties a local linear fit handles badly:
a singularity at the fovea (inclination is undefined at eccentricity 0 —
the paper describes every inclination sector converging there), and a
locally varying scale (a 1-degree change in inclination spans a very
different visual angle near the fovea than in the periphery). The
Cartesian frame avoids both, though the fit right at the foveal
singularity is still not meaningful for the same underlying reason.

## How the fit is computed at 21.5M voxels

Solving one least-squares problem per voxel in a Python loop — ~3.6M
independent problems — would be far too slow. Instead:

1. **Accumulate the weighted normal equations for every voxel at once,
   per neighbour offset.** Each offset contributes a single physical
   displacement vector `a` shared by every voxel, so its contribution
   everywhere is `w · outer(a, a)` and `w · outer(a, dq)`, with a
   per-voxel 0/1 weight `w` marking whether that neighbour is valid
   there.
2. **Store `AᵀA` as six scalar `(*shape,)` fields** rather than one
   `(*shape, 3, 3)` array, to avoid rebuilding a multi-GB broadcast
   temporary on every neighbourhood offset.
3. **Solve the per-voxel 3×3 systems with a closed-form symmetric
   inverse** (elementwise arithmetic over the six unique components)
   instead of `np.linalg.solve`, which for ~21M tiny systems is
   dominated by per-call LAPACK overhead rather than the linear algebra.
4. **Get the residual from the OLS identity `RSS = y'y − β'(X'y)`**,
   reusing quantities already accumulated, instead of a second pass over
   every neighbour offset.
5. **Shift with explicit slices, not `np.roll`**, so volume edges do not
   wrap around.

Arrays are freed as soon as they are consumed. At 21.5M voxels every
full-grid float32 array is 86 MB and the routine would otherwise hold
around forty alive at once — the difference between running and not
running on a laptop.

`decompose_valid` SVDs only the fit-ok voxels, in chunks, and scatters
the results back into full-grid arrays. On the real atlas that is ~3.5M
small SVDs instead of ~21.5M, and it keeps LAPACK's float64 temporaries
from peaking at several gigabytes.

The `ridge` parameter is a tiny diagonal regulariser so that voxels with
a rank-deficient (but still ≥ `min_neighbors`) neighbourhood do not
raise a linear-algebra error. It is negligible next to well-determined
voxels' own normal-equation magnitudes, and those voxels are filtered by
`min_neighbors` regardless of how the regularised solve came out.

## `unreliable_beyond_neighborhood_flag`, and why it is not a tear detector

The flag was originally named `near_transition_flag`, for the
six-to-four layer transition. That name claimed more than the diagnostic
can deliver.

**It cannot be based on the fit's own residual.** Erwin et al.'s tears
are ~500 um wide (p. 97: "The width of the slit is approximately
500 um"), which at 25 um/voxel is ~20 voxels — while the fit's
neighbourhood radius is typically 1 voxel, i.e. 25 um. A neighbourhood
that small sits entirely on one side of a tear and never sees the
discontinuity. Confirmed empirically: with radius 1 on the real atlas,
well under 0.01% of voxels exceeded a 1 deg residual.

**So it probes further out.** It extrapolates each voxel's own fitted
Jacobian `transition_probe_distance_vox` voxels along each physical axis
(default 20, i.e. 500 um, matching the reported tear width) and compares
against what the atlas actually holds there. The threshold sits well
above the in-neighbourhood reconstruction noise the paper reports (211 um,
SD 217 um, p. 98) but below the scale of an actual tear (~5 deg of
violation at one example location, p. 97).

**But it fires on 60.6% of fit-ok voxels**, rising monotonically with
eccentricity to near-saturation beyond 20 deg. That pattern tracks how
fast the retinotopic map curves — flagged voxels' median σ₁ is
18.1 deg/mm vs 4.27 deg/mm unflagged — not proximity to the documented
tears, which sit in a bounded inclination band rather than at high
eccentricity. A plain linear extrapolation cannot separate
tear-corruption from ordinary second-order curvature.

The flag is therefore documented for what it can honestly deliver: *do
not trust this voxel's Jacobian beyond its fit neighbourhood.* That is
exactly the question the rendering pipeline needs answered when choosing
between the single-Jacobian ellipse shortcut and per-voxel scatter.

`LAYERS.DAT`'s four codes (contra/ipsi × magno/parvo) do not preserve
which of the paper's six anatomical layers a voxel belonged to, so a
tear's exact location cannot be read off the released files anyway.

## Caching

The Jacobian is derived, not published, so it is cached in a
self-describing `.npz` rather than the raw headerless layout. On the
real atlas it is a few minutes of work and a ~109 MB compressed cache —
emphatically a build-time step. The on-disk key for the unreliability
flag is still `near_transition_flag`, so caches written before the
rename still load.
