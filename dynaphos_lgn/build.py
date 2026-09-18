"""One-call assembly of a configured LGN simulator.

`build_simulator(params)` walks the config in `config/params_lgn.yaml`
and wires up atlas -> Jacobian -> magnification -> electrodes ->
recruitment kernel -> simulator, caching the expensive Jacobian
computation to disk on the way. Every intermediate object is returned
alongside the simulator, so a script that wants to inspect or replace
one of them does not have to rebuild the rest.

Run as a script, this builds the **Jacobian cache** -- the one setup step
that is genuinely expensive and that everything else depends on::

    python -m dynaphos_lgn.build --cache-jacobian

It reads the four .DAT files, fits a local linear tissue ->
visual-field map at every valid voxel, writes ``JACOBIAN.npz`` beside
the atlas, and prints a validation report against Erwin et al. (1999)'s
published totals. Expect a few minutes, several GB of peak RAM and a
~110 MB cache. The report matters as much as the cache: a silently
mis-read atlas shows up there as a wrong total rather than as plausible
nonsense downstream.

There is one cache however many nuclei you simulate. The right LGN is
the left one reflected, and reflecting a Jacobian is re-indexing plus a
sign pattern -- so it is derived from this cache at build time rather
than fitted again.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from dynaphos.utils import load_params
from dynaphos_lgn.atlas import (ErwinAtlas, JacobianAtlas, MirroredAtlas,
                                MirroredJacobianAtlas)
from dynaphos_lgn.bilateral import (BilateralLGNSimulator,
                                    split_targets_by_hemifield)
from dynaphos_lgn.current_spread import RecruitmentKernel
from dynaphos_lgn.electrodes import LGNElectrodeArray
from dynaphos_lgn.magnification import (AnisotropicGradientMagnification,
                                        AtlasGradientMagnification,
                                        JacobianMagnification,
                                        MalpeliDensityMagnification)
from dynaphos_lgn.params import optional, require, resolve_class_codes
from dynaphos_lgn.simulator import LGNPhospheneSimulator


def load_lgn_params(path: Union[str, Path]) -> dict:
    """Load a params YAML (thin alias, kept so callers need one import)."""
    return load_params(str(path))


def atlas_file_specs(params: Mapping) -> dict:
    """{filename: bytes_per_voxel} from `atlas.files`.

    LAYERS.DAT is the odd one out at one byte, and a headerless read at
    the wrong width silently succeeds -- so `check_atlas_files` measures
    each file rather than trusting the config.
    """
    return {spec['name']: spec['bytes_per_voxel']
            for spec in require(params, 'atlas.files').values()}


def check_atlas_files(directory: Union[str, Path], params: Mapping) -> str:
    """Verify each .DAT file's size against the dtype it is read with.

    A headerless binary read at the wrong width does not fail -- it
    silently produces a differently-shaped, entirely wrong volume. This
    is the cheapest possible check that it has not happened.
    """
    directory = Path(directory)
    n_voxels = int(np.prod(require(params, 'atlas.shape_ml_dv_ap')))
    specs = atlas_file_specs(params)
    lines = ["Atlas file sizes",
             "-" * 68]
    ok = True
    for name, bytes_per_voxel in specs.items():
        path = directory / name
        if not path.exists():
            lines.append(f"  {name:<12} MISSING")
            ok = False
            continue
        size = path.stat().st_size
        expected = n_voxels * bytes_per_voxel
        verdict = 'ok' if size == expected else 'MISMATCH'
        if size != expected:
            ok = False
            implied = size / n_voxels
            verdict = (f"MISMATCH -- implies {implied:.3g} bytes/voxel, "
                       f"not {bytes_per_voxel}")
        lines.append(f"  {name:<12} {size:>12,} bytes "
                     f"(expected {expected:,})  {verdict}")
    if not ok:
        lines.append("")
        lines.append("  A size mismatch means the file is being read at the "
                     "wrong width. Everything downstream would be wrong "
                     "without erroring, so fix this before going further.")
    return '\n'.join(lines)


def build_atlas(params: dict, atlas_dir: Optional[Union[str, Path]] = None,
                synthetic: bool = False,
                fallback_to_synthetic: bool = True):
    """Load the real atlas, or build a synthetic stand-in.

    Falls back to the synthetic atlas -- loudly -- when the real files
    are absent, so a fresh checkout can still run the demo. Anything
    produced that way is a code check, not a result.

    :param fallback_to_synthetic: Set False where a synthetic result
        would be worse than no result, such as building the Jacobian
        cache -- falling back would write a file that looks real.
    """
    if synthetic:
        from dynaphos_lgn.synthetic_atlas import SyntheticAtlas
        return SyntheticAtlas(params)

    directory = Path(atlas_dir or require(params, 'atlas.directory'))
    specs = atlas_file_specs(params)
    missing = [f for f in specs if not (directory / f).exists()]
    if missing:
        if not fallback_to_synthetic:
            raise FileNotFoundError(
                f"Atlas files {missing} not found in {directory}. Download "
                f"the four .DAT files from malpeli.psychology.illinois.edu/"
                f"atlas and put them there, or point --atlas-dir at them.")
        logging.warning(
            "Atlas files %s not found in %s -- falling back to the SYNTHETIC "
            "atlas. Output will exercise the code but means nothing "
            "biologically.", missing, directory)
        from dynaphos_lgn.synthetic_atlas import SyntheticAtlas
        return SyntheticAtlas(params)

    return ErwinAtlas(params).build_atlas(directory)


def build_jacobian(params: dict, atlas, cache_dir=None,
                   recompute: bool = False) -> JacobianAtlas:
    """Load the cached per-voxel Jacobian, or compute and cache it.

    On the real atlas this is a few minutes of work and a ~109 MB
    compressed cache, so it is emphatically a build-time step, not
    something to redo per run.
    """
    # Every fit parameter comes from `atlas.jacobian`; JacobianAtlas
    # reads them itself, so nothing is duplicated here.
    if getattr(atlas, 'is_synthetic', False) or cache_dir is None:
        return JacobianAtlas(params).compute(atlas)

    cache_dir = Path(cache_dir)
    path = cache_dir / require(params, 'atlas.jacobian_cache')
    if path.exists() and not recompute:
        logging.info("Loading cached Jacobian atlas from %s.", path)
        return JacobianAtlas.load(path, params)
    logging.info("Computing the Jacobian atlas (this takes a while); "
                 "caching to %s.", path)
    jac = JacobianAtlas(params).compute(atlas)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jac.save(path)
    return jac


# ======================================================================
# Hemispheres
# ======================================================================
# The published atlas is one LEFT LGN, which represents the RIGHT
# hemifield. The right nucleus is that same volume reflected, not a
# second dataset -- `dynaphos_lgn.atlas.MirroredAtlas` says what that
# assumes. Both nuclei share the left one's atlas load and its cached
# Jacobian fit; reflecting is re-indexing, and refitting would spend
# another ~90 s and ~4.2 GB reproducing numbers that follow exactly.


def configured_hemispheres(params: Mapping) -> list:
    """The nuclei `atlas.hemispheres` asks for, validated and ordered.

    Left first when both are present, so electrode numbering is
    predictable.
    """
    names = require(params, 'atlas.hemispheres')
    if isinstance(names, str):
        names = [names]
    names = list(names)
    if not names:
        raise ValueError("atlas.hemispheres is empty; it must list at least "
                         "one of 'left', 'right'.")
    unknown = [n for n in names if n not in ('left', 'right')]
    if unknown:
        raise ValueError(f"Unknown hemisphere(s) {unknown} in "
                         f"atlas.hemispheres; expected 'left' and/or "
                         f"'right'.")
    if len(set(names)) != len(names):
        raise ValueError(f"atlas.hemispheres lists a nucleus twice: {names}.")
    return sorted(names, key=lambda n: 0 if n == 'left' else 1)


def n_electrodes_for(params: Mapping, hemisphere: str) -> int:
    """How many electrodes go in one nucleus.

    `electrodes.n_electrodes` is the per-nucleus count;
    `electrodes.n_electrodes_by_hemisphere` overrides it for one side,
    for an implant that is not symmetric. A null there means "use the
    shared value", which is why this reads through `optional`.
    """
    override = optional(
        params, f'electrodes.n_electrodes_by_hemisphere.{hemisphere}')
    if override is not None:
        return int(override)
    return int(require(params, 'electrodes.n_electrodes'))


def build_hemisphere_atlas(params: Mapping, atlas, hemisphere: str):
    """The left atlas as loaded, or its mirror image for the right."""
    if hemisphere == 'left':
        return atlas
    if hemisphere == 'right':
        return MirroredAtlas(atlas, params)
    raise ValueError(f"Unknown hemisphere {hemisphere!r}; expected 'left' "
                     f"or 'right'.")


def build_hemisphere_jacobian(params: Mapping, jacobian_atlas,
                              hemisphere: str):
    """The cached fit, or the reflection of it. Never a second fit."""
    if hemisphere == 'left':
        return jacobian_atlas
    if hemisphere == 'right':
        return MirroredJacobianAtlas(jacobian_atlas, params)
    raise ValueError(f"Unknown hemisphere {hemisphere!r}; expected 'left' "
                     f"or 'right'.")


def build_kernel(params: dict) -> RecruitmentKernel:
    """The recruitment kernel the config describes."""
    return RecruitmentKernel.from_params(params)


def build_magnification_model(params: dict, atlas, jacobian_atlas):
    """Whichever magnification model the config asks for."""
    choice = require(params, 'magnification.model')
    if choice == 'jacobian':
        return JacobianMagnification(jacobian_atlas, params, atlas)
    if choice == 'atlas_gradient':
        return AtlasGradientMagnification.from_jacobian(
            JacobianMagnification(jacobian_atlas, params, atlas))
    if choice == 'anisotropic_gradient':
        return AnisotropicGradientMagnification.from_jacobian(
            JacobianMagnification(jacobian_atlas, params, atlas))
    if choice == 'malpeli_density':
        return MalpeliDensityMagnification.from_atlas(atlas, params)
    raise ValueError(
        f"Unknown magnification.model {choice!r}; expected 'jacobian', "
        f"'atlas_gradient', 'anisotropic_gradient' or 'malpeli_density'.")


def build_electrode_array(params: dict, atlas, jacobian_atlas,
                          kernel: RecruitmentKernel,
                          rng: Optional[np.random.Generator] = None,
                          voxel_indices: Optional[np.ndarray] = None,
                          visual_field_targets=None,
                          n_electrodes: Optional[int] = None
                          ) -> LGNElectrodeArray:
    # Everything else (I_max, tail weight, budgets, scatter points, cell
    # class, sentinel handling) is read from the config by
    # LGNElectrodeArray itself, so there is exactly one place each value
    # lives.
    common = dict(kernel=kernel, jacobian_atlas=jacobian_atlas)

    if voxel_indices is not None:
        return LGNElectrodeArray(atlas, params, voxel_indices, rng=rng,
                                 **common)
    if visual_field_targets is not None:
        ecc, incl = visual_field_targets
        return LGNElectrodeArray.from_visual_field_targets(
            atlas, params, ecc, incl, **common)
    return LGNElectrodeArray.spread_in_visual_field(
        atlas, params, rng=rng, n_electrodes=n_electrodes, **common)


def build_simulator(params: dict, atlas_dir=None, synthetic: bool = False,
                    rng: Optional[np.random.Generator] = None,
                    voxel_indices: Optional[np.ndarray] = None,
                    visual_field_targets=None,
                    recompute_jacobian: bool = False,
                    hemispheres: Optional[Sequence[str]] = None
                    ) -> Tuple[Union[LGNPhospheneSimulator,
                                     BilateralLGNSimulator], dict]:
    """Assemble a simulator from a parameter dict.

    How much of the visual field it covers is `atlas.hemispheres`'s
    decision. One nucleus gives one hemifield and an
    `LGNPhospheneSimulator`, exactly as before this key existed; both
    give the whole field and a `BilateralLGNSimulator` that sums them.

    :param hemispheres: Override `atlas.hemispheres` for this build.
    :param voxel_indices: Explicit electrode voxels. With two nuclei the
        indices are per-nucleus and so ambiguous on their own -- pass
        ``{'left': ..., 'right': ...}`` instead.
    :param visual_field_targets: ``(eccentricity, inclination)`` percept
        locations. With two nuclei each target is routed to the nucleus
        whose hemifield it falls in.
    :return: (simulator, parts). For one hemisphere `parts` holds the
        atlas, Jacobian atlas, kernel, magnification model and electrode
        array as it always has. For two it holds the shared `kernel` and
        a `hemispheres` dict of that same set per nucleus, plus
        `simulators`.
    """
    rng = (np.random.default_rng(require(params, 'run.seed'))
           if rng is None else rng)
    names = (configured_hemispheres(params) if hemispheres is None
             else configured_hemispheres({'atlas': {
                 'hemispheres': list(hemispheres)}}))

    atlas = build_atlas(params, atlas_dir, synthetic)
    cache_dir = (None if getattr(atlas, 'is_synthetic', False)
                 else Path(atlas_dir or require(params, 'atlas.directory')))
    jacobian = build_jacobian(params, atlas, cache_dir, recompute_jacobian)
    kernel = build_kernel(params)

    voxels_by_hemisphere = _voxel_indices_by_hemisphere(voxel_indices, names)
    targets_by_hemisphere = _targets_by_hemisphere(visual_field_targets,
                                                   names)
    # One child generator per nucleus, drawn from the caller's, so the
    # two arrays are independent draws rather than exact reflections of
    # each other -- and still reproducible from `run.seed`.
    generators = ([rng] if len(names) == 1
                  else [np.random.default_rng(int(seed)) for seed
                        in rng.integers(0, 2 ** 62, len(names))])

    parts = {'kernel': kernel, 'hemispheres': {}}
    simulators = {}
    for name, hemisphere_rng in zip(names, generators):
        hemisphere_atlas = build_hemisphere_atlas(params, atlas, name)
        hemisphere_jacobian = build_hemisphere_jacobian(params, jacobian,
                                                        name)
        magnification = build_magnification_model(
            params, hemisphere_atlas, hemisphere_jacobian)
        array = build_electrode_array(
            params, hemisphere_atlas, hemisphere_jacobian, kernel,
            hemisphere_rng, voxels_by_hemisphere[name],
            targets_by_hemisphere[name], n_electrodes_for(params, name))
        array.magnification_model = magnification
        simulators[name] = LGNPhospheneSimulator(params, array,
                                                 rng=hemisphere_rng)
        parts['hemispheres'][name] = {
            'atlas': hemisphere_atlas,
            'jacobian_atlas': hemisphere_jacobian,
            'kernel': kernel,
            'magnification': magnification,
            'electrode_array': array,
            'simulator': simulators[name]}

    if len(names) == 1:
        only = parts['hemispheres'][names[0]]
        return simulators[names[0]], {k: v for k, v in only.items()
                                      if k != 'simulator'}

    parts['simulators'] = simulators
    return BilateralLGNSimulator(simulators, params), parts


def _voxel_indices_by_hemisphere(voxel_indices, names: Sequence[str]) -> dict:
    """Resolve the `voxel_indices` argument to one entry per nucleus."""
    if voxel_indices is None:
        return {name: None for name in names}
    if isinstance(voxel_indices, Mapping):
        missing = [name for name in names if name not in voxel_indices]
        if missing:
            raise ValueError(
                f"voxel_indices has no entry for {missing}; it must name "
                f"every hemisphere being built ({list(names)}).")
        return {name: np.asarray(voxel_indices[name]) for name in names}
    if len(names) > 1:
        raise ValueError(
            f"voxel_indices are indices into ONE nucleus's grid, so a plain "
            f"array is ambiguous when building {list(names)}. Pass "
            f"{{'left': ..., 'right': ...}}, or build one hemisphere at a "
            f"time with hemispheres=.")
    return {names[0]: np.asarray(voxel_indices)}


def _targets_by_hemisphere(visual_field_targets, names: Sequence[str]) -> dict:
    """Route requested percept locations to the nucleus that can reach."""
    if visual_field_targets is None:
        return {name: None for name in names}
    if len(names) == 1:
        return {names[0]: visual_field_targets}
    eccentricity, inclination = visual_field_targets
    return split_targets_by_hemifield(eccentricity, inclination, names)


# ======================================================================
# Validation report
# ======================================================================
# Erwin, Baker, Busen & Malpeli (1999)'s own published totals live in
# `validation.erwin_1999`. Reproducing them from the raw .DAT files
# confirms the grid layout, the Fortran ordering, the scale factors and
# the layer-code assignment all at once -- a subtly wrong read produces
# plausible-looking output everywhere else. They sit in the config with
# everything else, but note what that means: editing them edits the
# CHECK, not the model.


def atlas_report(atlas, params: Mapping) -> str:
    """Check a loaded atlas against the paper's published totals."""
    if getattr(atlas, 'is_synthetic', False):
        return ("Atlas check: SYNTHETIC atlas -- there is nothing to check "
                "it against, and nothing it says means anything "
                "biologically.")

    expected = require(params, 'validation.erwin_1999')
    max_ecc = require(params, 'atlas.max_eccentricity_deg')
    n_voxels = int(np.prod(atlas.atlas_shape))
    magno = np.isin(atlas.layer, resolve_class_codes(params, 'magno'))
    parvo = np.isin(atlas.layer, resolve_class_codes(params, 'parvo'))
    magno_cells = float(atlas.cells_per_voxel[magno].sum())
    parvo_cells = float(atlas.cells_per_voxel[parvo].sum())
    tissue = atlas.layer != atlas._missing_layer
    volume = float(tissue.sum()) * atlas._voxel_size_mm ** 3
    density = atlas.cells_per_voxel[tissue]

    def line(label, got, expected, unit='', fmt='{:,.0f}'):
        delta = (got - expected) / expected * 100 if expected else float('nan')
        return (f"  {label:<34}{fmt.format(got)}{unit}"
                f"   (paper {fmt.format(expected)}{unit}, "
                f"{delta:+.2f}%)")

    lines = [
        "Atlas read-back check, against Erwin et al. (1999)'s own totals",
        "-" * 68,
        f"  grid                              {atlas.atlas_shape} "
        f"= {n_voxels:,} voxels at "
        f"{atlas._voxel_size_mm * 1e3:.0f} um",
        f"  LAYERS == 0 (extralaminar)        "
        f"{int((atlas.layer == 0).sum()):,}",
        f"  ECC == 999                        "
        f"{int((atlas.eccentricity == 999).sum()):,}",
        f"  INCL == 999                       "
        f"{int((atlas.inclination == 999).sum()):,}",
        f"  valid (all three gates)           "
        f"{int(atlas.valid.sum()):,} "
        f"({atlas.valid.mean() * 100:.1f}% of the grid)",
        "",
        line('magnocellular cells', magno_cells,
             expected['magno_cells']),
        line('parvocellular cells', parvo_cells,
             expected['parvo_cells']),
        line('reconstructed volume', volume, expected['volume_mm3'],
             ' mm^3', '{:,.2f}'),
        line('mean density', float(density.mean()),
             expected['mean_density_per_voxel'], ' cells/voxel',
             '{:.3f}'),
        line('max density', float(density.max()),
             expected['max_density_per_voxel'], ' cells/voxel',
             '{:.3f}'),
        line('max eccentricity, parvo',
             float(atlas.eccentricity_deg[parvo & atlas.valid].max()),
             max_ecc['parvo'], ' deg', '{:.1f}'),
        line('max eccentricity, magno',
             float(atlas.eccentricity_deg[magno & atlas.valid].max()),
             max_ecc['magno'], ' deg', '{:.1f}'),
        "",
        "  Accuracy floor on everything downstream: mean "
        f"{expected['reconstruction_accuracy_um'][0]} um "
        f"(SD {expected['reconstruction_accuracy_um'][1]} um) between "
        "recorded",
        "  sites and the reconstructed projection column. No choice of "
        "processing improves on that.",
    ]
    return '\n'.join(lines)


def jacobian_report(jacobian_atlas, atlas, params: Mapping) -> str:
    """Validation summary for a computed or loaded Jacobian atlas.

    Reports fit coverage, fit quality, the two confidence flags, and the
    magnification/anisotropy field binned by eccentricity. Read the
    anisotropy column against the Connolly & Van Essen checkpoint -- a
    checkpoint, not a target.

    `validation.jacobian_report_sample_size` fixes how many voxels are
    drawn for the magnification statistics. Counts, flag rates and
    residual statistics are computed exactly over every voxel -- they are
    cheap array reductions. The per-bin magnification medians are not:
    they need an SVD per voxel, which over millions of them dominates the
    runtime of a command whose actual job is already done. A median over
    a few hundred thousand randomly drawn voxels is stable to far better
    than the two decimal places printed here, so the report samples and
    says so rather than quietly taking a minute.
    """
    ecc_edges = require(params, 'validation.report_eccentricity_edges')
    sample_size = require(params, 'validation.jacobian_report_sample_size')
    seed = require(params, 'validation.jacobian_report_seed')

    valid_in = atlas.valid
    fit_ok = jacobian_atlas.jacobian_valid
    sentinel = atlas.is_ipsi_sentinel() & valid_in

    from dynaphos_lgn.magnification import JacobianMagnification
    rng = np.random.default_rng(seed)
    fit_index = np.flatnonzero(fit_ok.ravel())
    sampled = (rng.choice(fit_index, sample_size, replace=False)
               if fit_index.size > sample_size else fit_index)
    sample_major, sample_minor, _ = JacobianMagnification.decompose(
        jacobian_atlas.jacobian.reshape(-1, 2, 3)[sampled])
    sample_ecc = atlas.eccentricity_deg.ravel()[sampled]

    residual = jacobian_atlas.residual_rms[fit_ok]
    residual = residual[np.isfinite(residual)]
    neighbours = jacobian_atlas.n_neighbors[fit_ok]
    isotropic = jacobian_atlas.isotropic_flag
    unreliable = jacobian_atlas.unreliable_beyond_neighborhood_flag

    lines = [
        "Jacobian atlas validation",
        "-" * 68,
        f"  valid input voxels                {int(valid_in.sum()):,}",
        f"  of those, ipsi +-135 sentinel     {int(sentinel.sum()):,} "
        f"({sentinel.sum() / max(valid_in.sum(), 1) * 100:.1f}%) -- excluded "
        f"as a different part of the visual field, not missing data",
        f"  voxels with a usable fit          {int(fit_ok.sum()):,} "
        f"({fit_ok.sum() / max(valid_in.sum(), 1) * 100:.1f}% of valid "
        f"input)",
        f"  neighbours per fit (median/min)   "
        f"{np.median(neighbours):.0f} / {neighbours.min():.0f} "
        f"(26 is the maximum possible at radius 1)",
        "",
        "  Fit quality (in-neighbourhood residual, degrees):",
        f"    median {np.median(residual):.3f}   mean "
        f"{residual.mean():.3f}   p99 {np.percentile(residual, 99):.3f}   "
        f"above 1 deg: "
        f"{(residual > 1).mean() * 100:.4f}%",
        "",
        f"  isotropic-by-construction (central "
        f"{jacobian_atlas.ISOTROPIC_CONSTRUCTION_RADIUS_DEG:.0f} deg): "
        f"{int(isotropic.sum()):,} "
        f"({isotropic[fit_ok].mean() * 100:.1f}% of fit-ok)",
        f"  unreliable beyond neighbourhood:  {int(unreliable.sum()):,} "
        f"({unreliable[fit_ok].mean() * 100:.1f}% of fit-ok)",
        "",
        f"  By eccentricity (counts and flag rates exact; magnification "
        f"medians over a {len(sampled):,}-voxel sample):",
        "    ecc (deg)      n      sigma1   sigma2   aniso   unreliable",
    ]

    ecc = atlas.eccentricity_deg
    flag_field = unreliable
    for lo, hi in zip(ecc_edges[:-1], ecc_edges[1:]):
        sel = fit_ok & (ecc >= lo) & (ecc < hi)
        n = int(sel.sum())
        if n == 0:
            lines.append(f"    {lo:>3.0f}-{hi:<3.0f}    {n:>10,}   (empty)")
            continue
        flag = flag_field[sel].mean()
        in_bin = (sample_ecc >= lo) & (sample_ecc < hi)
        if not in_bin.any():
            lines.append(
                f"    {lo:>3.0f}-{hi:<3.0f}    {n:>10,}   "
                f"(no sampled voxels)   {flag * 100:>6.1f}%")
            continue
        a = float(np.nanmedian(sample_major[in_bin]))
        b = float(np.nanmedian(sample_minor[in_bin]))
        lines.append(
            f"    {lo:>3.0f}-{hi:<3.0f}    {n:>10,}   {a:>6.2f}   {b:>6.2f}"
            f"   {a / b if b else float('nan'):>5.2f}   {flag * 100:>6.1f}%")

    lines += [
        "",
        "    sigma1/sigma2 are the two principal magnifications in deg/mm;",
        "    aniso is their ratio. Connolly & Van Essen report 2-3x near the",
        "    horizontal meridian (parvocellular layer 6), reversed near the",
        "    vertical meridian (magnocellular layer 1) -- QUALITATIVE, and a",
        "    checkpoint rather than a target. Rows inside the central 1 deg",
        "    are isotropic by construction: Erwin et al. assigned",
        "    eccentricity there from Malpeli's own isotropic formula, so any",
        "    anisotropy found there is an artefact of how the atlas was",
        "    built, not a measurement.",
        "",
        "    The 'unreliable' column rising with eccentricity is expected",
        "    and correct: deg/mm rises toward the periphery, so a fixed",
        "    physical current-spread radius covers progressively more",
        "    visual field -- exactly where trusting one point-evaluated",
        "    Jacobian across a whole footprint is riskiest. It does NOT",
        "    mark the six-to-four layer transition; a plain linear",
        "    extrapolation cannot separate a tear from ordinary fast",
        "    curvature, and this flag does not claim to.",
    ]
    return '\n'.join(lines)


def cache_jacobian(params: dict, atlas_dir=None, recompute: bool = False,
                   report_path=None, allow_synthetic: bool = False) -> dict:
    """Build (or load) the Jacobian cache and print a validation report.

    :return: dict with the atlas, the Jacobian atlas, the cache path and
        the report text.
    """
    directory = Path(atlas_dir or require(params, 'atlas.directory'))
    atlas = build_atlas(params, directory, synthetic=allow_synthetic,
                        fallback_to_synthetic=allow_synthetic)

    started = time.time()
    jacobian = build_jacobian(params, atlas, directory, recompute)
    elapsed = time.time() - started

    path = directory / require(params, 'atlas.jacobian_cache')
    size_mb = path.stat().st_size / 1e6 if path.exists() else float('nan')

    report = '\n\n'.join([
        check_atlas_files(directory, params),
        atlas_report(atlas, params),
        jacobian_report(jacobian, atlas, params),
        (f"Cache\n{'-' * 68}\n  file        {path}\n"
         f"  size        {size_mb:,.1f} MB\n"
         f"  wall time   {elapsed:,.1f} s"),
    ])

    if report_path is not None:
        Path(report_path).write_text(report, encoding='utf-8')

    return {'atlas': atlas, 'jacobian_atlas': jacobian, 'path': path,
            'report': report, 'fit_seconds': elapsed}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m dynaphos_lgn.build',
        description='Build the LGN simulator, or just its Jacobian cache.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical use, once per checkout:\n"
            "  python -m dynaphos_lgn.build --cache-jacobian\n\n"
            "Measured on a full-size (240x280x320) rehearsal volume:\n"
            "  ~90 s wall time, ~4.2 GB peak RSS.\n"
            "The real atlas compresses less well than that rehearsal did,\n"
            "so expect JACOBIAN.npz to land near 110 MB.\n\n"
            "The 4 GB is the part to watch -- it is nearly all full-grid\n"
            "float32 arrays, so there is no way to trade time for memory\n"
            "here. Close other things first if RAM is tight.\n\n"
            "Everything downstream loads the cache instead of recomputing.\n"
            "One cache covers both nuclei: the right LGN is derived from\n"
            "this fit by reflection, never refitted."))
    parser.add_argument('--params', default=None,
                        help='params_lgn.yaml (default: config/ beside the '
                             'repo root)')
    parser.add_argument('--atlas-dir', default=None,
                        help='directory holding ECC/INCL/LAYERS/CELLS.DAT '
                             '(default: the path in the config)')
    parser.add_argument('--cache-jacobian', action='store_true',
                        help='compute and cache the Jacobian atlas, then '
                             'print a validation report')
    parser.add_argument('--recompute', action='store_true',
                        help='recompute even if a cache already exists')
    parser.add_argument('--report', default=None,
                        help='also write the validation report to this file')
    parser.add_argument('--allow-synthetic', action='store_true',
                        help='fall back to the synthetic atlas if the real '
                             'files are missing. Off by default: a synthetic '
                             'cache would look exactly like a real one and '
                             'be worthless')
    parser.add_argument('--check-only', action='store_true',
                        help='read the atlas and report on it, without '
                             'computing or writing anything')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format='%(levelname)s %(message)s')

    params_path = Path(args.params) if args.params else (
        Path(__file__).resolve().parents[1] / 'config' / 'params_lgn.yaml')
    if not params_path.exists():
        parser.error(f"No params file at {params_path}; pass --params.")
    params = load_lgn_params(params_path)

    if args.check_only:
        directory = Path(args.atlas_dir or require(params,
                                                   'atlas.directory'))
        text = check_atlas_files(directory, params)
        print(text)
        print()
        try:
            atlas = build_atlas(params, directory,
                                synthetic=args.allow_synthetic,
                                fallback_to_synthetic=args.allow_synthetic)
        except FileNotFoundError as error:
            print(f"error: {error}")
            return 2
        text = atlas_report(atlas, params)
        print(text)
        if args.report:
            Path(args.report).write_text(text, encoding='utf-8')
        return 0

    if not args.cache_jacobian:
        parser.error("Nothing to do. Pass --cache-jacobian (the usual case) "
                     "or --check-only.")

    directory = Path(args.atlas_dir or require(params, 'atlas.directory'))
    if not args.quiet:
        print(f"Building the Jacobian cache from {directory}.\n"
              f"Expect roughly 90 s and about 4.2 GB of peak RAM at the "
              f"atlas's full 21.5M voxels;\n"
              f"the result is written once and loaded thereafter.\n",
              flush=True)

    try:
        result = cache_jacobian(params, args.atlas_dir, args.recompute,
                                args.report, args.allow_synthetic)
    except FileNotFoundError as error:
        # A clean message, not a traceback: a missing download is the
        # single most likely way this command fails, and it is the user's
        # to fix, not a bug to report.
        print(f"error: {error}")
        return 2
    except MemoryError:
        print("error: ran out of memory fitting the Jacobian. The fit "
              "holds several full-grid float32 arrays at once (about "
              "4.2 GB peak at the atlas's 21.5M voxels) and cannot trade "
              "time for memory. Close other applications and retry.")
        return 3
    print(result['report'])
    if args.report:
        print(f"\nReport also written to {args.report}.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
