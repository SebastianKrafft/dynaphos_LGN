"""Assemble a configured LGN simulator.

`build_simulator(params)` wires up atlas -> Jacobian -> electrodes ->
kernel -> simulator and returns the intermediate objects alongside the
simulator.

Run as a script to build the Jacobian cache and print a validation
report against Erwin et al. (1999)::

    python -m dynaphos_lgn.build --cache-jacobian
"""
import argparse
import logging
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from dynaphos.utils import load_params
from dynaphos_lgn.atlas import (ErwinAtlas, JacobianAtlas, MirroredAtlas,
                                MirroredJacobianAtlas, hemifield_of)
from dynaphos_lgn.bilateral import (BilateralLGNSimulator,
                                    split_targets_by_hemifield)
from dynaphos_lgn.current_spread import RecruitmentKernel
from dynaphos_lgn.electrodes import LGNElectrodeArray
from dynaphos_lgn.params import PARAMS_PATH
from dynaphos_lgn.report import (atlas_file_specs, atlas_report,
                                 check_atlas_files, jacobian_report)
from dynaphos_lgn.simulator import LGNPhospheneSimulator
from dynaphos_lgn.synthetic_atlas import SyntheticAtlas


def build_atlas(params: dict, atlas_dir: Optional[Union[str, Path]] = None,
                synthetic: bool = False,
                fallback_to_synthetic: bool = True):
    """Load the real atlas, or the synthetic stand-in.

    :param fallback_to_synthetic: Use the synthetic atlas, with a
        warning, when the real files are missing.
    """
    if synthetic:
        return SyntheticAtlas(params)

    directory = Path(atlas_dir or params['atlas']['directory'])
    missing = [name for name in atlas_file_specs(params)
               if not (directory / name).exists()]
    if not missing:
        return ErwinAtlas(params).read(directory)
    if not fallback_to_synthetic:
        raise FileNotFoundError(
            f"Atlas files {missing} not found in {directory}. Download the "
            f"four .DAT files from malpeli.psychology.illinois.edu/atlas "
            f"and put them there, or point --atlas-dir at them.")
    logging.warning("Atlas files %s not found in %s; using the synthetic "
                    "atlas.", missing, directory)
    return SyntheticAtlas(params)


def jacobian_cache_name(params: Mapping, atlas=None) -> str:
    """The cache file; a rebuilt center has its own, since it changes the
    fitted values.

    :param atlas: The loaded atlas, whose own `central_eccentricity`
        then wins over the config's.
    """
    name = Path(params['atlas']['jacobian_cache'])
    central = (params['atlas']['central_eccentricity'] if atlas is None
               else atlas.central_eccentricity)
    if central == 'rebuilt':
        name = name.with_stem(name.stem + '_CENTRAL_REBUILT')
    return str(name)


def build_jacobian(params: dict, atlas, cache_dir=None,
                   recompute: bool = False) -> JacobianAtlas:
    """Load the cached per-voxel Jacobian, or compute and cache it."""
    if atlas.is_synthetic or cache_dir is None:
        return JacobianAtlas(params).compute(atlas)

    cache_dir = Path(cache_dir)
    path = cache_dir / jacobian_cache_name(params, atlas)
    if path.exists() and not recompute:
        logging.info("Loading cached Jacobian atlas from %s.", path)
        return JacobianAtlas.load(path, params)
    logging.info("Computing the Jacobian atlas (this takes a while); "
                 "caching to %s.", path)
    jac = JacobianAtlas(params).compute(atlas)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jac.save(path)
    return jac


def configured_hemispheres(params: Mapping) -> list:
    """The nuclei `atlas.hemispheres` asks for, validated, left first."""
    names = params['atlas']['hemispheres']
    names = [names] if isinstance(names, str) else list(names)
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
    """`electrodes.n_electrodes_by_hemisphere` for this nucleus, or
    `electrodes.n_electrodes` when that is null."""
    override = params['electrodes']['n_electrodes_by_hemisphere'][hemisphere]
    if override is not None:
        return int(override)
    return int(params['electrodes']['n_electrodes'])


def build_hemisphere_atlas(params: Mapping, atlas, hemisphere: str):
    """The left atlas as loaded, or its mirror image for the right."""
    hemifield_of(hemisphere)  # validates the name
    return atlas if hemisphere == 'left' else MirroredAtlas(atlas, params)


def build_hemisphere_jacobian(params: Mapping, jacobian_atlas,
                              hemisphere: str):
    """The cached fit, or the reflection of it. Never a second fit."""
    hemifield_of(hemisphere)  # validates the name
    return (jacobian_atlas if hemisphere == 'left'
            else MirroredJacobianAtlas(jacobian_atlas, params))


def build_electrode_array(params: dict, atlas, jacobian_atlas,
                          kernel: RecruitmentKernel,
                          rng: Optional[np.random.Generator] = None,
                          voxel_indices: Optional[np.ndarray] = None,
                          visual_field_targets=None,
                          n_electrodes: Optional[int] = None
                          ) -> LGNElectrodeArray:
    """Electrodes at `voxel_indices`, at the voxels nearest
    `visual_field_targets`, or spread in the visual field."""
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

    One nucleus in `atlas.hemispheres` gives an `LGNPhospheneSimulator`,
    two give a `BilateralLGNSimulator`; both expose the same
    per-electrode views.

    :param hemispheres: Override `atlas.hemispheres` for this build.
    :param voxel_indices: Explicit electrode voxels; with two nuclei a
        ``{'left': ..., 'right': ...}`` dict, in which None spreads that
        nucleus's electrodes in the visual field.
    :param visual_field_targets: ``(eccentricity, inclination)`` percept
        locations, routed to the nucleus whose hemifield they fall in.
    :return: (simulator, parts). `parts['hemispheres']` maps each nucleus
        to its atlas, Jacobian atlas, kernel, magnification, electrode
        array and simulator. With one nucleus those are also at the top
        level of `parts`.
    """
    rng = (np.random.default_rng(params['run']['seed'])
           if rng is None else rng)
    names = (configured_hemispheres(params) if hemispheres is None
             else configured_hemispheres({'atlas': {
                 'hemispheres': hemispheres}}))

    atlas = build_atlas(params, atlas_dir, synthetic)
    cache_dir = (None if atlas.is_synthetic
                 else Path(atlas_dir or params['atlas']['directory']))
    jacobian = build_jacobian(params, atlas, cache_dir, recompute_jacobian)
    kernel = RecruitmentKernel.from_params(params)

    voxels_by_hemisphere = _voxel_indices_by_hemisphere(voxel_indices, names)
    targets_by_hemisphere = _targets_by_hemisphere(visual_field_targets,
                                                   names)
    # One child generator per nucleus, so the two arrays are independent
    # but reproducible draws.
    generators = ([rng] if len(names) == 1
                  else [np.random.default_rng(int(seed)) for seed
                        in rng.integers(0, 2 ** 62, len(names))])

    parts = {'kernel': kernel, 'hemispheres': {}}
    simulators = {}
    for name, hemisphere_rng in zip(names, generators):
        hemisphere_atlas = build_hemisphere_atlas(params, atlas, name)
        hemisphere_jacobian = build_hemisphere_jacobian(params, jacobian,
                                                        name)
        array = build_electrode_array(
            params, hemisphere_atlas, hemisphere_jacobian, kernel,
            hemisphere_rng, voxels_by_hemisphere[name],
            targets_by_hemisphere[name], n_electrodes_for(params, name))
        simulators[name] = LGNPhospheneSimulator(params, array,
                                                 rng=hemisphere_rng)
        parts['hemispheres'][name] = {
            'atlas': hemisphere_atlas,
            'jacobian_atlas': hemisphere_jacobian,
            'kernel': kernel,
            'magnification': array.magnification_model,
            'electrode_array': array,
            'simulator': simulators[name]}

    if len(names) == 1:
        only = parts['hemispheres'][names[0]]
        return simulators[names[0]], {**only, **parts}

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
        return {name: (None if voxel_indices[name] is None
                       else np.asarray(voxel_indices[name]))
                for name in names}
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


def cache_jacobian(params: dict, atlas_dir=None, recompute: bool = False,
                   report_path=None, allow_synthetic: bool = False) -> dict:
    """Build (or load) the Jacobian cache and write a validation report.

    :return: dict with the atlas, the Jacobian atlas, the cache path and
        the report text.
    """
    directory = Path(atlas_dir or params['atlas']['directory'])
    atlas = build_atlas(params, directory, synthetic=allow_synthetic,
                        fallback_to_synthetic=allow_synthetic)

    started = time.time()
    jacobian = build_jacobian(params, atlas, directory, recompute)
    elapsed = time.time() - started

    path = directory / jacobian_cache_name(params, atlas)
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
        description='Build the Jacobian cache of the LGN atlas, or check '
                    'the atlas files.')
    parser.add_argument('--params', default=str(PARAMS_PATH))
    parser.add_argument('--atlas-dir', default=None,
                        help='directory holding ECC/INCL/LAYERS/CELLS.DAT '
                             '(default: atlas.directory in the config)')
    parser.add_argument('--cache-jacobian', action='store_true',
                        help='compute and cache the Jacobian atlas, then '
                             'print a validation report')
    parser.add_argument('--recompute', action='store_true',
                        help='recompute even if a cache already exists')
    parser.add_argument('--report', default=None,
                        help='also write the validation report to this file')
    parser.add_argument('--allow-synthetic', action='store_true',
                        help='fall back to the synthetic atlas if the real '
                             'files are missing')
    parser.add_argument('--check-only', action='store_true',
                        help='read the atlas and report on it, without '
                             'computing or writing anything')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    if not (args.cache_jacobian or args.check_only):
        parser.error("Pass --cache-jacobian or --check-only.")

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format='%(levelname)s %(message)s')
    params = load_params(args.params)
    directory = Path(args.atlas_dir or params['atlas']['directory'])

    try:
        if args.check_only:
            atlas = build_atlas(params, directory,
                                synthetic=args.allow_synthetic,
                                fallback_to_synthetic=args.allow_synthetic)
            report = '\n\n'.join([check_atlas_files(directory, params),
                                  atlas_report(atlas, params)])
            if args.report:
                Path(args.report).write_text(report, encoding='utf-8')
        else:
            report = cache_jacobian(params, directory, args.recompute,
                                    args.report, args.allow_synthetic
                                    )['report']
    except FileNotFoundError as error:
        print(f"error: {error}")
        return 2
    print(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
