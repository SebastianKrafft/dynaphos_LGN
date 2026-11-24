"""Text reports on a loaded atlas and its Jacobian fit, checked against
the totals Erwin et al. (1999) published (`validation.erwin_1999`)."""
from pathlib import Path
from typing import Mapping, Union

import numpy as np

from dynaphos_lgn.fovea import foveal_reference_report
from dynaphos_lgn.magnification import (JacobianMagnification,
                                        isotropic_construction_radius_deg)
from dynaphos_lgn.params import resolve_class_codes


def atlas_file_specs(params: Mapping) -> dict:
    """{filename: bytes_per_voxel} from `atlas.files`."""
    return {spec['name']: spec['bytes_per_voxel']
            for spec in params['atlas']['files'].values()}


def check_atlas_files(directory: Union[str, Path], params: Mapping) -> str:
    """Check each .DAT file's size against the dtype it is read with,
    since a headerless read at the wrong width does not fail."""
    directory = Path(directory)
    n_voxels = int(np.prod(params['atlas']['shape_ml_dv_ap']))
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
        lines.append("  A size mismatch means the file is read at the wrong "
                     "width; fix this before going further.")
    return '\n'.join(lines)


def atlas_report(atlas, params: Mapping) -> str:
    """Check a loaded atlas against the paper's published totals."""
    if atlas.is_synthetic:
        return ("Atlas check: SYNTHETIC atlas -- there is nothing to check "
                "it against, and nothing it says means anything "
                "biologically.")

    expected = params['validation']['erwin_1999']
    max_ecc = params['atlas']['max_eccentricity_deg']
    n_voxels = int(np.prod(atlas.atlas_shape))
    magno = np.isin(atlas.layer, resolve_class_codes(params, 'magno'))
    parvo = np.isin(atlas.layer, resolve_class_codes(params, 'parvo'))
    magno_cells = float(atlas.cells_per_voxel[magno].sum())
    parvo_cells = float(atlas.cells_per_voxel[parvo].sum())
    # Erwin et al.'s totals cover the magno and parvo laminae only.
    tissue = magno | parvo
    volume = float(tissue.sum()) * atlas.voxel_size_mm ** 3
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
        f"{atlas.voxel_size_mm * 1e3:.0f} um",
        f"  LAYERS == 0 (extralaminar)        "
        f"{int((atlas.layer == 0).sum()):,}",
        f"  ECC == 999                        "
        f"{int((atlas.eccentricity == 999).sum()):,}",
        f"  INCL == 999                       "
        f"{int((atlas.inclination == 999).sum()):,}",
        f"  valid (all three gates)           "
        f"{int(atlas.valid.sum()):,} "
        f"({atlas.valid.mean() * 100:.1f}% of the grid)",
        f"  foveola column                    "
        + ("not loaded" if atlas.foveola_column is None
           else f"{len(atlas.foveola_column)} voxels, all at ECC == 0"),
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
        f"  Reconstruction accuracy: mean "
        f"{expected['reconstruction_accuracy_um'][0]} um "
        f"(SD {expected['reconstruction_accuracy_um'][1]} um).",
    ]
    if atlas.foveola_column is not None:
        lines += ['', foveal_reference_report(atlas, params)]
    return '\n'.join(lines)


def jacobian_report(jacobian_atlas, atlas, params: Mapping) -> str:
    """Fit coverage, fit quality, flags and the magnification field
    binned by eccentricity.

    Counts and rates are exact; the per-bin magnification medians use a
    random sample of `validation.jacobian_report_sample_size` voxels.
    """
    ecc_edges = params['validation']['report_eccentricity_edges']
    sample_size = params['validation']['jacobian_report_sample_size']
    seed = params['validation']['jacobian_report_seed']

    valid_in = atlas.valid
    fit_ok = jacobian_atlas.jacobian_valid
    sentinel = atlas.is_ipsi_sentinel() & valid_in

    rng = np.random.default_rng(seed)
    fit_index = np.flatnonzero(fit_ok.ravel())
    sampled = (rng.choice(fit_index, sample_size, replace=False)
               if fit_index.size > sample_size else fit_index)
    sample_major, sample_minor, _ = JacobianMagnification.decompose(
        jacobian_atlas.jacobian.reshape(-1, 2, 3)[sampled])
    sample_ecc = atlas.eccentricity_deg.ravel()[sampled]

    residual = jacobian_atlas.residual_rms[fit_ok]
    residual = residual[np.isfinite(residual)]
    neighbors = jacobian_atlas.n_neighbors[fit_ok]
    isotropic = jacobian_atlas.isotropic_flag
    unreliable = jacobian_atlas.unreliable_beyond_neighborhood_flag

    lines = [
        "Jacobian atlas validation",
        "-" * 68,
        f"  valid input voxels                {int(valid_in.sum()):,}",
        f"  of those, ipsi +-135 sentinel     {int(sentinel.sum()):,} "
        f"({sentinel.sum() / max(valid_in.sum(), 1) * 100:.1f}%)",
        f"  voxels with a usable fit          {int(fit_ok.sum()):,} "
        f"({fit_ok.sum() / max(valid_in.sum(), 1) * 100:.1f}% of valid "
        f"input)",
        f"  neighbors per fit (median/min)   "
        f"{np.median(neighbors):.0f} / {neighbors.min():.0f} "
        f"(26 is the maximum possible at radius 1)",
        "",
        "  Fit quality (in-neighborhood residual, degrees):",
        f"    median {np.median(residual):.3f}   mean "
        f"{residual.mean():.3f}   p99 {np.percentile(residual, 99):.3f}   "
        f"above 1 deg: "
        f"{(residual > 1).mean() * 100:.4f}%",
        "",
        f"  isotropic-by-construction (central "
        f"{isotropic_construction_radius_deg(params):.0f} deg): "
        f"{int(isotropic.sum()):,} "
        f"({isotropic[fit_ok].mean() * 100:.1f}% of fit-ok)",
        f"  unreliable beyond neighborhood:  {int(unreliable.sum()):,} "
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
        "    sigma1/sigma2: principal magnifications in deg/mm; aniso: their",
        "    ratio (Connolly & Van Essen report 2-3x).",
    ]
    return '\n'.join(lines)
