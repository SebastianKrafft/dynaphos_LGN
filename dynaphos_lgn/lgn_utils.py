"""Electrode placement in the LGN atlas.

Spreads electrodes in retinotopic rather than physical space -- see
`docs/electrodes-and-rendering.md`.
"""
from typing import Optional, Tuple

import numpy as np

from dynaphos.utils import Map, polar_to_complex


def get_electrode_layout(atlas,
                         n_electrodes: int,
                         cell_class: Optional[str] = None,
                         min_eccentricity: float = 0.0,
                         max_eccentricity: Optional[float] = None,
                         bin_size_deg: float = 1.0,
                         exclude_ipsi_flat_inclination: bool = False,
                         depth_selection: str = 'random',
                         rng: Optional[np.random.Generator] = None
                         ) -> Tuple[Map, np.ndarray]:
    """Place `n_electrodes` electrodes spread apart in visual-field
    coordinates, by greedy farthest-point sampling.

    :param atlas: A built `ErwinAtlas`.
    :param n_electrodes: Number of electrodes to place.
    :param cell_class: Restrict placement to 'parvo', 'magno', or None
        (both, the default).
    :param min_eccentricity: Lower eccentricity bound, degrees.
    :param max_eccentricity: Upper bound, degrees. None (default) uses
        the maximum available under the other constraints.
    :param bin_size_deg: Visual-field bin size used to collapse voxels
        sharing a retinotopic position into one candidate. Also caps the
        candidate pool size.
    :param exclude_ipsi_flat_inclination: Exclude voxels carrying the
        coarse +-135 deg ipsilateral placeholder.
    :param depth_selection: Which voxel represents each bin, and so
        which LAYER the electrodes land in:
        - 'central': the bin's median voxel along the depth axis.
        - 'random': a uniform draw from the bin, using `rng`. Spreads
          electrodes across laminae.
    :param rng: Used only to pick the farthest-point-sampling start. A
        fresh (non-reproducible) generator if None.
    :return: (visual_field, voxel_indices), a `dynaphos.utils.Map` of
        electrode locations, and an (n_electrodes, 3) int array of
        (ml, dv, ap) atlas indices.
    """
    if rng is None:
        rng = np.random.default_rng()

    valid = atlas.valid.copy()
    if cell_class is not None:
        valid &= np.isin(atlas.layer, list(atlas.LAYER_CODES[cell_class]))
    if exclude_ipsi_flat_inclination:
        valid &= ~atlas.is_ipsi_sentinel()

    ecc = atlas.eccentricity_deg
    if max_eccentricity is None:
        max_eccentricity = float(np.max(ecc[valid])) if valid.any() else 0.0
    valid &= (ecc >= min_eccentricity) & (ecc <= max_eccentricity)

    voxel_idx = np.argwhere(valid)
    if voxel_idx.shape[0] == 0:
        raise ValueError("No valid voxels match the given constraints.")

    ecc_valid = ecc[valid]
    incl_valid = atlas.inclination_deg[valid]

    # Put all voxels with the same retinotopic position into one bin.
    # Else positions with deeper layers are more likely to be picked.
    ecc_bin = np.round(ecc_valid / bin_size_deg).astype(int)
    incl_bin = np.round(incl_valid / bin_size_deg).astype(int)
    bins = np.stack([ecc_bin, incl_bin], axis=1)
    _, inverse = np.unique(bins, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind='stable')
    sorted_inverse = inverse[order]
    starts = np.searchsorted(sorted_inverse,
                             np.arange(sorted_inverse[-1] + 1),
                             side='left')
    ends = np.searchsorted(sorted_inverse,
                           np.arange(sorted_inverse[-1] + 1),
                           side='right')
    if depth_selection == 'central':
        picks = (starts + ends) // 2
    elif depth_selection == 'random':
        picks = starts + (rng.random(len(starts))
                          * np.maximum(ends - starts, 1)).astype(int)
    else:
        raise ValueError(f"Unknown depth_selection {depth_selection!r}; "
                         f"expected 'central' or 'random'.")
    picks = np.minimum(picks, np.maximum(ends - 1, starts))
    unique_rows = order[picks]

    candidate_voxels = voxel_idx[unique_rows]
    candidate_ecc = ecc_valid[unique_rows]
    candidate_incl_rad = np.deg2rad(incl_valid[unique_rows])

    n_candidates = len(candidate_ecc)
    if n_candidates < n_electrodes:
        raise ValueError(
            f"Only {n_candidates} distinct visual-field positions "
            f"available (after {bin_size_deg} deg binning). Fewer "
            f"than the requested {n_electrodes} electrodes. Reduce "
            f"n_electrodes, reduce bin_size_deg, or widen the "
            f"eccentricity range.")

    z = polar_to_complex(candidate_ecc, candidate_incl_rad)

    selected = np.empty(n_electrodes, dtype=int)
    selected[0] = rng.integers(n_candidates)
    min_dist = np.abs(z - z[selected[0]])
    for i in range(1, n_electrodes):
        next_idx = int(np.argmax(min_dist))
        selected[i] = next_idx
        min_dist = np.minimum(min_dist, np.abs(z - z[next_idx]))

    visual_field = Map(r=candidate_ecc[selected],
                       phi=candidate_incl_rad[selected])
    return visual_field, candidate_voxels[selected]
