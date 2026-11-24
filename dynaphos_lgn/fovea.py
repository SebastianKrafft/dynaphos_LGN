"""The atlas's central 1 deg, rebuilt from the recipe in its notes.

Erwin et al. did not interpolate eccentricity inside 1 deg. Per the atlas
website's "Eccentricity Assignments for the Central 1 Degree", they
ranked voxels by linear interpolation between the foveola's projection
column and the 1 deg isoeccentricity surface, then assigned eccentricity
outward so that the laminar volume below each value matched Malpeli et
al. (1996) Eq. 2.

ECC.DAT's zone volumes follow V(E) dE times a constant (~1.6), not the
hemifield area's pi V(E) E dE: the factor E appears to be missing. Here
the recipe is redone with it, without the 0.1 deg rounding, and scaled so
the region still ends at the 1 deg surface.
"""
import logging
from typing import Dict, Mapping, NamedTuple

import numpy as np
from scipy import ndimage
from scipy.integrate import cumulative_trapezoid

from dynaphos_lgn.magnification import (isotropic_construction_radius_deg,
                                        malpeli_volume_magnification)

#: Samples of the cumulative volume function over [0, radius].
_N_SAMPLES = 10_001


def central_region(atlas, params: Mapping) -> np.ndarray:
    """Voxels with retinotopy whose published eccentricity is inside
    `atlas.isotropic_construction_radius_deg`."""
    known = ((atlas.eccentricity != atlas.missing_ecc_incl)
             & (atlas.inclination != atlas.missing_ecc_incl))
    radius = isotropic_construction_radius_deg(params) * atlas.ecc_scale
    return known & (atlas.eccentricity < radius)


def interpolation_rank(atlas, params: Mapping) -> np.ndarray:
    """0 on the foveola's column, 1 on the surrounding retinotopy outside
    the central region, linear in Euclidean distance between them."""
    central = central_region(atlas, params)
    known = ((atlas.eccentricity != atlas.missing_ecc_incl)
             & (atlas.inclination != atlas.missing_ecc_incl))
    to_column = distance_from_column_mm(atlas)
    to_outside = (ndimage.distance_transform_edt(~(known & ~central))
                  * atlas.voxel_size_mm)
    return to_column / (to_column + to_outside)


def rebuilt_central_eccentricity(atlas, params: Mapping) -> np.ndarray:
    """`atlas.eccentricity_deg` with the central region re-assigned.

    Voxels are taken in rank order; each gets the eccentricity at which
    the cumulative Eq. 2 volume, pi V(E) E dE over the hemifield, reaches
    its share of the region's laminar volume.

    :return: float32 eccentricity, deg, of the atlas's shape.
    """
    if atlas.foveola_column is None:
        raise ValueError("Rebuilding the central eccentricity needs the "
                         "foveola's column (atlas.foveola_file).")
    central = central_region(atlas, params)
    index = np.flatnonzero(central)
    rank = interpolation_rank(atlas, params).ravel()[index]
    order = index[np.argsort(rank, kind='stable')]

    laminar = (atlas.layer.ravel()[order] != atlas.missing_layer)
    share = (np.cumsum(laminar) - 0.5 * laminar) / max(laminar.sum(), 1)

    radius = isotropic_construction_radius_deg(params)
    e = np.linspace(0.0, radius, _N_SAMPLES)
    cumulative = cumulative_trapezoid(
        malpeli_volume_magnification(e, params) * np.pi * e, e, initial=0)

    rebuilt = np.array(atlas.eccentricity_deg, dtype=np.float32)
    rebuilt[np.unravel_index(order, atlas.atlas_shape)] = np.interp(
        share * cumulative[-1], cumulative, e)
    return rebuilt


class FovealReference(NamedTuple):
    """One layer's closest approach to the foveola's column."""
    #: (n, 3) (ML, DV, AP) indices of the layer's voxels there.
    voxels: np.ndarray
    #: Distance from the column at the closest approach, mm.
    distance_mm: float
    #: Median eccentricity of `voxels`, deg.
    eccentricity_deg: float


def distance_from_column_mm(atlas) -> np.ndarray:
    """Euclidean distance of every voxel from the foveola's column, mm."""
    off_column = np.ones(atlas.atlas_shape, dtype=bool)
    off_column[tuple(atlas.foveola_column.T)] = False
    return (ndimage.distance_transform_edt(off_column)
            * atlas.voxel_size_mm)


def foveal_references(atlas, params: Mapping) -> Dict[int, FovealReference]:
    """Each valid layer's voxels within
    `atlas.foveal_reference_tolerance_vox` of its closest approach to the
    column. Zero distance for a layer the column crosses."""
    tolerance = (params['atlas']['foveal_reference_tolerance_vox']
                 * atlas.voxel_size_mm)
    distance = distance_from_column_mm(atlas)
    references = {}
    for code in np.unique(atlas.layer[atlas.valid]):
        layer = atlas.valid & (atlas.layer == code)
        closest = float(distance[layer].min())
        near = layer & (distance <= closest + tolerance)
        references[int(code)] = FovealReference(
            np.argwhere(near), closest,
            float(np.median(atlas.eccentricity_deg[near])))
    return references


def foveal_reference_report(atlas, params: Mapping) -> str:
    """A table of `foveal_references`, by layer name."""
    names = params['atlas']['layers']['names']
    lines = ["Foveal reference per layer (closest approach to the "
             "foveola's column)",
             "-" * 68,
             "  layer            distance (mm)   voxels   eccentricity (deg)"]
    for code, ref in foveal_references(atlas, params).items():
        lines.append(f"  {names.get(code, code)!s:<16} {ref.distance_mm:>13.3f}"
                     f" {len(ref.voxels):>8} {ref.eccentricity_deg:>20.2f}")
    return '\n'.join(lines)


def apply_central_eccentricity(atlas, params: Mapping):
    """Replace `eccentricity_deg` when `atlas.central_eccentricity` is
    'rebuilt', and record the outcome on the atlas. Raw `eccentricity`
    keeps the published values. Without the foveola's column the
    published center is kept, with a warning."""
    choice = params['atlas']['central_eccentricity']
    if choice not in ('rebuilt', 'published'):
        raise ValueError(f"atlas.central_eccentricity must be 'published' "
                         f"or 'rebuilt', not {choice!r}.")
    if choice == 'rebuilt' and atlas.foveola_column is None:
        logging.warning("No foveola column, so the central 1 deg keeps "
                        "ECC.DAT's published eccentricities.")
        choice = 'published'
    if choice == 'rebuilt':
        atlas.eccentricity_deg = rebuilt_central_eccentricity(atlas, params)
    atlas.central_eccentricity = choice
    return atlas
