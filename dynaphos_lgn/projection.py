"""Projection columns: the tissue in every layer that represents one
point of the visual field.

The atlas maps 3-D tissue onto the 2-D field, so the tissue representing
a point is a line crossing the laminae (Erwin et al. 1999). It is traced
here as the voxels within `atlas.projection_column` tolerance of the
point, layer by layer. A layer that never comes that close contributes
its closest voxels instead, and says how far off they are.
"""
from typing import Dict, Mapping, NamedTuple, Tuple

import numpy as np



class ProjectionColumn(NamedTuple):
    """One visual-field point's column in one nucleus."""
    #: {layer code: (n, 3) (ML, DV, AP) indices}.
    voxels: Dict[int, np.ndarray]
    #: {layer code: visual-field distance of its closest voxel, deg}.
    offset_deg: Dict[int, float]
    #: The tolerance the voxels were selected with, deg.
    tolerance_deg: float

    @property
    def all_voxels(self) -> np.ndarray:
        return np.vstack(list(self.voxels.values()))


def visual_field_xy(atlas, mask: np.ndarray) -> np.ndarray:
    """(n, 2) visual-field position, deg, of every `mask` voxel."""
    e = atlas.eccentricity_deg[mask].astype(float)
    i = np.deg2rad(atlas.inclination_deg[mask].astype(float))
    return np.column_stack([e * np.cos(i), e * np.sin(i)])


def projection_column(atlas, params: Mapping, eccentricity_deg: float,
                      inclination_deg: float) -> ProjectionColumn:
    """The voxels of each layer representing (`eccentricity_deg`,
    `inclination_deg`), in the atlas's own inclination frame.

    The tolerance is `tolerance_deg` or `relative_tolerance` times the
    eccentricity, whichever is larger. Placeholder voxels, whose
    inclination is unknown, are left out.
    """
    cfg = params['atlas']['projection_column']
    tolerance = max(float(cfg['tolerance_deg']),
                    float(cfg['relative_tolerance']) * eccentricity_deg)
    mask = atlas.eligible(exclude_ipsi=True)
    index = np.argwhere(mask)
    layer = atlas.layer[mask]
    angle = np.deg2rad(inclination_deg)
    target = eccentricity_deg * np.array([np.cos(angle), np.sin(angle)])
    distance = np.linalg.norm(visual_field_xy(atlas, mask) - target, axis=1)

    voxels, offset = {}, {}
    for code in np.unique(layer):
        here = layer == code
        closest = float(distance[here].min())
        keep = distance[here] <= max(tolerance, closest)
        voxels[int(code)] = index[here][keep]
        offset[int(code)] = closest
    return ProjectionColumn(voxels, offset, tolerance)


def column_axis_mm(atlas, voxels: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Center, unit direction and length of a column, in Horsley-Clarke
    mm, from its voxels' principal axis."""
    points = np.column_stack(atlas.index_to_horsley_clarke(*voxels.T))
    center = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - center).T))
    axis = vectors[:, -1]
    along = (points - center) @ axis
    return center, axis, float(along.max() - along.min())
