"""Projection columns on the synthetic atlas, whose map is known in
closed form: eccentricity along AP, inclination across ML, laminae
stacked along DV. A column there runs straight down DV."""
import numpy as np
import pytest

from dynaphos_lgn.projection import (column_axis_mm, projection_column,
                                     visual_field_xy)

from .conftest import requires_real_atlas


@pytest.fixture(scope='module')
def column(synthetic_atlas, lgn_params):
    return projection_column(synthetic_atlas, lgn_params, 5.0, 20.0)


def test_every_layer_contributes(synthetic_atlas, column):
    layers = np.unique(synthetic_atlas.layer[synthetic_atlas.valid])
    assert sorted(column.voxels) == sorted(int(c) for c in layers)
    assert all(len(v) for v in column.voxels.values())


def test_voxels_represent_the_point(synthetic_atlas, column):
    voxels = column.all_voxels
    mask = np.zeros(synthetic_atlas.atlas_shape, dtype=bool)
    mask[tuple(voxels.T)] = True
    xy = visual_field_xy(synthetic_atlas, mask)
    target = 5.0 * np.array([np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20))])
    assert np.all(np.linalg.norm(xy - target, axis=1)
                  <= column.tolerance_deg + 1e-6)


def test_column_runs_across_the_laminae(synthetic_atlas, column):
    _, axis, length = column_axis_mm(synthetic_atlas, column.all_voxels)
    assert abs(axis[1]) > 0.9          # DV
    assert length > 0


def test_tolerance_grows_with_eccentricity(synthetic_atlas, lgn_params):
    near = projection_column(synthetic_atlas, lgn_params, 0.5, 0.0)
    far = projection_column(synthetic_atlas, lgn_params, 20.0, 0.0)
    assert far.tolerance_deg > near.tolerance_deg


@requires_real_atlas
def test_center_of_the_fovea_traces_the_foveola_column(real_atlas,
                                                       lgn_params):
    """Tracing (0, 0) in the parvo layers lands on FOVEOLA.DAT."""
    from scipy.spatial import cKDTree
    from dynaphos_lgn.params import resolve_class_codes
    if real_atlas.foveola_column is None:
        pytest.skip('FOVEOLA.DAT not loaded.')
    column = projection_column(real_atlas, lgn_params, 0.0, 0.0)
    parvo = np.vstack([column.voxels[c]
                       for c in resolve_class_codes(lgn_params, 'parvo')])
    distance, _ = cKDTree(real_atlas.foveola_column).query(parvo)
    assert np.median(distance) <= 1.5
