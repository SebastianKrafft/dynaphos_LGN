"""The rebuilt central 1 deg, on the synthetic atlas.

Its closed-form map puts zero eccentricity on a sheet rather than a
column, which the recipe handles the same way.
"""
import numpy as np
import pytest
from scipy.integrate import quad

from dynaphos_lgn.build import jacobian_cache_name
from dynaphos_lgn.fovea import (apply_central_eccentricity, central_region,
                                interpolation_rank,
                                rebuilt_central_eccentricity)
from dynaphos_lgn.magnification import (isotropic_construction_radius_deg,
                                        malpeli_volume_magnification)


@pytest.fixture(scope='module')
def rebuilt(synthetic_atlas, lgn_params):
    return rebuilt_central_eccentricity(synthetic_atlas, lgn_params)


def test_volume_magnification_is_malpeli_eq_2(lgn_params):
    assert float(malpeli_volume_magnification(0.0, lgn_params)) == (
        pytest.approx(17.93 * 2.6612 ** -2.2908))


def test_rank_runs_from_the_column(synthetic_atlas, lgn_params):
    rank = interpolation_rank(synthetic_atlas, lgn_params)
    assert np.all(rank[tuple(synthetic_atlas.foveola_column.T)] == 0)
    central = central_region(synthetic_atlas, lgn_params)
    assert np.all((rank[central] >= 0) & (rank[central] <= 1))


def test_only_the_central_region_changes(synthetic_atlas, lgn_params,
                                         rebuilt):
    central = central_region(synthetic_atlas, lgn_params)
    assert central.sum() > 0
    assert np.array_equal(rebuilt[~central],
                          synthetic_atlas.eccentricity_deg[~central]
                          .astype(np.float32))
    radius = isotropic_construction_radius_deg(lgn_params)
    assert np.all((rebuilt[central] >= 0) & (rebuilt[central] <= radius))


def test_eccentricity_rises_with_rank(synthetic_atlas, lgn_params, rebuilt):
    central = central_region(synthetic_atlas, lgn_params)
    rank = interpolation_rank(synthetic_atlas, lgn_params)[central]
    order = np.argsort(rank, kind='stable')
    assert np.all(np.diff(rebuilt[central][order]) >= 0)


def test_laminar_volume_follows_eq_2(synthetic_atlas, lgn_params, rebuilt):
    """The share of central laminar tissue below E is Eq. 2's share of
    the hemifield volume below E."""
    central = central_region(synthetic_atlas, lgn_params)
    laminar = central & (synthetic_atlas.layer
                         != synthetic_atlas.missing_layer)
    radius = isotropic_construction_radius_deg(lgn_params)

    def volume(e):
        return quad(lambda x: malpeli_volume_magnification(x, lgn_params)
                    * np.pi * x, 0, e)[0]

    for e in (0.2, 0.5, 0.8):
        share = np.mean(rebuilt[laminar] <= e)
        assert share == pytest.approx(volume(e) / volume(radius),
                                      abs=2 / laminar.sum() + 1e-3)


def test_published_is_left_alone(synthetic_atlas, params_override):
    before = synthetic_atlas.eccentricity_deg
    apply_central_eccentricity(
        synthetic_atlas,
        params_override(atlas__central_eccentricity='published'))
    assert synthetic_atlas.eccentricity_deg is before


def test_unknown_choice_is_refused(synthetic_atlas, params_override):
    with pytest.raises(ValueError, match='published'):
        apply_central_eccentricity(
            synthetic_atlas,
            params_override(atlas__central_eccentricity='smoothed'))


def test_rebuilding_needs_the_column(params_override):
    from dynaphos_lgn.synthetic_atlas import SyntheticAtlas
    params = params_override(synthetic_atlas__shape_ml_dv_ap=(24, 24, 40))
    atlas = SyntheticAtlas(params)
    atlas.foveola_column = None
    with pytest.raises(ValueError, match='foveola'):
        rebuilt_central_eccentricity(atlas, params)


def test_missing_column_keeps_the_published_center(params_override,
                                                    caplog):
    from dynaphos_lgn.synthetic_atlas import SyntheticAtlas
    params = params_override(synthetic_atlas__shape_ml_dv_ap=(24, 24, 40))
    atlas = SyntheticAtlas(params)
    atlas.foveola_column = None
    before = atlas.eccentricity_deg
    apply_central_eccentricity(atlas, params)
    assert atlas.eccentricity_deg is before
    assert atlas.central_eccentricity == 'published'
    assert jacobian_cache_name(params, atlas) == 'JACOBIAN.npz'
    assert any('foveola' in r.message for r in caplog.records)


def test_rebuilt_center_has_its_own_jacobian_cache(params_override):
    params = params_override(atlas__central_eccentricity='rebuilt')
    assert jacobian_cache_name(params) == 'JACOBIAN_CENTRAL_REBUILT.npz'


def test_every_layer_reaches_a_sheet_shaped_fovea(synthetic_atlas,
                                                  lgn_params):
    """The synthetic fovea spans all laminae, so each layer's reference
    sits on it."""
    from dynaphos_lgn.fovea import foveal_references
    column = {tuple(v) for v in synthetic_atlas.foveola_column}
    for code, ref in foveal_references(synthetic_atlas, lgn_params).items():
        assert ref.distance_mm == 0
        assert all(synthetic_atlas.layer[tuple(v)] == code
                   for v in ref.voxels)
        assert any(tuple(v) in column for v in ref.voxels)


def test_reference_is_within_tolerance_of_the_closest_approach(
        synthetic_atlas, params_override):
    from dynaphos_lgn.fovea import distance_from_column_mm, foveal_references
    params = params_override(atlas__foveal_reference_tolerance_vox=2)
    distance = distance_from_column_mm(synthetic_atlas)
    for ref in foveal_references(synthetic_atlas, params).values():
        d = distance[tuple(ref.voxels.T)]
        assert np.all(d <= ref.distance_mm
                      + 2 * synthetic_atlas.voxel_size_mm + 1e-9)
