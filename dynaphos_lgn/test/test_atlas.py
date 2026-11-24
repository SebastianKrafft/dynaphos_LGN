"""Sanity checks for dynaphos_lgn.atlas against real landmarks in the
Erwin, Baker, Busen & Malpeli (1999) LGN atlas -- exercising
ErwinAtlas.horsley_clarke_to_index specifically, not just whether the raw
.DAT arrays load.
"""
from pathlib import Path

import numpy as np
import pytest

from dynaphos_lgn.atlas import ErwinAtlas, MirroredAtlas

ATLAS_DIR = (Path(__file__).resolve().parents[2] / 'data' / 'Erwin_Atlas')

# These checks are landmark tests against the REAL atlas -- they are the
# point of this file, so they are skipped rather than faked when the
# (separately downloaded) atlas files are absent. Everything that can be
# checked without them lives in the other test modules, which run
# against the synthetic atlas.
pytestmark = pytest.mark.skipif(
    not all((ATLAS_DIR / f).exists()
            for f in ('ECC.DAT', 'INCL.DAT', 'LAYERS.DAT', 'CELLS.DAT')),
    reason=f'Real Erwin atlas not found in {ATLAS_DIR}.')

# Independently found by a separate, prior investigation against this
# exact Atlas_Data directory -- a regression check on the count of raw
# ECC.DAT == 0 voxels, not just a number quoted from the paper.
EXPECTED_ZERO_ECCENTRICITY_VOXEL_COUNT = 11_055
# Rows in foveola.dat, as supplied by J. Malpeli.
EXPECTED_FOVEOLA_COLUMN_LENGTH = 66

requires_foveola_file = pytest.mark.skipif(
    not (ATLAS_DIR / 'foveola.dat').exists(),
    reason=f'foveola.dat not found in {ATLAS_DIR}.')


@pytest.fixture(scope='module')
def atlas(lgn_params):
    # ErwinAtlas now takes the parameter dictionary: the grid shape,
    # sentinels, voxel size, Horsley-Clarke origin and laminar codes all
    # come from `atlas` in dynaphos_lgn/config/params_lgn.yaml rather than from
    # hardcoded values.
    return ErwinAtlas(lgn_params).read(ATLAS_DIR)


def zero_eccentricity_voxels(atlas: ErwinAtlas) -> np.ndarray:
    """Voxels with raw ECC.DAT == 0, found from the array rather than
    from the function under test. A ~0.4 mm wide region around the
    foveola's column, not the column itself."""
    return np.argwhere(atlas.eccentricity == 0)


def test_horsley_clarke_to_index_roundtrips_at_zero_eccentricity(atlas):
    """horsley_clarke_to_index should be the exact inverse of the atlas's
    own index -> mm convention."""
    foveal_voxels = zero_eccentricity_voxels(atlas)
    assert len(foveal_voxels) == EXPECTED_ZERO_ECCENTRICITY_VOXEL_COUNT

    ml_idx, dv_idx, ap_idx = foveal_voxels.T
    ml_mm = atlas._origin_ml_mm + ml_idx * atlas.voxel_size_mm
    dv_mm = atlas._origin_dv_mm + dv_idx * atlas.voxel_size_mm
    ap_mm = atlas._origin_ap_mm + ap_idx * atlas.voxel_size_mm

    roundtrip_ml, roundtrip_dv, roundtrip_ap = atlas.horsley_clarke_to_index(
        ml_mm, dv_mm, ap_mm)

    assert np.array_equal(roundtrip_ml, ml_idx)
    assert np.array_equal(roundtrip_dv, dv_idx)
    assert np.array_equal(roundtrip_ap, ap_idx)


@requires_foveola_file
def test_foveola_column_sits_posteriorly(atlas):
    """The foveola's column sits at the posterior (low-AP) end of the
    nucleus, as the paper states."""
    ap_idx = atlas.foveola_column[:, 2]
    assert ap_idx.mean() < 0.25 * atlas.atlas_shape[2]


@requires_foveola_file
def test_foveola_column_lies_in_parvo_tissue_at_zero_eccentricity(
        atlas, lgn_params):
    """Every voxel is at ECC.DAT == 0 (the loader checks this too) and
    in a parvocellular layer or an interlaminar gap: the magno layers
    never reach zero eccentricity in this atlas."""
    from dynaphos_lgn.params import resolve_class_codes

    column = atlas.foveola_column
    assert column.shape == (EXPECTED_FOVEOLA_COLUMN_LENGTH, 3)
    assert np.all(atlas.eccentricity[tuple(column.T)] == 0)
    allowed = [atlas.missing_layer, *resolve_class_codes(lgn_params,
                                                          'parvo')]
    assert np.all(np.isin(atlas.layer[tuple(column.T)], allowed))


@requires_foveola_file
def test_foveola_column_mm_maps_back_to_its_indices(atlas):
    ml_mm, dv_mm, ap_mm = atlas.foveola_column_mm.T
    assert np.array_equal(
        np.column_stack(atlas.horsley_clarke_to_index(ml_mm, dv_mm, ap_mm)),
        atlas.foveola_column)


def test_no_ipsilateral_placeholder_at_zero_eccentricity(atlas):
    """The fovea belongs to both hemifields, so none of it may be
    excluded as ipsilateral tissue."""
    fovea = atlas.valid & (atlas.eccentricity == 0)
    assert fovea.sum() > 0
    assert not (fovea & atlas.is_ipsi_sentinel()).any()
    assert np.array_equal(atlas.eligible(exclude_ipsi=True) & fovea, fovea)


# ======================================================================
# The mirrored right LGN, against the real data
# ======================================================================
# `test_bilateral.py` checks the reflection as arithmetic, on the
# synthetic atlas. These check it against the real thing, where the
# awkward parts actually exist: 16.9M sentinel voxels, a ~169k-voxel
# ipsilateral placeholder at +-135 deg, and eccentricities running out
# past 90 deg.


@pytest.fixture(scope='module')
def mirrored(atlas):
    return MirroredAtlas(atlas)


def hemifield_x(an_atlas):
    """Horizontal visual-field coordinate of every voxel, degrees.

    Positive is the right hemifield. This is what one nucleus can and
    cannot represent, so it is the quantity the whole exercise is about.
    """
    return (an_atlas.eccentricity_deg
            * np.cos(np.deg2rad(an_atlas.inclination_deg)))


def contralateral(an_atlas):
    """Tissue with a real retinotopic position: valid, and not the
    coarse ipsilateral placeholder."""
    return an_atlas.valid & ~an_atlas.is_ipsi_sentinel()


def test_mirror_covers_the_hemifield_the_left_nucleus_cannot(atlas,
                                                             mirrored):
    """The point of the whole exercise, on the real atlas: one nucleus
    reaches one hemifield, and the two together reach both."""
    left, right = hemifield_x(atlas), hemifield_x(mirrored)
    assert left[contralateral(atlas)].max() > 90
    assert left[contralateral(atlas)].min() >= -1e-3
    assert right[contralateral(mirrored)].min() < -90
    assert right[contralateral(mirrored)].max() <= 1e-3


def test_mirror_preserves_the_published_totals(atlas, mirrored, lgn_params):
    """A reflection cannot create or destroy tissue. These are the same
    quantities build.py's validation report checks against Erwin et
    al.'s own numbers, so if the mirror passes here it passes there.
    """
    from dynaphos_lgn.params import resolve_class_codes

    assert int(mirrored.valid.sum()) == int(atlas.valid.sum())
    assert int(mirrored.is_ipsi_sentinel().sum()) == int(
        atlas.is_ipsi_sentinel().sum())
    for cell_class in ('magno', 'parvo'):
        codes = resolve_class_codes(lgn_params, cell_class)
        assert (float(mirrored.cells_per_voxel[np.isin(mirrored.layer,
                                                       codes)].sum())
                == pytest.approx(float(atlas.cells_per_voxel[
                                           np.isin(atlas.layer,
                                                   codes)].sum())))
        assert (float(mirrored.eccentricity_deg[
                          np.isin(mirrored.layer, codes)
                          & mirrored.valid].max())
                == pytest.approx(float(atlas.eccentricity_deg[
                                           np.isin(atlas.layer, codes)
                                           & atlas.valid].max())))


def test_mirror_keeps_the_missing_data_missing(atlas, mirrored):
    """999 means "no retinotopic position here". Reflected as if it were
    an angle it would become -819, stop matching the sentinel, and
    16.9M placeholder voxels would start reading as valid tissue.
    """
    assert int((mirrored.inclination == 999).sum()) == int(
        (atlas.inclination == 999).sum())
    assert int((mirrored.eccentricity == 999).sum()) == int(
        (atlas.eccentricity == 999).sum())


def test_ipsilateral_placeholder_still_marks_the_other_hemifield(mirrored):
    """+-135 deg reflects to +-45 deg. Both name tissue that coarsely
    represents the hemifield its own nucleus does not, which is why it
    is excluded -- so the reflected values have to land on the side the
    reflected nucleus does NOT represent.
    """
    assert sorted(mirrored.ipsi_sentinel_values) == [-45.0, 45.0]
    placeholder = mirrored.is_ipsi_sentinel() & mirrored.valid
    assert placeholder.sum() > 0
    # Strictly positive: the loader moved the placeholder off zero
    # eccentricity, where it would name no hemifield.
    assert np.all(hemifield_x(mirrored)[placeholder] > 0)


def test_zero_eccentricity_reflects_to_the_mirrored_indices(atlas, mirrored):
    """A landmark that is found in the raw array rather than derived
    from the transform under test: the zero-eccentricity voxels must
    appear in the mirrored volume at the reflected ML index, and nowhere
    else."""
    foveal = zero_eccentricity_voxels(atlas)
    reflected = foveal.copy()
    reflected[:, 0] = atlas.atlas_shape[0] - 1 - reflected[:, 0]
    assert np.array_equal(
        np.argwhere(mirrored.eccentricity == 0)[np.lexsort(
            np.argwhere(mirrored.eccentricity == 0).T[::-1])],
        reflected[np.lexsort(reflected.T[::-1])])


@requires_foveola_file
def test_foveola_column_reflects_with_the_nucleus(atlas, mirrored):
    """Reflected indices, still at zero eccentricity, and the same
    Horsley-Clarke position: ML is lateral distance in both nuclei."""
    column = mirrored.foveola_column
    assert np.array_equal(column[:, 0],
                          atlas.atlas_shape[0] - 1 - atlas.foveola_column[:, 0])
    assert np.array_equal(column[:, 1:], atlas.foveola_column[:, 1:])
    assert np.all(mirrored.eccentricity[tuple(column.T)] == 0)
    assert np.allclose(mirrored.foveola_column_mm, atlas.foveola_column_mm)


# ======================================================================
# The central 1 deg (atlas notes, "Eccentricity Assignments for the
# Central 1 Degree")
# ======================================================================
def test_published_center_omits_the_area_factor(atlas, lgn_params):
    """ECC.DAT's laminar volume below each 0.1 deg zone boundary is a
    fixed multiple of the integral of Eq. 2 over E, not over the
    hemifield's area: the area element's factor E appears to be
    missing."""
    from scipy.integrate import quad
    from dynaphos_lgn.magnification import malpeli_volume_magnification

    laminar = atlas.valid
    voxel_mm3 = atlas.voxel_size_mm ** 3
    ratios = []
    for level in range(10):
        bound = level / 10 + 0.05
        count = int((laminar & (atlas.eccentricity <= level)).sum())
        linear = quad(lambda e: malpeli_volume_magnification(e, lgn_params),
                      0, bound)[0] / voxel_mm3
        ratios.append(count / linear)
    assert np.ptp(ratios) < 0.03 * np.mean(ratios)


@requires_foveola_file
def test_rebuilt_center_keeps_the_published_order(atlas, lgn_params):
    """The recipe's ranking reproduces ECC.DAT's ordering; only the
    values assigned along it differ."""
    from scipy.stats import spearmanr
    from dynaphos_lgn.fovea import central_region, interpolation_rank

    central = central_region(atlas, lgn_params)
    rank = interpolation_rank(atlas, lgn_params)
    assert spearmanr(rank[central], atlas.eccentricity[central])[0] > 0.98


@requires_foveola_file
def test_only_the_parvo_layers_reach_the_column(atlas, lgn_params):
    """The column crosses parvo layers 3 and 4; the magno layers' closest
    approach lies off it, with a non-zero eccentricity."""
    from dynaphos_lgn.fovea import foveal_references
    from dynaphos_lgn.params import resolve_class_codes

    references = foveal_references(atlas, lgn_params)
    for code in resolve_class_codes(lgn_params, 'parvo'):
        assert references[code].distance_mm == 0
    for code in resolve_class_codes(lgn_params, 'magno'):
        assert references[code].distance_mm > 0.05
        assert references[code].eccentricity_deg > 0.1
