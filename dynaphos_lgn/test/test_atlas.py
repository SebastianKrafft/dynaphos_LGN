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
# exact Atlas_Data directory -- a regression check on the foveal-column
# count, not just a number quoted from the paper.
EXPECTED_FOVEAL_VOXEL_COUNT = 11_055


@pytest.fixture(scope='module')
def atlas(lgn_params):
    # ErwinAtlas now takes the parameter dictionary: the grid shape,
    # sentinels, voxel size, Horsley-Clarke origin and laminar codes all
    # come from `atlas` in config/params_lgn.yaml rather than from
    # hardcoded values.
    return ErwinAtlas(lgn_params).build_atlas(ATLAS_DIR)


def get_foveal_voxels(atlas: ErwinAtlas) -> np.ndarray:
    """Voxels belonging to the fovea's projection column, per the paper:
    "inclination sectors converge on the projection column of the
    foveola posteriorly" -- operationalized as every voxel with raw
    ECC.DAT == 0 (not the 999 missing-data sentinel). Found directly from
    the loaded array, independently of horsley_clarke_to_index, so it's a
    genuine external landmark rather than something derived from the
    function under test.
    """
    return np.argwhere(atlas.eccentricity == 0)


def test_horsley_clarke_to_index_roundtrips_on_foveal_column(atlas):
    """horsley_clarke_to_index should be the exact inverse of the atlas's
    own index -> mm convention.

    Converting the foveal column's voxel indices to mm and back through
    the function under test is a non-circular check: the indices come
    from the raw array, not from this function. Using many non-degenerate
    indices spread across the volume (rather than just the coordinate
    system's origin) also means this would catch an axis-order bug (e.g.
    ML/DV/AP swapped in the return tuple) that an origin-only check
    cannot -- any permutation of an all-zero index still reads (0, 0, 0).
    """
    foveal_voxels = get_foveal_voxels(atlas)
    assert len(foveal_voxels) == EXPECTED_FOVEAL_VOXEL_COUNT

    ml_idx, dv_idx, ap_idx = foveal_voxels.T
    ml_mm = atlas._origin_ml_mm + ml_idx * atlas._voxel_size_mm
    dv_mm = atlas._origin_dv_mm + dv_idx * atlas._voxel_size_mm
    ap_mm = atlas._origin_ap_mm + ap_idx * atlas._voxel_size_mm

    roundtrip_ml, roundtrip_dv, roundtrip_ap = atlas.horsley_clarke_to_index(
        ml_mm, dv_mm, ap_mm)

    assert np.array_equal(roundtrip_ml, ml_idx)
    assert np.array_equal(roundtrip_dv, dv_idx)
    assert np.array_equal(roundtrip_ap, ap_idx)


def test_foveal_column_sits_posteriorly(atlas):
    """The paper states the foveal column sits at the posterior end of
    the nucleus. Since increasing AP index means more anterior (per the
    atlas's origin documentation), the foveal voxels' AP indices should
    cluster near the low end of the 0..319 range rather than the middle
    or high end -- confirming the AP axis is assigned and oriented
    correctly, independently of horsley_clarke_to_index.

    The specific 25%-of-range threshold below isn't itself from a cited
    source -- it's a deliberately loose cutoff, just tight enough to
    distinguish "clustered posteriorly" from "spread evenly" or
    "clustered anteriorly".
    """
    foveal_voxels = get_foveal_voxels(atlas)
    ap_idx = foveal_voxels[:, 2]

    ap_size = atlas.atlas_shape[2]
    assert ap_idx.mean() < 0.25 * ap_size


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
    # Not strictly positive: a placeholder voxel at zero eccentricity
    # sits on the fovea, which belongs to no hemifield.
    x = hemifield_x(mirrored)[placeholder]
    assert np.all(x >= 0)
    assert (x > 0).mean() > 0.9


def test_foveal_column_reflects_to_the_mirrored_indices(atlas, mirrored):
    """A landmark that is found in the raw array rather than derived
    from the transform under test: the foveola's projection column must
    appear in the mirrored volume at the reflected ML index, and nowhere
    else."""
    foveal = get_foveal_voxels(atlas)
    reflected = foveal.copy()
    reflected[:, 0] = atlas.atlas_shape[0] - 1 - reflected[:, 0]
    assert np.array_equal(
        np.argwhere(mirrored.eccentricity == 0)[np.lexsort(
            np.argwhere(mirrored.eccentricity == 0).T[::-1])],
        reflected[np.lexsort(reflected.T[::-1])])
