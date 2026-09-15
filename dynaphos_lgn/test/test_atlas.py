"""Sanity checks for dynaphos_lgn.atlas against real landmarks in the
Erwin, Baker, Busen & Malpeli (1999) LGN atlas -- exercising
ErwinAtlas.horsley_clarke_to_index specifically, not just whether the raw
.DAT arrays load.
"""
from pathlib import Path

import numpy as np
import pytest

from dynaphos_lgn.atlas import ErwinAtlas

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
