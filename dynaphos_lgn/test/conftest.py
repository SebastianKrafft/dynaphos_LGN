"""Shared fixtures.

Tests run against the synthetic atlas, so the suite needs no atlas
download. Tests that need the real atlas skip themselves when it is
absent.
"""
import copy
from pathlib import Path

import numpy as np
import pytest

from dynaphos.utils import load_params
from dynaphos_lgn.params import PARAMS_PATH

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_ATLAS_DIR = REPO_ROOT / 'data' / 'Erwin_Atlas'
REAL_ATLAS_AVAILABLE = all(
    (REAL_ATLAS_DIR / f).exists()
    for f in ('ECC.DAT', 'INCL.DAT', 'LAYERS.DAT', 'CELLS.DAT'))

requires_real_atlas = pytest.mark.skipif(
    not REAL_ATLAS_AVAILABLE,
    reason=f"Real Erwin atlas not found in {REAL_ATLAS_DIR}.")


@pytest.fixture(scope='session')
def lgn_params():
    """The shipped config, shrunk so the suite runs in seconds.

    Session-scoped and therefore shared: a test that needs a different
    value must deep-copy rather than mutate, or it will leak into every
    test that runs after it. `params_override` does that for you.
    """
    params = load_params(str(PARAMS_PATH))
    params['run']['resolution'] = [96, 96]
    params['run']['gpu'] = None
    params['electrodes']['n_electrodes'] = 8
    params['electrodes']['max_eccentricity_deg'] = 6.0
    params['electrodes']['n_scatter_points'] = 128
    params['electrodes']['max_candidate_voxels'] = 8000
    params['atlas']['jacobian']['min_neighbors'] = 6
    return params


@pytest.fixture
def params_override(lgn_params):
    """Deep-copy the shared config and apply dotted-path overrides."""
    def override(**paths):
        params = copy.deepcopy(lgn_params)
        for path, value in paths.items():
            node = params
            *parents, leaf = path.split('__')
            for key in parents:
                node = node[key]
            node[leaf] = value
        return params
    return override


@pytest.fixture(scope='session')
def real_atlas(lgn_params):
    """The real Erwin atlas, loaded once. Tests using it must carry
    `@requires_real_atlas`."""
    if not REAL_ATLAS_AVAILABLE:
        pytest.skip(f"Real Erwin atlas not found in {REAL_ATLAS_DIR}.")
    from dynaphos_lgn.atlas import ErwinAtlas
    return ErwinAtlas(lgn_params).read(REAL_ATLAS_DIR)


@pytest.fixture(scope='session')
def synthetic_atlas(lgn_params):
    from dynaphos_lgn.synthetic_atlas import SyntheticAtlas
    return SyntheticAtlas(lgn_params)


@pytest.fixture(scope='session')
def synthetic_jacobian(synthetic_atlas, lgn_params):
    from dynaphos_lgn.atlas import JacobianAtlas
    return JacobianAtlas(lgn_params).compute(synthetic_atlas)


@pytest.fixture
def built(lgn_params, synthetic_atlas, synthetic_jacobian):
    """A small simulator on the synthetic atlas."""
    from dynaphos_lgn.build import build_electrode_array
    from dynaphos_lgn.current_spread import RecruitmentKernel
    from dynaphos_lgn.simulator import LGNPhospheneSimulator
    rng = np.random.default_rng(0)
    kernel = RecruitmentKernel.from_params(lgn_params)
    array = build_electrode_array(lgn_params, synthetic_atlas,
                                  synthetic_jacobian, kernel, rng)
    sim = LGNPhospheneSimulator(lgn_params, array, rng=rng)
    return sim, array
