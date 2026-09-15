"""Shared fixtures.

Every test in this package runs against the synthetic atlas by default,
so the suite needs no 145 MB download and finishes in seconds. The tests
that genuinely need the real atlas skip themselves when it is absent --
and say so, rather than silently passing.
"""
import copy
from pathlib import Path

import numpy as np
import pytest

from dynaphos.utils import load_params

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
    params = load_params(str(REPO_ROOT / 'config' / 'params_lgn.yaml'))
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
    """The real Erwin atlas, for the few tests that need real biology
    rather than working code.

    Session-scoped: loading the four .DAT files costs a couple of
    seconds and ~350 MB, so it should happen once. Tests using this must
    carry `@requires_real_atlas` so the suite still passes on a checkout
    without the data.
    """
    if not REAL_ATLAS_AVAILABLE:
        pytest.skip(f"Real Erwin atlas not found in {REAL_ATLAS_DIR}.")
    from dynaphos_lgn.atlas import ErwinAtlas
    return ErwinAtlas(lgn_params).build_atlas(REAL_ATLAS_DIR)


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
    from dynaphos_lgn.build import (build_electrode_array, build_kernel,
                                    build_magnification_model)
    from dynaphos_lgn.simulator import LGNPhospheneSimulator
    rng = np.random.default_rng(0)
    kernel = build_kernel(lgn_params)
    array = build_electrode_array(lgn_params, synthetic_atlas,
                                  synthetic_jacobian, kernel, rng)
    array.magnification_model = build_magnification_model(
        lgn_params, synthetic_atlas, synthetic_jacobian)
    sim = LGNPhospheneSimulator(lgn_params, array, rng=rng)
    return sim, array
