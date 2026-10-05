"""Bit-identical outputs against a stored reference.

The reference holds the outputs of a small, seeded, CPU-only run on the
synthetic atlas. Any change to the default model shows up here; when a
change is intended, regenerate it with

    python -m dynaphos_lgn.test.test_regression
"""
import copy
import itertools
from pathlib import Path

import numpy as np
import torch

from dynaphos.utils import load_params

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).parent / 'data' / 'regression_reference.npz'

CELL_CLASSES = [None, 'parvo', 'magno']
SOFT_THRESHOLD = [False, True]
HEMISPHERES = [['left'], ['left', 'right']]
N_FRAMES = 20
STORED_FRAMES = (4, 9, 14, 19)
STATE_KEYS = ('brightness', 'activation', 'threshold', 'detection_probability')
RF_ECCENTRICITIES = np.linspace(0.0, 20.0, 9)


def small_params(cell_class, soft_threshold, hemispheres) -> dict:
    """The shipped config, shrunk as in conftest, with one case applied."""
    params = load_params(str(REPO_ROOT / 'config' / 'params_lgn.yaml'))
    params['run']['resolution'] = [64, 64]
    params['run']['gpu'] = None
    params['electrodes']['n_electrodes'] = 8
    params['electrodes']['max_eccentricity_deg'] = 6.0
    params['electrodes']['n_scatter_points'] = 128
    params['electrodes']['max_candidate_voxels'] = 8000
    params['atlas']['jacobian']['min_neighbors'] = 6
    params['electrodes']['cell_class'] = cell_class
    params['thresholding']['soft_threshold'] = soft_threshold
    params['atlas']['hemispheres'] = hemispheres
    return params


def _stimulus(params) -> np.ndarray:
    res_x, res_y = params['run']['resolution']
    y, x = np.mgrid[0:res_y, 0:res_x]
    return ((x + 2 * y) % 17 / 16.0).astype(np.float32)


def simulator_outputs(params) -> dict:
    """Percept and state for a few frames of a fixed stimulus."""
    from dynaphos_lgn.build import build_simulator
    torch.manual_seed(0)
    sim, _ = build_simulator(copy.deepcopy(params), synthetic=True)
    # Offset by 40 uA (the mean detection threshold) so most electrodes
    # cross threshold and the percepts are not empty.
    amplitude = 40e-6 + sim.sample_stimulus(_stimulus(params), rescale=True)
    out = {'amplitude': amplitude}
    for frame in range(N_FRAMES):
        percept = sim(amplitude)
        if frame in STORED_FRAMES:
            out[f'percept_{frame}'] = percept
    state = sim.get_state()
    out.update({key: state[key] for key in STATE_KEYS})
    return {key: value.detach().cpu().numpy() for key, value in out.items()}


def receptive_field_outputs(params) -> dict:
    from dynaphos_lgn.receptive_fields import receptive_field
    out = {}
    for cell_class in ('parvo', 'magno'):
        rf = receptive_field(RF_ECCENTRICITIES, params, cell_class)
        out[f'rf_{cell_class}_center'] = np.asarray(rf.center_radius_deg)
        out[f'rf_{cell_class}_surround'] = np.asarray(rf.surround_radius_deg)
    return out


def all_outputs() -> dict:
    outputs = {}
    for cell_class, soft, hemispheres in itertools.product(
            CELL_CLASSES, SOFT_THRESHOLD, HEMISPHERES):
        case = f'{cell_class}_{"soft" if soft else "hard"}_' \
               f'{"+".join(hemispheres)}'
        params = small_params(cell_class, soft, hemispheres)
        for key, value in simulator_outputs(params).items():
            outputs[f'{case}/{key}'] = value
    outputs.update(receptive_field_outputs(small_params(None, True, ['left'])))
    return outputs


def test_outputs_match_reference():
    reference = np.load(REFERENCE)
    outputs = all_outputs()
    assert sorted(outputs) == sorted(reference.files)
    mismatched = [key for key in reference.files
                  if not np.array_equal(outputs[key], reference[key])]
    assert not mismatched, f"Outputs differ from the reference: {mismatched}"


if __name__ == '__main__':
    REFERENCE.parent.mkdir(exist_ok=True)
    np.savez_compressed(REFERENCE, **all_outputs())
    print(f"Wrote {REFERENCE}")
