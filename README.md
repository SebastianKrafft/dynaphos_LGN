# About the project
A fully differentiable and biologically plausible simulation of cortical prosthetic vision, which can be used for end-to-end optimization.

## This fork: an LGN adaptation

This is a fork of [neuralcodinglab/dynaphos](https://github.com/neuralcodinglab/dynaphos) that adds `dynaphos_lgn`, a simulator for electrical stimulation of the **lateral geniculate nucleus** rather than V1.

- `dynaphos/`: the original V1/cortex package, unmodified. `dynaphos_lgn` reuses its image processing, utilities and temporal state classes.
- `dynaphos_lgn/`: the LGN adaptation: atlas-based retinotopic lookup, per-voxel Jacobian magnification, per-lamina recruitment and two phosphene renderers. See [`dynaphos_lgn/README.md`](dynaphos_lgn/README.md).
- `dynaphos_lgn/config/params_lgn.yaml`: the LGN parameters, each with its source.
- `examples/`: `demo_simulator_lgn.py` and the `demo_simulator_lgn*.ipynb` notebooks.

```
python examples/demo_simulator_lgn.py --synthetic   # no atlas download needed
python -m dynaphos_lgn.build --cache-jacobian       # once, with the real atlas
python -m pytest dynaphos_lgn/test -q
```

Almost nothing about LGN stimulation has been measured, so most parameters are transplanted from V1, borrowed across species, or fitted in this project. The model is built to make those choices easy to change, not to be accurate.

## Installation
The LGN adaptation is not on PyPI. Clone this repository and install it; the LGN config ships inside the package:

```
pip install -e .          # add [test] for the tests
```

The original V1 package alone is `pip install dynaphos`.

## Getting Started
- For V1, start from `config/params.yaml`; for the LGN, from `dynaphos_lgn/config/params_lgn.yaml`. Adjust according to needs.
- Run the `test` suite.
- See the `examples` directory for simple use cases.

## Citation
van der Grinten, M., van Steveninck, J. D. R., Lozano, A., Pijnacker, L., Rueckauer, B., Roelfsema, P., Marcel van Gerven, Richard van Wezel, Umut Güçlü & Güçlütürk, Y. (2024). Towards biologically plausible phosphene simulation for the differentiable optimization of visual cortical prostheses. eLife, 13, e85812. [https://doi.org/10.7554/eLife.85812](https://doi.org/10.7554/eLife.85812). 

## Experiments
For the end-to-end optimization experiments in our publication see [this repository](https://github.com/neuralcodinglab/viseon/tree/dynaphos-paper). The code for the experiments described in our publication (using this simulator) can be found in [this repository](https://github.com/neuralcodinglab/dynaphos-experiments). All other code that was used in our publication can be made available on request.

## Contact
Feel free to submit an issue, we're happy to help.
