# About the project
A fully differentiable and biologically plausible simulation of cortical prosthetic vision, which can be used for end-to-end optimization.

## This fork: an LGN adaptation

This is a fork of [neuralcodinglab/dynaphos](https://github.com/neuralcodinglab/dynaphos) that adds `dynaphos_lgn`, a simulator for electrical stimulation of the **lateral geniculate nucleus** rather than V1.

- `dynaphos/`: the original V1/cortex package, **unmodified**, kept as the reference baseline and reused for the generic pieces (image processing, differentiable utilities, the leaky-integrator state classes).
- `dynaphos_lgn/`: the LGN adaptation: atlas-based retinotopic lookup, per-voxel Jacobian magnification, per-lamina recruitment, and two phosphene renderers. See [`dynaphos_lgn/README.md`](dynaphos_lgn/README.md).
- `config/params_lgn.yaml`: LGN parameters, annotated with where every number comes from.
- `examples/demo_simulator_lgn.py`: end-to-end demo, including a gradient-descent optimisation and a figure showing the uncertainty in the transplanted excitability constant.

```
python examples/demo_simulator_lgn.py --synthetic   # no atlas download needed
python -m pytest dynaphos_lgn/test -q
```

The LGN adaptation is deliberately built for modularity rather than accuracy: almost nothing about LGN stimulation has been measured, so every parameter is transplanted, borrowed across species, or fitted in-project. Every one of them lives in `config/params_lgn.yaml`, which names its source. [`docs/`](docs/README.md) explains the reasoning a reader of the code needs.

## Installation
`pip install dynaphos`

## Getting Started
- Download the default configuration file (config/params.yaml) from [our repository](https://github.com/neuralcodinglab/dynaphos/) and adjust according to needs. 
- Run the `test` suite.
- See the `examples` directory for simple use cases.

## Citation
van der Grinten, M., van Steveninck, J. D. R., Lozano, A., Pijnacker, L., Rueckauer, B., Roelfsema, P., Marcel van Gerven, Richard van Wezel, Umut Güçlü & Güçlütürk, Y. (2024). Towards biologically plausible phosphene simulation for the differentiable optimization of visual cortical prostheses. eLife, 13, e85812. [https://doi.org/10.7554/eLife.85812](https://doi.org/10.7554/eLife.85812). 

## Experiments
For the end-to-end optimization experiments in our publication see [this repository](https://github.com/neuralcodinglab/viseon/tree/dynaphos-paper). The code for the experiments described in our publication (using this simulator) can be found in [this repository](https://github.com/neuralcodinglab/dynaphos-experiments). All other code that was used in our publication can be made available on request.

## Contact
Feel free to submit an issue, we're happy to help.
