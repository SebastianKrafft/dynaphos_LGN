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
python -m dynaphos_lgn.viz3d                        # 3-D atlas viewer
python -m dynaphos_lgn.slices                       # coronal/sagittal/horizontal
jupyter lab examples/demo_lgn_explorer.ipynb        # the same, live
python -m pytest dynaphos_lgn/test -q
```

`dynaphos_lgn.viz3d` draws the nucleus in stereotaxic millimetres and
paints the atlas's own per-voxel maps onto it -- eccentricity,
inclination, lamina, cell class, eye of origin, cell counts, and the
Jacobian-derived magnification maps -- with electrodes, probes and
planar arrays placed inside it. It writes an interactive HTML file
(`pip install plotly`) or, with `--backend matplotlib`, a PNG that
needs nothing further. `dynaphos_lgn.slices` cuts the same volume into
coronal, sagittal and horizontal planes, where the laminae actually
become visible -- and `--view combined` puts a slice and the 3-D
nucleus in one figure, with one slider moving the cut through both.
Both write a **figure**: a self-contained file in which every state a
control can reach was computed in advance, which is what makes it
shareable and what caps how interactive it can be. For the other half
of that trade, `dynaphos_lgn.explore` drives the same volume from
ipywidgets with the kernel still attached, so nothing is precomputed
and the controls are free: any slice position, any number of parallel
cut planes at any spacing, any plane, any overlay. It writes no file,
so it hands back the YAML and the command line that reproduce what you
are looking at. See
[`dynaphos_lgn/README.md`](dynaphos_lgn/README.md#looking-at-the-atlas)
and [`examples/demo_lgn_explorer.ipynb`](examples/demo_lgn_explorer.ipynb).

The LGN adaptation is deliberately built for modularity rather than accuracy: almost nothing about LGN stimulation has been measured, so every parameter is transplanted, borrowed across species, or fitted in-project. Every one of them lives in `config/params_lgn.yaml`, which names its source.

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
