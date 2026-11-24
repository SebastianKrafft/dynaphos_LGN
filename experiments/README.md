# Experiments

One folder per experiment, named by a plain slug. Model code lives in the
packages and is switched by config flags; an experiment folder holds only
what is specific to it:

- `config.yaml`: the settings it changes, on top of `dynaphos_lgn/config/params_lgn.yaml`
- the script(s) that run it, from the repository root
- `out/`: generated results, not tracked

| folder           | what it does                                              |
|------------------|-----------------------------------------------------------|
| `compare-params` | compares parameter choices under optimized stimulation    |
| `reproduction`   | reproduces the Dynaphos calibration fits with V1 and LGN  |
