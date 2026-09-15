# dynaphos_lgn documentation

Background that belongs with the code but not *in* it: where each number
comes from, which assumptions are load-bearing, and why particular
implementation choices were made. The code's own docstrings say what a
function does and what its arguments mean, and point here for the rest.

| Page | Covers |
| --- | --- |
| [atlas.md](atlas.md) | The Erwin et al. (1999) atlas — file format, the two regions it constructs rather than measures, and how the per-voxel Jacobian is fitted |
| [magnification.md](magnification.md) | The three magnification models, Malpeli's density formulas, the density→magnification conversion, and the validation checkpoints |
| [current-spread.md](current-spread.md) | Stoney's law and `K`, why the recruitment kernel is smooth, cell-class scaling, and multi-electrode interaction |
| [receptive-fields.md](receptive-fields.md) | Difference-of-Gaussians sizes by cell class, the conventions that are easy to get wrong, and the flagged assumptions |
| [electrodes-and-rendering.md](electrodes-and-rendering.md) | The setup/forward split, the candidate neighbourhood, ellipse vs. scatter rendering, and phosphene shape |
| [simulator.md](simulator.md) | The forward pass, gradient flow, the soft threshold, units, and strength–duration |
| [calibration.md](calibration.md) | The two quantitative LGN benchmarks that exist, and the implied-`K` diagnostic |

## Validation scripts

`dynaphos_lgn/check_*.py` are standalone, run-by-hand checks against the
real Erwin atlas. Each writes a dated markdown report (and usually a CSV)
into `research/`. They are never imported by the simulator, and they
deliberately return no pass/fail — the atlas and most of the literature
it is compared against trace back to the same animal, so agreement is
partly by construction and the *residuals* are the output.

| script | question |
| --- | --- |
| `check_magnification_agreement.py` | does the density route agree with the live Jacobian, at the configured column thickness? |
| `check_column_thickness.py` | what *is* the column thickness, measured from the atlas rather than assumed? |
| `check_sheet_thickness_morphology.py` | how thick are the layer sheets *geometrically*, with no retinotopy in the chain — and what does that leave for the Jacobian to explain? |
| `check_cell_density_profile.py` | is `rho_v` a constant, and does the atlas's cell *distribution* match Malpeli's formula or only its total? |
| `check_anisotropy_local_frame.py` | does the atlas reproduce Connolly & Van Essen's anisotropy claim, tested in their own coordinate frame and per layer? |
| `check_anisotropy_report.py` | (superseded for orientation by the local-frame script; its magnitude results stand) |
| `check_soft_threshold.py` | what does turning the soft detection threshold on do to the rendered percept and to the gradient? |

## A note on uncertainty

Almost nothing about LGN microstimulation has been measured. Every
physiological parameter in this package is transplanted from another
area, borrowed across species, fitted in-project, or a free
hyperparameter — and several of the load-bearing ones rest on a single
study, sometimes a single animal.

Those pages record what the code needs a reader to know. The full
provenance and verification status of each number is tracked in the
project's external documentation and report, not in the codebase; the
config file's own comments name the source for each block.
