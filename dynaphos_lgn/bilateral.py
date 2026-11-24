"""Both nuclei at once: one percept covering the whole visual field.

`BilateralLGNSimulator` holds one independent `LGNPhospheneSimulator`
per nucleus and sums their unclamped percepts before a single clamp.
Everything per electrode is the nuclei's own, concatenated left nucleus
first; `electrode_slices` and `hemisphere_of_electrode` say which is
which.
"""
from operator import attrgetter
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from dynaphos.utils import Map
from dynaphos_lgn.atlas import hemifield_of
from dynaphos_lgn.simulator import LGNPhospheneSimulator


def _per_electrode(attribute: str, doc: str = None) -> property:
    """A numpy attribute of every nucleus's simulator, concatenated."""
    get = attrgetter(attribute)
    return property(lambda self: np.concatenate(
        [get(sim) for sim in self.simulators.values()]), doc=doc)


class BilateralLGNSimulator:
    """A percept summed over one simulator per nucleus.

    :param simulators: ``{hemisphere: LGNPhospheneSimulator}``, in the
        order the electrodes should be numbered. `build_simulator`
        supplies left first.
    :param params: The parameter dictionary both were built from.
    """

    def __init__(self, simulators: Mapping[str, LGNPhospheneSimulator],
                 params: dict):
        if not simulators:
            raise ValueError("BilateralLGNSimulator needs at least one "
                             "hemisphere.")
        self.params = params
        self.simulators: Dict[str, LGNPhospheneSimulator] = dict(simulators)
        self.hemispheres: List[str] = list(self.simulators)

        first = self._first
        self.data_kwargs = first.data_kwargs
        self.electrode_dimension = first.electrode_dimension
        self.batch_size = first.batch_size
        self.layer_names = first.layer_names
        self.to_tensor = first.to_tensor
        if any(sim.electrode_dimension != self.electrode_dimension
               for sim in self.simulators.values()):
            raise ValueError("Both nuclei must be built from the same "
                             "config: their electrode dimensions differ.")

        self.counts = [sim.num_phosphenes for sim in self.simulators.values()]
        self.num_phosphenes = int(sum(self.counts))
        bounds = np.cumsum([0] + self.counts)
        self.electrode_slices = {
            name: slice(int(lo), int(hi))
            for name, lo, hi in zip(self.hemispheres, bounds[:-1], bounds[1:])}

    @property
    def _first(self) -> LGNPhospheneSimulator:
        return self.simulators[self.hemispheres[0]]

    vf_xy = _per_electrode(
        'vf_xy', "(n, 2) visual-field position of every electrode, degrees.")
    eccentricity_deg = _per_electrode('array.eccentricity_deg')
    inclination_deg = _per_electrode('array.inclination_deg')
    out_of_view = _per_electrode('out_of_view')
    no_magnification = _per_electrode(
        'no_magnification', "Electrodes with no finite magnification.")
    hemisphere_of_electrode = _per_electrode('hemisphere_of_electrode')

    @property
    def visual_field(self) -> Map:
        """Every phosphene's position, as one polar `Map`."""
        return Map(r=self.eccentricity_deg,
                   phi=np.deg2rad(self.inclination_deg))

    @property
    def shape(self) -> tuple:
        """The per-electrode tensor shape, over both nuclei."""
        first = self._first.shape
        return (first[:self.electrode_dimension] + (self.num_phosphenes,)
                + first[self.electrode_dimension + 1:])

    def phosphene_sigma_deg(self, current_ua):
        """(major, minor) phosphene sigmas in degrees, both nuclei."""
        pairs = [sim.phosphene_sigma_deg(current_ua)
                 for sim in self.simulators.values()]
        return tuple(np.concatenate(axis) for axis in zip(*pairs))

    def _cat(self, values, dim: Optional[int] = None) -> torch.Tensor:
        """Per-nucleus tensors joined along the electrode dimension."""
        return torch.cat(list(values),
                         dim=self.electrode_dimension if dim is None else dim)

    def split(self, x, name: str = 'value') -> List[Optional[torch.Tensor]]:
        """Cut a per-electrode tensor into one piece per hemisphere.
        Scalars are broadcast to every hemisphere.

        :param x: Tensor, array or scalar, with the electrode dimension
            sized `num_phosphenes`. None passes through as None.
        :param name: Used in the error message.
        """
        if x is None:
            return [None] * len(self.hemispheres)
        tensor = torch.as_tensor(x, **self.data_kwargs) \
            if not isinstance(x, torch.Tensor) else x
        if tensor.ndim == 0 or tensor.numel() == 1:
            return [tensor.reshape(()).expand(sim.shape)
                    for sim in self.simulators.values()]

        dim = self.electrode_dimension
        if tensor.ndim <= dim or tensor.shape[dim] != self.num_phosphenes:
            breakdown = ' + '.join(f'{n} {h}' for h, n
                                   in zip(self.hemispheres, self.counts))
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}, but dimension "
                f"{dim} must hold all {self.num_phosphenes} electrodes "
                f"({breakdown}). Electrodes are numbered left nucleus "
                f"first; see electrode_slices.")
        return list(torch.split(tensor, self.counts, dim=dim))

    def reset(self):
        for sim in self.simulators.values():
            sim.reset()

    def update(self, amplitude, pulse_width=None, frequency=None):
        """Advance both nuclei by one frame."""
        inputs = zip(self.split(amplitude, 'amplitude'),
                     self.split(pulse_width, 'pulse_width'),
                     self.split(frequency, 'frequency'))
        for sim, pieces in zip(self.simulators.values(), inputs):
            sim.update(*pieces)

    def __call__(self, amplitude, pulse_width=None,
                 frequency=None) -> torch.Tensor:
        """Render one frame of simulated percept, whole visual field."""
        self.update(amplitude, pulse_width, frequency)
        return self._first.finish(sum(sim.unclamped_percept()
                                      for sim in self.simulators.values()))

    def spatial_activation(self) -> torch.Tensor:
        """Per-electrode activation maps, both nuclei, left first."""
        return self._cat(sim.spatial_activation()
                         for sim in self.simulators.values())

    @property
    def phosphene_maps(self) -> torch.Tensor:
        """(n, H, W) distance from every pixel to each phosphene center."""
        return self._cat((sim.phosphene_maps
                          for sim in self.simulators.values()), dim=0)

    @property
    def sampling_mask(self) -> torch.Tensor:
        return self._cat((sim.sampling_mask
                          for sim in self.simulators.values()), dim=0)

    def sample_stimulus(self, activation_mask,
                        rescale: bool = False) -> torch.Tensor:
        """Turn an image into per-electrode amplitudes for both nuclei."""
        return self._cat((sim.sample_stimulus(activation_mask, rescale)
                          for sim in self.simulators.values()), dim=-1)

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Both nuclei's state, concatenated over electrodes. Strings,
        such as the aggregation modes, are shared by both nuclei and
        passed through."""
        states = [sim.get_state() for sim in self.simulators.values()]
        out = {}
        for key, first in states[0].items():
            values = [state[key] for state in states]
            if isinstance(first, str):
                out[key] = first
            elif any(value is None for value in values):
                out[key] = None
            else:
                out[key] = self._cat(values)
        return out

    def layer_activation_table(self) -> Dict[str, np.ndarray]:
        """Last frame's per-layer recruitment, keyed by layer name, rows
        over all electrodes."""
        tables = [sim.layer_activation_table()
                  for sim in self.simulators.values()]
        return {name: np.concatenate([table[name] for table in tables],
                                     axis=self.electrode_dimension)
                for name in self.layer_names}

    def describe(self) -> str:
        lines = [f"BilateralLGNSimulator: {self.num_phosphenes} electrodes "
                 f"over {len(self.hemispheres)} "
                 f"{'nucleus' if len(self.hemispheres) == 1 else 'nuclei'}",
                 ""]
        for name, sim in self.simulators.items():
            span = self.electrode_slices[name]
            lines.append(f"[{name} LGN] electrodes {span.start}-"
                         f"{span.stop - 1}")
            lines.append(sim.describe())
            lines.append("")
        return '\n'.join(lines).rstrip()


def split_targets_by_hemifield(eccentricity_deg: Sequence[float],
                               inclination_deg: Sequence[float],
                               hemispheres: Sequence[str]) -> dict:
    """Route requested percept locations to the nucleus that covers them:
    x = E cos(I) >= 0 to the left LGN, x < 0 to the right one.

    :return: ``{hemisphere: (eccentricity, inclination)}`` for each
        hemisphere in `hemispheres`.
    """
    ecc = np.atleast_1d(np.asarray(eccentricity_deg, dtype=float))
    incl = np.atleast_1d(np.asarray(inclination_deg, dtype=float))
    if ecc.shape != incl.shape:
        raise ValueError(f"eccentricity and inclination must have the same "
                         f"shape, got {ecc.shape} and {incl.shape}.")
    x = ecc * np.cos(np.deg2rad(incl))
    masks = {'left': x >= 0, 'right': x < 0}
    out = {}
    for name in hemispheres:
        mask = masks[name]
        if not mask.any():
            raise ValueError(
                f"No visual-field target falls in the {hemifield_of(name)} "
                f"hemifield, so the {name} LGN would get no electrodes. "
                f"Drop it from atlas.hemispheres (or pass hemispheres=) if "
                f"that is what you meant.")
        out[name] = (ecc[mask], incl[mask])
    return out
