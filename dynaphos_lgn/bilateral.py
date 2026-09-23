"""Both nuclei at once: one percept covering the whole visual field.

One LGN represents one hemifield, so a single `LGNPhospheneSimulator`
can only ever fill half the frame. `BilateralLGNSimulator` holds one
per nucleus -- the published left atlas and its mirror image -- and adds
their percepts.

The two are genuinely independent simulators, each with its own
electrode array, its own leaky-integrator state and its own per-electrode
thresholds, which is what the biology says: stimulating the left LGN
does not charge the right one. What they share is the rendered frame,
and the sum happens where it would anyway, in the eye:

    percept = clamp( sum over BOTH arrays of brightness * p(detect) *
                     activation map , 0, 1 )

That is the same expression a single array of all the electrodes would
evaluate, with the clamp applied once at the end rather than per
nucleus, so a bilateral run and a hypothetical one-array run of the same
electrodes agree exactly.

Electrodes are ordered left nucleus first, then right, and every
per-electrode quantity this class exposes -- amplitudes in, brightness
and sizes out -- uses that same order. `electrode_slices` and
`hemisphere_of_electrode` say which is which.

What the right half of the percept inherits is spelled out in
`dynaphos_lgn.atlas.MirroredAtlas`: it is the left nucleus reflected,
not a second measurement.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from dynaphos.utils import Map, to_numpy
from dynaphos_lgn.atlas import hemifield_of
from dynaphos_lgn.simulator import LGNPhospheneSimulator


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

        first = self.simulators[self.hemispheres[0]]
        self.data_kwargs = first.data_kwargs
        self.electrode_dimension = first.electrode_dimension
        self.batch_size = first.batch_size
        self.layer_names = first.layer_names

        for name, sim in self.simulators.items():
            if sim.electrode_dimension != self.electrode_dimension:
                raise ValueError(
                    f"The {name} simulator numbers its electrodes on "
                    f"dimension {sim.electrode_dimension}, the "
                    f"{self.hemispheres[0]} one on "
                    f"{self.electrode_dimension}. Both nuclei must be built "
                    f"from the same config.")

        self.counts = [sim.num_phosphenes for sim in self.simulators.values()]
        self.num_phosphenes = int(sum(self.counts))

        bounds = np.cumsum([0] + self.counts)
        self.electrode_slices = {
            name: slice(int(lo), int(hi))
            for name, lo, hi in zip(self.hemispheres, bounds[:-1], bounds[1:])}
        self.hemisphere_of_electrode = np.concatenate(
            [np.full(n, name) for name, n in zip(self.hemispheres,
                                                 self.counts)])

        self._phosphene_maps = None
        self._sampling_mask = None

    # ------------------------------------------------------------------
    # Per-electrode views, in left-then-right order
    # ------------------------------------------------------------------
    def _concat_arrays(self, attribute: str) -> np.ndarray:
        return np.concatenate([getattr(sim.array, attribute)
                               for sim in self.simulators.values()])

    @property
    def vf_xy(self) -> np.ndarray:
        """(n, 2) visual-field position of every electrode, degrees."""
        return self._concat_arrays('vf_xy')

    @property
    def eccentricity_deg(self) -> np.ndarray:
        return self._concat_arrays('eccentricity_deg')

    @property
    def inclination_deg(self) -> np.ndarray:
        return self._concat_arrays('inclination_deg')

    @property
    def out_of_view(self) -> np.ndarray:
        return np.concatenate([sim.out_of_view
                               for sim in self.simulators.values()])

    @property
    def no_magnification(self) -> np.ndarray:
        """Electrodes with no finite magnification, over both nuclei.

        The other way an electrode can be in the model yet absent from
        the image; see `LGNSize.no_magnification`.
        """
        return np.concatenate([sim.no_magnification
                               for sim in self.simulators.values()])

    @property
    def visual_field(self) -> Map:
        """Every phosphene's position, as one polar `Map`.

        The same thing `LGNElectrodeArray.visual_field` is for a single
        nucleus, over both. Note that inclination now uses its whole
        range: one nucleus spans (-90, 90) deg, the other the rest.
        """
        return Map(r=self.eccentricity_deg,
                   phi=np.deg2rad(self.inclination_deg))

    @property
    def shape(self) -> tuple:
        """The per-electrode tensor shape, over both nuclei.

        The sub-simulators' own shape with the electrode dimension
        widened to hold all of them, so a target built against this
        broadcasts against what `get_state` returns.
        """
        first = self.simulators[self.hemispheres[0]].shape
        return (first[:self.electrode_dimension] + (self.num_phosphenes,)
                + first[self.electrode_dimension + 1:])

    def phosphene_sigma_deg(self, current_ua):
        """(major, minor) phosphene sigmas in degrees, both nuclei.

        A reflection leaves both principal magnifications untouched, so
        the right nucleus's sizes are its mirror-image voxels' own --
        nothing about the size model changes between the two.
        """
        pairs = [sim.array.phosphene_sigma_deg(current_ua)
                 for sim in self.simulators.values()]
        return (np.concatenate([major for major, _ in pairs]),
                np.concatenate([minor for _, minor in pairs]))

    # ------------------------------------------------------------------
    # Splitting and joining
    # ------------------------------------------------------------------
    def split(self, x, name: str = 'value') -> List[Optional[torch.Tensor]]:
        """Cut a per-electrode tensor into one piece per hemisphere.

        Scalars and single-element tensors are passed to every
        hemisphere unchanged, so a frequency or pulse width that applies
        to the whole array still does.

        :param x: Tensor, array or scalar, with the electrode dimension
            sized `num_phosphenes`. None passes through as None.
        :param name: Used in the error message.
        """
        if x is None:
            return [None] * len(self.hemispheres)
        tensor = torch.as_tensor(x, **self.data_kwargs) \
            if not isinstance(x, torch.Tensor) else x
        if tensor.ndim == 0 or tensor.numel() == 1:
            # Broadcast rather than hand the same one-element
            # tensor to a simulator that wants one value per
            # electrode. `expand` keeps it a view, so a scalar
            # stays differentiable and costs nothing per side.
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

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    def reset(self):
        for sim in self.simulators.values():
            sim.reset()

    def to_tensor(self, x) -> torch.Tensor:
        return self.simulators[self.hemispheres[0]].to_tensor(x)

    def update(self, amplitude, pulse_width=None, frequency=None):
        """Advance both nuclei by one frame."""
        amplitudes = self.split(amplitude, 'amplitude')
        pulse_widths = self.split(pulse_width, 'pulse_width')
        frequencies = self.split(frequency, 'frequency')
        for sim, amp, pw, freq in zip(self.simulators.values(), amplitudes,
                                      pulse_widths, frequencies):
            sim.update(amp, pw, freq)

    def spatial_activation(self) -> torch.Tensor:
        """Per-electrode activation maps, both nuclei, left first."""
        return torch.cat([sim.spatial_activation()
                          for sim in self.simulators.values()],
                         dim=self.electrode_dimension)

    def __call__(self, amplitude, pulse_width=None,
                 frequency=None) -> torch.Tensor:
        """Render one frame of simulated percept, whole visual field.

        The per-nucleus contributions are summed before the clamp, not
        after, so this agrees exactly with what one array holding all
        the same electrodes would produce.
        """
        self.update(amplitude, pulse_width, frequency)
        return sum(sim.unclamped_percept()
                   for sim in self.simulators.values()).clamp(0, 1)

    # ------------------------------------------------------------------
    # Stimulus sampling
    # ------------------------------------------------------------------
    @property
    def phosphene_maps(self) -> torch.Tensor:
        """(n, H, W) distance from every pixel to each phosphene centre."""
        if self._phosphene_maps is None:
            self._phosphene_maps = torch.cat(
                [sim.phosphene_maps for sim in self.simulators.values()],
                dim=0)
        return self._phosphene_maps

    @property
    def sampling_mask(self) -> torch.Tensor:
        if self._sampling_mask is None:
            self._sampling_mask = torch.cat(
                [sim.sampling_mask for sim in self.simulators.values()],
                dim=0)
        return self._sampling_mask

    def sample_stimulus(self, activation_mask,
                        rescale: bool = False) -> torch.Tensor:
        """Turn an image into per-electrode amplitudes for both nuclei.

        Each nucleus samples the same whole image; because their
        electrodes sit in opposite hemifields, between them they read
        all of it.
        """
        return torch.cat([sim.sample_stimulus(activation_mask, rescale)
                          for sim in self.simulators.values()], dim=-1)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Both nuclei's state, concatenated over electrodes."""
        states = [sim.get_state() for sim in self.simulators.values()]
        dim = self.electrode_dimension
        out = {}
        for key in states[0]:
            values = [state[key] for state in states]
            out[key] = (None if any(v is None for v in values)
                        else torch.cat(values, dim=dim))
        return out

    def layer_activation_table(self) -> Dict[str, np.ndarray]:
        """Last frame's per-layer recruitment, keyed by layer name.

        Both nuclei have the same laminae, so the columns are shared and
        the rows are all the electrodes, left first.
        """
        activations = []
        for name, sim in self.simulators.items():
            if sim.layer_activation is None:
                raise RuntimeError("Call the simulator at least once first.")
            activations.append(to_numpy(sim.layer_activation.detach()))
        values = np.concatenate(activations, axis=self.electrode_dimension)
        return {name: values[..., i]
                for i, name in enumerate(self.layer_names)}

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
        if 'right' in self.simulators:
            lines.append(
                "The right nucleus is the published left atlas reflected "
                "across the")
            lines.append(
                "midline, not a second reconstruction. Its half of the "
                "percept assumes")
            lines.append(
                "the two LGNs are mirror images; see MirroredAtlas for what "
                "that does")
            lines.append("and does not claim.")
        return '\n'.join(lines).rstrip()


def split_targets_by_hemifield(eccentricity_deg: Sequence[float],
                               inclination_deg: Sequence[float],
                               hemispheres: Sequence[str]) -> dict:
    """Route requested percept locations to the nucleus that can reach.

    A target's hemifield is the sign of ``x = E cos(I)``: positive is the
    right hemifield, which the left LGN represents. Targets on the
    vertical meridian (x = 0) go to the left nucleus, arbitrarily but
    consistently -- both represent that strip.

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
