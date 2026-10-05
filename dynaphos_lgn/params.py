"""Config access for the LGN package.

Parameters are read with `require`, which names the missing dotted path
instead of falling back to a default in code.
"""
from typing import Any, Mapping, Sequence, Union

import numpy as np


class ParameterError(KeyError):
    """A required parameter is missing, or is not the expected shape."""

    def __str__(self) -> str:                      # pragma: no cover
        return self.args[0] if self.args else ''


def require(params: Mapping, path: str) -> Any:
    """Fetch ``params['a']['b']['c']`` for a dotted path ``'a.b.c'``.

    :raises ParameterError: naming the full path and what was found at
        the point it broke.
    """
    node: Any = params
    walked = []
    for key in path.split('.'):
        walked.append(key)
        if not isinstance(node, Mapping):
            raise ParameterError(
                f"Parameter {path!r}: {'.'.join(walked[:-1])!r} is a "
                f"{type(node).__name__}, not a section, so it has no "
                f"{key!r}.")
        if key not in node:
            available = ', '.join(sorted(map(str, node))) or '(empty)'
            raise ParameterError(
                f"Parameter {path!r} is missing: no {key!r} in "
                f"{'.'.join(walked[:-1]) or '<root>'}. Available there: "
                f"{available}.")
        node = node[key]
    return node


def optional(params: Mapping, path: str, default: Any = None) -> Any:
    """Fetch a parameter whose absence (or null) is meaningful, such as
    a null ``cells_per_mm3`` meaning "derive it from the atlas"."""
    try:
        value = require(params, path)
    except ParameterError:
        return default
    return default if value is None else value


def class_codes(params: Mapping) -> dict:
    """{name: layer codes} for every class and subclass in the config."""
    return {**require(params, 'atlas.layers.classes'),
            **optional(params, 'atlas.layers.subclasses', {})}


def resolve_class_codes(params: Mapping,
                        cell_class: Union[None, str, Sequence[str]]
                        ) -> Sequence[int]:
    """Layer codes of a cell class, a list of classes, or (None) all.

    :param cell_class: A class or subclass name from `atlas.layers`, a
        list of them, or None for every laminar code the config defines.
    """
    classes = class_codes(params)
    if cell_class is None:
        cell_class = list(require(params, 'atlas.layers.classes'))
    names = [cell_class] if isinstance(cell_class, str) else list(cell_class)
    unknown = [name for name in names if name not in classes]
    if unknown:
        raise ParameterError(
            f"Unknown cell class {unknown}; atlas.layers defines "
            f"{', '.join(sorted(classes))}.")
    return sorted({code for name in names for code in classes[name]})


def class_membership(params: Mapping, layer_codes: Sequence[int],
                     classes: Sequence[str]) -> np.ndarray:
    """(n_layers, n_classes) 0/1 matrix: which layer belongs to which
    class."""
    return np.array([[code in resolve_class_codes(params, name)
                      for name in classes] for code in layer_codes],
                    dtype=np.float32)
