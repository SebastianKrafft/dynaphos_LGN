"""Config access for the LGN package.

Parameters are read with `require`, which names the missing dotted path
instead of falling back to a default in code.
"""
from typing import Any, Mapping, Optional, Sequence


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


def resolve_class_codes(params: Mapping, cell_class: Optional[str]
                        ) -> Sequence[int]:
    """LAYERS.DAT codes belonging to a cell class, or all of them.

    :param cell_class: 'magno', 'parvo', or None for every laminar code
        the config defines.
    """
    classes = require(params, 'atlas.layers.classes')
    if cell_class is None:
        return sorted({code for codes in classes.values() for code in codes})
    if cell_class not in classes:
        raise ParameterError(
            f"Unknown cell class {cell_class!r}; atlas.layers.classes "
            f"defines {', '.join(sorted(classes))}.")
    return list(classes[cell_class])
