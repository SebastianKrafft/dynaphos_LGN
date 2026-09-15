"""Config access for the LGN package.

Every number this package uses comes from the parameter dictionary:
no module-level constants, no defaults in function signatures. A value
you cannot change without editing code is not refinable, and this
simulator exists to be refined as data arrives.

Use `require`, not ``params['a']['b']``, and never ``.get(key,
fallback)`` -- a fallback in code is the module-level constant this
module exists to abolish, hidden one level deeper. `require` fails
immediately and names the dotted path that was missing.
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
    """Fetch a genuinely optional parameter.

    For values whose absence is meaningful -- a null ``cells_per_mm3``
    meaning "derive it from the atlas" -- not as a way to reintroduce
    defaults. If you are reaching for this to avoid adding a config
    key, add the key instead.
    """
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
