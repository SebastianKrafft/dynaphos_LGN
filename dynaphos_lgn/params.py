"""The LGN config shipped with the package, and cell-class lookups in it."""
from pathlib import Path
from typing import Sequence, Union

PARAMS_PATH = Path(__file__).resolve().parent / 'config' / 'params_lgn.yaml'


def resolve_class_codes(params: dict,
                        cell_class: Union[None, str, Sequence[str]]
                        ) -> Sequence[int]:
    """Layer codes of a cell class, a list of classes, or (None) all.

    :param cell_class: A class name from `atlas.layers.classes`, a list
        of them, or None for every laminar code the config defines.
    """
    classes = params['atlas']['layers']['classes']
    if cell_class is None:
        cell_class = list(classes)
    names = [cell_class] if isinstance(cell_class, str) else list(cell_class)
    unknown = [name for name in names if name not in classes]
    if unknown:
        raise ValueError(f"Unknown cell class {unknown}; atlas.layers "
                         f"defines {', '.join(sorted(classes))}.")
    return sorted({code for name in names for code in classes[name]})
