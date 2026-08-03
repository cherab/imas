from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from imas.ids_primitive import IDSNumericArray
from imas.ids_structure import IDSStructure


def get_ids_numeric_field(ids_struct: IDSStructure, name: str) -> NDArray[np.float64] | None:
    """Return a numeric IDS field as a float64 NumPy array.

    Returns
    -------
    `NDArray[numpy.float64]` or None
        The field values, or ``None`` when the field is missing or empty.
    """
    data = getattr(ids_struct, name, None)
    if isinstance(data, IDSNumericArray) and len(data):
        return np.asarray(data, dtype=np.float64)
    return None
