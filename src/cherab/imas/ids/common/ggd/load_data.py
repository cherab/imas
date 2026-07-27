"""Functions for loading GGD-related data."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from imas.ids_defs import EMPTY_INT
from imas.ids_primitive import IDSNumericArray
from imas.ids_structure import IDSStructArray, IDSStructure

__all__ = ["get_ggd_subset_data"]


def get_ggd_subset_data(
    ids_struct: IDSStructure,
    name: str,
    grid_subset_index: int,
    field: Literal["values", "coefficients"] = "values",
) -> NDArray[np.float64] | None:
    """Return ``ids_struct.<name>.<field>`` on the requested GGD subset.

    The list of grid subset indices (GGD subset identifiers) can be seen at
    https://imas-data-dictionary.readthedocs.io/en/latest/generated/identifier/ggd_subset_identifier.html

    Parameters
    ----------
    ids_struct
        The IDS structure containing the GGD data.
        Must contain a field named "grid_subset_index".
    name
        The name of the GGD field to retrieve (e.g., ``"electrons"``).
    grid_subset_index
        The index of the GGD subset to retrieve.
    field
        The field to retrieve from the GGD subset, by default "values".

    Returns
    -------
    `NDArray[numpy.float64]` | None
        If `field` is "values", returns 1D `numpy.ndarray`.
        If `field` is "coefficients", returns 2D `numpy.ndarray`.
        Returns None if the requested GGD subset or field is not present in the IDS structure.

    Raises
    ------
    ValueError
        If `field` is not one of "values" or "coefficients".

    Examples
    --------
    >>> get_ggd_subset_data(ids.ggd[0].electrons, "density", 5)

    >>> get_ggd_subset_data(ids.ggd[0].ion[0]., "temperature", 5, field="coefficients")
    """
    if field not in {"values", "coefficients"}:
        raise ValueError(f"Invalid field '{field}', must be 'values' or 'coefficients'")

    struct_arr = getattr(ids_struct, name, None)
    if not isinstance(struct_arr, IDSStructArray) or not len(struct_arr):
        return None

    for sub_struct in struct_arr:
        index = getattr(sub_struct, "grid_subset_index", EMPTY_INT)
        if index == grid_subset_index:
            data = getattr(sub_struct, field, None)
            if isinstance(data, IDSNumericArray) and len(data):
                return np.asarray(data, dtype=np.float64)
            return None

    return None
