# Copyright 2023 Euratom
# Copyright 2023 United Kingdom Atomic Energy Authority
# Copyright 2023 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.
"""Module for loading radiation emissivity from the radiation IDS."""

import numpy as np
from numpy.typing import NDArray

from imas.ids_defs import EMPTY_INT
from imas.ids_primitive import IDSNumericArray
from imas.ids_structure import IDSStructArray, IDSStructure

__all__ = ["load_radiation_emissivity", "load_radiation_coefficients"]


def _get_emissivity(
    ggd_struct: IDSStructure,
    grid_subset_index: int,
) -> NDArray[np.float64] | None:
    """Extract emissivity values for a given grid subset index from a ggd time-slice structure.

    Parameters
    ----------
    ggd_struct
        A single element of the ``ggd`` (or ``process[i].ggd``) array-of-structures.
    grid_subset_index
        The ``grid_subset_index`` to match against.

    Returns
    -------
    `NDArray[numpy.float64]` or None
        Emissivity values [W/m³], or ``None`` if the requested subset is not found.
    """
    emissivity_arr = getattr(ggd_struct, "emissivity", None)
    if not isinstance(emissivity_arr, IDSStructArray) or not len(emissivity_arr):
        return None

    for item in emissivity_arr:
        idx = getattr(item, "grid_subset_index", EMPTY_INT)
        if idx == grid_subset_index:
            values = getattr(item, "values", None)
            if isinstance(values, IDSNumericArray) and len(values):
                return np.asarray(values, dtype=np.float64)

    return None


def load_radiation_emissivity(
    radiation_ids,
    process_index: int | None = None,
    grid_subset_index: int = 5,
) -> NDArray[np.float64]:
    """Load emissivity values from a radiation IDS time slice.

    Parameters
    ----------
    radiation_ids
        Radiation IDS object (top-level or time slice) obtained from `~imas.db_entry.DBEntry`.
    process_index
        Index of the radiation process whose emissivity to load.
        If ``None`` (default), the total emissivity is read from the top-level ``ggd`` array of the
        IDS.
    grid_subset_index
        ``grid_subset_index`` identifier of the grid subset to read, by default 5 (``"Cells"``).

    Returns
    -------
    `NDArray[numpy.float64]`
        Emissivity values [W/m³] for each cell of the requested grid subset.

    Raises
    ------
    RuntimeError
        If the required AOS is empty or the requested subset cannot be found.
    """
    if process_index is None:
        # Total emissivity from the top-level ggd AOS
        if not len(radiation_ids.ggd):
            raise RuntimeError("The 'ggd' AOS of the radiation IDS is empty.")

        values = _get_emissivity(radiation_ids.ggd[0], grid_subset_index)
        if values is None:
            raise RuntimeError(
                f"Emissivity with grid_subset_index={grid_subset_index} not found "
                "in radiation.ggd[0]."
            )
    else:
        # Per-process emissivity
        if not len(radiation_ids.process):
            raise RuntimeError("The 'process' AOS of the radiation IDS is empty.")
        if process_index >= len(radiation_ids.process):
            raise RuntimeError(
                f"process_index={process_index} is out of range "
                f"[0, {len(radiation_ids.process) - 1}]."
            )

        process = radiation_ids.process[process_index]
        if not len(process.ggd):
            raise RuntimeError(f"The 'ggd' AOS of radiation.process[{process_index}] is empty.")

        values = _get_emissivity(process.ggd[0], grid_subset_index)
        if values is None:
            raise RuntimeError(
                f"Emissivity with grid_subset_index={grid_subset_index} not found "
                f"in radiation.process[{process_index}].ggd[0]."
            )

    return values


def load_radiation_coefficients(
    radiation_ids,
    process_index: int = 0,
    ion_index: int = 0,
    emissivity_index: int = 0,
    grid_subset_index: int | None = None,
) -> NDArray[np.float64]:
    """Load JOREK-style emissivity coefficients from a radiation IDS time slice.

    This accessor targets the coefficient layout typically used by JOREK:
    ``radiation.process[i].ggd[0].ion[j].emissivity[k].coefficients``.

    Parameters
    ----------
    radiation_ids
        Radiation IDS object (top-level or time slice).
    process_index
        Index of the process in the IDS ``process`` AOS.
    ion_index
        Index of the ion entry in ``process[...].ggd[0].ion``.
    emissivity_index
        Index of the emissivity entry in ``ion[...].emissivity``.
    grid_subset_index
        Optional subset index constraint. If provided, coefficient data is accepted
        only when ``grid_subset_index`` matches the emissivity entry.

    Returns
    -------
    `NDArray[numpy.float64]`
        Coefficient array, typically shaped ``(num_vertices * num_modes, 4)``.

    Raises
    ------
    RuntimeError
        If the requested process/ion/emissivity structure is missing or empty.
    """
    if not len(radiation_ids.process):
        raise RuntimeError("The 'process' AOS of the radiation IDS is empty.")
    if process_index < 0 or process_index >= len(radiation_ids.process):
        raise RuntimeError(
            f"process_index={process_index} is out of range [0, {len(radiation_ids.process) - 1}]."
        )

    process = radiation_ids.process[process_index]
    if not len(process.ggd):
        raise RuntimeError(f"The 'ggd' AOS of radiation.process[{process_index}] is empty.")

    ggd = process.ggd[0]
    ions = getattr(ggd, "ion", None)
    if not isinstance(ions, IDSStructArray) or not len(ions):
        raise RuntimeError(
            f"No ion emissivity data found in radiation.process[{process_index}].ggd[0]."
        )
    if ion_index < 0 or ion_index >= len(ions):
        raise RuntimeError(f"ion_index={ion_index} is out of range [0, {len(ions) - 1}].")

    emissivities = getattr(ions[ion_index], "emissivity", None)
    if not isinstance(emissivities, IDSStructArray) or not len(emissivities):
        raise RuntimeError(
            "No emissivity coefficients found in "
            f"radiation.process[{process_index}].ggd[0].ion[{ion_index}]."
        )
    if emissivity_index < 0 or emissivity_index >= len(emissivities):
        raise RuntimeError(
            f"emissivity_index={emissivity_index} is out of range [0, {len(emissivities) - 1}]."
        )

    emissivity = emissivities[emissivity_index]
    if grid_subset_index is not None:
        idx = getattr(emissivity, "grid_subset_index", EMPTY_INT)
        if idx != grid_subset_index:
            raise RuntimeError(
                "Requested emissivity coefficients do not match the requested "
                f"grid_subset_index={grid_subset_index}."
            )

    coefficients = getattr(emissivity, "coefficients", None)
    if not isinstance(coefficients, IDSNumericArray) or not len(coefficients):
        raise RuntimeError(
            "Emissivity coefficients are missing in "
            f"radiation.process[{process_index}].ggd[0].ion[{ion_index}]."
        )

    return np.asarray(coefficients, dtype=np.float64)
