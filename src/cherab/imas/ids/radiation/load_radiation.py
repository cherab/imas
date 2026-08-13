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

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from imas.ids_structure import IDSStructArray, IDSStructure

from ..common import get_ids_numeric_field
from ..common.ggd.load_data import get_ggd_subset_data

__all__ = [
    "EmissivityData",
    "load_core_emissivity",
    "load_ggd_emissivity",
]


@dataclass
class EmissivityData:
    """Emissivity data."""

    electron: NDArray[np.float64] | None = None
    ion: NDArray[np.float64] | None = None
    neutral: NDArray[np.float64] | None = None

    def sum(self) -> NDArray[np.float64] | None:
        """Return the sum of all available emissivity arrays.

        Returns
        -------
        `NDArray[numpy.float64]` or None
            The sum of all available emissivity arrays, or None if no arrays are available.
        """
        arrays = [arr for arr in (self.electron, self.ion, self.neutral) if arr is not None]
        if not arrays:
            return None
        return np.sum(arrays, axis=0)


def _sum_profile_species_emissivity(
    profile_1d: IDSStructure,
    species_name: str,
) -> NDArray[np.float64] | None:
    """Sum ``<species_name>(:)/emissivity(:)`` for 1D profiles.

    Returns
    -------
    `NDArray[numpy.float64]` or None
        Summed emissivity across all species entries with available data.
    """
    species_arr = getattr(profile_1d, species_name, None)
    if not isinstance(species_arr, IDSStructArray) or not len(species_arr):
        return None

    total = None
    for species in species_arr:
        values = get_ids_numeric_field(species, "emissivity")
        if values is None:
            continue
        total = values.copy() if total is None else total + values

    return total


def _sum_ggd_species_emissivity(
    ggd_struct: IDSStructure,
    species_name: str,
    grid_subset_index: int,
    field: Literal["values", "coefficients"] = "values",
) -> NDArray[np.float64] | None:
    """Sum ``<species_name>(:)/emissivity(:)/<field>(:)`` for one GGD structure.

    For example, this sums ``ggd.ion(i)/emissivity(grid_subset_index)/values(:)`` across all ion
    species entries.

    Returns
    -------
    `NDArray[numpy.float64]` or None
        Summed emissivity across all species entries on the requested subset.
    """
    species_arr = getattr(ggd_struct, species_name, None)
    if not isinstance(species_arr, IDSStructArray) or not len(species_arr):
        return None

    total = None
    for species in species_arr:
        values = get_ggd_subset_data(species, "emissivity", grid_subset_index, field=field)
        if values is None:
            continue
        total = values.copy() if total is None else total + values

    return total


def load_core_emissivity(process: IDSStructure) -> EmissivityData:
    """Load emissivity arrays from ``process.profiles_1d[0]``.

    All species emissivity arrays are summed to produce a single array for each species type
    (electron, ion, neutral).

    Parameters
    ----------
    process
        The IDS structure containing the radiation process data.

    Returns
    -------
    `.EmissivityData`
        Core-region emissivity arrays.

    Examples
    --------
    >>> load_core_emissivity(ids.process[0])
    EmissivityData(
        electron=array([...]),
        ion=array([...]),
        neutral=array([...])
    )
    """
    profiles = getattr(process, "profiles_1d", None)
    if not isinstance(profiles, IDSStructArray) or not len(profiles):
        return EmissivityData()

    profile_1d = profiles[0]
    electrons = getattr(profile_1d, "electrons", None)

    return EmissivityData(
        electron=(
            get_ids_numeric_field(electrons, "emissivity")
            if isinstance(electrons, IDSStructure)
            else None
        ),
        ion=(
            get_ids_numeric_field(profile_1d, "emissivity_ion_total")
            or _sum_profile_species_emissivity(profile_1d, "ion")
        ),
        neutral=(
            get_ids_numeric_field(profile_1d, "emissivity_neutral_total")
            or _sum_profile_species_emissivity(profile_1d, "neutral")
        ),
    )


def load_ggd_emissivity(
    process: IDSStructure,
    grid_subset_index: int,
    field: Literal["values", "coefficients"] = "values",
) -> EmissivityData:
    """Load emissivity data from ``process.ggd[0]``.

    Parameters
    ----------
    process
        The IDS structure containing the radiation process data.
    grid_subset_index
        The index of the GGD subset to retrieve.
    field
        The field to retrieve from the GGD subset, by default "values".

    Returns
    -------
    `.EmissivityData`
        GGD-region emissivity arrays for the requested grid subset.

    Examples
    --------
    >>> load_ggd_emissivity(ids.process[0], grid_subset_index=5)
    EmissivityData(
        electron=array([...]),
        ion=array([...]),
        neutral=array([...])
    )

    >>> load_ggd_emissivity(ids.process[0], grid_subset_index=1, field="coefficients")
    EmissivityData(
        electron=array([[...], [...]]),
        ion=array([[...], [...]]),
        neutral=array([[...], [...]])
    )
    """
    ggd_arr = getattr(process, "ggd", None)
    if not isinstance(ggd_arr, IDSStructArray) or not len(ggd_arr):
        return EmissivityData()

    ggd = ggd_arr[0]
    electrons = getattr(ggd, "electrons", None)

    return EmissivityData(
        electron=(
            get_ggd_subset_data(electrons, "emissivity", grid_subset_index, field=field)
            if isinstance(electrons, IDSStructure)
            else None
        ),
        ion=_sum_ggd_species_emissivity(ggd, "ion", grid_subset_index, field=field),
        neutral=_sum_ggd_species_emissivity(ggd, "neutral", grid_subset_index, field=field),
    )
