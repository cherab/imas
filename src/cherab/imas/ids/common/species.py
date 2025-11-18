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
"""Module for common functions used to get IDS species information."""

from cherab.core.atomic.elements import Element, lookup_isotope
from imas.ids_defs import EMPTY_FLOAT, EMPTY_INT
from imas.ids_struct_array import IDSStructArray
from imas.ids_structure import IDSStructure

__all__ = [
    "get_ion_state",
    "get_neutral_state",
    "get_ion",
    "get_neutral",
    "get_element_list",
]


def get_ion_state(
    state: IDSStructure,
    state_index: int,
    elements: list[Element],
    grid_subset_index: int | None = None,
) -> tuple[str, frozenset]:
    """Get a unique identifier for an ion state.

    Parameters
    ----------
    state
        The ion_state structure from IMAS.
    state_index
        Index of the state in the list of states.
    elements
        List of elements that make up the ion state.
    grid_subset_index
        The grid subset index to use for 1D profiles, by default None.

    Returns
    -------
    species_type : `str`
        The type of species: 'ion', 'ion_bundle', 'molecule', or 'molecular_bundle'.
    species_id : `frozenset`
        A frozenset of key-value pairs that uniquely identify the species.
    """
    if state.z_min == EMPTY_FLOAT or state.z_max == EMPTY_FLOAT:
        if grid_subset_index is None:  # 1D profiles
            z_average = state.z_average_1d
        else:
            for s in state.z_average:
                if s.grid_subset_index == grid_subset_index:
                    z_average = s.values
        if len(z_average):  # probably, a bundle
            z_min = state.z_min.value if state.z_min != EMPTY_FLOAT else z_average.min()
            z_max = state.z_max.value if state.z_max != EMPTY_FLOAT else z_average.max()
        else:  # probably, a single ion
            z_min = state.z_min.value if state.z_min != EMPTY_FLOAT else state_index + 1
            z_max = state.z_max.value if state.z_max != EMPTY_FLOAT else z_min
    else:
        z_min = state.z_min.value
        z_max = state.z_max.value

    state_dict = {"name": state.name.strip()}

    if len(elements) > 1:  # molecular ions and bundles
        state_dict["elements"] = elements
        if z_min == z_max:
            species_type = "molecule"
            state_dict["z"] = z_min
            state_dict["electron_configuration"] = (
                str(state.electron_configuration) if len(state.electron_configuration) else None
            )
            state_dict["vibrational_mode"] = (
                str(state.vibrational_mode) if len(state.vibrational_mode) else None
            )
            state_dict["vibrational_level"] = (
                state.vibrational_level if state.vibrational_level != EMPTY_FLOAT else None
            )
        else:
            species_type = "molecular_bundle"
            state_dict["z_min"] = z_min
            state_dict["z_max"] = z_max
    else:  # ions and bundles
        state_dict["element"] = elements[0]
        if z_min == z_max:
            species_type = "ion"
            state_dict["z"] = z_min
            state_dict["electron_configuration"] = (
                str(state.electron_configuration) if len(state.electron_configuration) else None
            )
        else:
            species_type = "ion_bundle"
            state_dict["z_min"] = z_min
            state_dict["z_max"] = z_max
    species_id = frozenset(state_dict.items())

    return species_type, species_id


def get_neutral_state(state: IDSStructure, elements: list[Element]) -> tuple[str, frozenset]:
    """Get a unique identifier for a neutral state.

    Parameters
    ----------
    state
        The neutral_state structure from IMAS.
    elements
        List of elements that make up the neutral state.

    Returns
    -------
    species_type : `str`
        The type of species: 'molecule' or 'ion'.
    species_id : `frozenset`
        A frozenset of key-value pairs that uniquely identify the species.
    """
    state_dict = {"name": state.name.strip()}

    if len(elements) > 1:  # molecules
        species_type = "molecule"
        state_dict["elements"] = elements
        state_dict["z"] = 0
        state_dict["electron_configuration"] = (
            str(state.electron_configuration) if len(state.electron_configuration) else None
        )
        state_dict["vibrational_mode"] = (
            str(state.vibrational_mode) if len(state.vibrational_mode) else None
        )
        state_dict["vibrational_level"] = (
            state.vibrational_level.value if state.vibrational_level != EMPTY_FLOAT else None
        )
    else:  # atoms
        species_type = "ion"
        state_dict["element"] = elements[0]
        state_dict["z"] = 0
        state_dict["electron_configuration"] = (
            str(state.electron_configuration) if len(state.electron_configuration) else None
        )

    species_id = frozenset(state_dict.items())

    return species_type, species_id


def get_ion(ion: IDSStructure, elements: list[Element]) -> tuple[str, frozenset]:
    """Get a unique identifier for an ion or molecule.

    Parameters
    ----------
    ion
        The ion structure from IMAS.
    elements
        List of elements that make up the ion.

    Returns
    -------
    species_type : `str`
        The type of species: 'molecule' or 'ion'.
    species_id : `frozenset`
        A frozenset of key-value pairs that uniquely identify the species.
    """
    z_ion = int(ion.z_ion) if ion.z_ion != EMPTY_FLOAT else elements[0].atomic_number
    if len(elements) > 1:
        species_id = frozenset(
            {
                ("name", ion.name.strip()),
                ("elements", elements),
                ("z", z_ion),
                ("electron_configuration", None),
                ("vibrational_mode", None),
                ("vibrational_level", None),
            }
        )
        return "molecule", species_id

    species_id = frozenset(
        {
            ("name", ion.name.strip()),
            ("element", elements[0]),
            ("z", z_ion),
            ("electron_configuration", None),
        }
    )
    return "ion", species_id


def get_neutral(neutral: IDSStructure, elements: list[Element]) -> tuple[str, frozenset]:
    """Get a unique identifier for a neutral or molecule.

    Parameters
    ----------
    neutral
        The neutral structure from IMAS.
    elements
        List of elements that make up the neutral state.

    Returns
    -------
    species_type : `str`
        The type of species: 'molecule' or 'ion'.
    species_id : `frozenset`
        A frozenset of key-value pairs that uniquely identify the species.
    """
    if len(elements) > 1:
        species_id = frozenset(
            {
                ("name", neutral.name.strip()),
                ("elements", elements),
                ("z", 0),
                ("electron_configuration", None),
                ("vibrational_mode", None),
                ("vibrational_level", None),
            }
        )
        return "molecule", species_id

    species_id = frozenset(
        {
            ("name", neutral.name.strip()),
            ("element", elements[0]),
            ("z", 0),
            ("electron_configuration", None),
        }
    )
    return "ion", species_id


def get_element_list(element_aos: IDSStructArray) -> list[Element]:
    """Get a list of elements from an IDS element_aos structure.

    Parameters
    ----------
    element_aos
        The element_aos structure from IMAS.

    Returns
    -------
    list[`~cherab.core.atomic.elements.Element`]
        List of elements extracted from the element_aos structure.
    """
    elements = []
    for element in element_aos:
        mass_number = int(round(element.a))
        zn = int(round(element.z_n))
        isotope = lookup_isotope(zn, number=mass_number)
        if int(round(isotope.element.atomic_weight)) == mass_number:
            isotope = isotope.element  # prefer element over isotope
        atoms_n = 1 if element.atoms_n == EMPTY_INT else element.atoms_n.value
        for _ in range(atoms_n):
            elements.append(isotope)

    return elements
