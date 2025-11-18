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
"""Module for loading core-profile-related data from IMAS IDS structures."""

import numpy as np

from imas.ids_structure import IDSStructure

from ..common.species import (
    get_element_list,
    get_ion,
    get_ion_state,
    get_neutral,
    get_neutral_state,
)

__all__ = ["load_core_profiles", "load_core_grid", "load_core_species"]


def load_core_profiles(
    species_struct: IDSStructure, backup_species_struct: IDSStructure | None = None
) -> dict[str, np.ndarray | None]:
    """Load core profiles from a given species structure.

    The returned dictionary values for missing profiles are None.

    Parameters
    ----------
    species_struct
        IDS structure containing the profiles for a single species.
    backup_species_struct
        The backup ids structure that is used if the profile is missing in species_struct,
        by default None.

    Returns
    -------
    dict[str, ndarray | None]
        Dictionary with the profiles: ``density``, ``density_thermal``, ``density_fast``,
        ``temperature`` and ``z_average_1d``.
    """
    profiles = {
        "density": None,
        "density_thermal": None,
        "density_fast": None,
        "temperature": None,
        "z_average_1d": None,
    }

    for name in profiles:
        profiles[name] = _get_profile(species_struct, name)
        if profiles[name] is None and backup_species_struct is not None:
            profiles[name] = _get_profile(backup_species_struct, name)

    return profiles


def load_core_grid(grid_struct: IDSStructure) -> dict[str, np.ndarray | None]:
    """Load grid properties of the core profiles.

    The returned dictionary values for missing data are None.

    Parameters
    ----------
    grid_struct
        The IDS structure containing the grid data for 1D profiles.

    Returns
    -------
    dict[str, np.ndarray | None]
        Dictionary with the following keys: ``rho_tor_norm``, ``psi``, ``volume``,
        ``area``, ``surface``.
    """
    grid = {
        "rho_tor_norm": None,
        "volume": None,
        "area": None,
        "surface": None,
        "psi": None,
    }

    for name in grid:
        grid[name] = _get_profile(grid_struct, name)

    return grid


def load_core_species(profiles_struct: IDSStructure) -> dict[str, dict[str, np.ndarray | None]]:
    """Load core plasma species and their profiles from a given profiles IDS structure.

    The returned dictionary has the following structure.

    .. autolink-skip::
    .. code-block:: python

        {
            'electron': {
                'density': array,
                'temperature': array,
                ...
            },
            'molecule': {
                molecule_id: {  # frozenset identifier
                    'density': array,
                    'temperature': array,
                    ...
                },
                ...
            'molecular_bundle': {
                molecular_bundle_id: {  # frozenset identifier
                    'density': array,
                    'temperature': array,
                    ...
                },
                ...
            'ion': {
                ion_id: {  # frozenset identifier
                    'density': array,
                    'temperature': array,
                    ...
                },
                ...
            'ion_bundle': {
                ion_bundle_id: {  # frozenset identifier
                    'density': array,
                    'temperature': array,
                    ...
                },
            },
        }

    where species are identified by frozensets with (key, value) pairs with the following keys.

    +----------------------+------------------------------------------------------------+
    | Species Type         | Identifier Keys                                            |
    +======================+============================================================+
    | ``molecule``         | ``name``, ``elements``, ``z``, ``electron_configuration``, |
    |                      | ``vibrational_level``, ``vibrational_mode``;               |
    +----------------------+------------------------------------------------------------+
    | ``molecular_bundle`` | ``name``, ``elements``, ``z_min``, ``z_max``;              |
    +----------------------+------------------------------------------------------------+
    | ``ion``              | ``name``, ``element``, ``z``, ``electron_configuration``;  |
    +----------------------+------------------------------------------------------------+
    | ``ion_bundle``       | ``name``, ``element``, ``z_min``, ``z_max``.               |
    +----------------------+------------------------------------------------------------+

    Parameters
    ----------
    profiles_struct
        The IDS structure containing the core profiles data.

    Returns
    -------
    dict[str, dict[str, ndarray | None]]
        Dictionary with the species and their profiles.

    Raises
    ------
    RuntimeError
        If unable to determine the species type or identifier.
    """
    species_types = ("molecule", "molecular_bundle", "ion", "ion_bundle")
    composition = {species_type: {} for species_type in species_types}

    composition["electron"] = load_core_profiles(profiles_struct.electrons)

    # ions
    ion_elements = []
    for ion in profiles_struct.ion:
        elements = tuple(get_element_list(ion.element))
        if not len(elements):
            raise RuntimeError("Unable to determine the ion species, ion.element AOS is empty.")
        ion_elements.append(elements)

        if len(ion.state):
            shared_temperature = _get_profile(ion, "temperature")
            backup_ids = None if len(ion.state) > 1 else ion
            for i, state in enumerate(ion.state):
                species_type, species_id = get_ion_state(state, i, elements)
                if species_id in composition[species_type]:
                    print(f"Warning! Skipping duplicated ion: {state.name.strip()}")
                    continue
                profiles = load_core_profiles(state, backup_ids)
                if backup_ids is None and profiles["temperature"] is None:
                    profiles["temperature"] = shared_temperature
                composition[species_type][species_id] = profiles
        else:
            species_type, species_id = get_ion(ion, elements)
            if species_id in composition[species_type]:
                print(f"Warning! Skipping duplicated ion: {ion.name.strip()}")
            else:
                composition[species_type][species_id] = load_core_profiles(ion)

    # neutrals
    for neutral in profiles_struct.neutral:
        elements = tuple(get_element_list(neutral.element))
        if not len(elements):
            elements = ion_elements[neutral.ion_index - 1]

        if len(neutral.state):
            shared_temperature = _get_profile(neutral, "temperature")
            backup_ids = None if len(neutral.state) > 1 else neutral
            for state in neutral.state:
                species_type, species_id = get_neutral_state(state, elements)
                if species_id in composition[species_type]:
                    print(f"Warning! Skipping duplicated neutral: {state.name.strip()}")
                    continue
                profiles = load_core_profiles(state, backup_ids)
                if backup_ids is None and profiles["temperature"] is None:
                    profiles["temperature"] = shared_temperature
                composition[species_type][species_id] = profiles
        else:
            species_type, species_id = get_neutral(neutral, elements)
            if species_id in composition[species_type]:
                print("Warning! Skipping duplicated neutral: ")
            else:
                composition[species_type][species_id] = load_core_profiles(neutral)

    # Replace missing species temperature with average ion temperature
    tion = profiles_struct.t_i_average
    if len(tion):
        for species_type in species_types:
            for species_id, profiles in composition[species_type].items():
                if profiles["temperature"] is None:
                    d = {first: second for first, second in species_id}
                    print(
                        "Warning! Using average ion temperature for the {} {}.".format(
                            d["name"], species_type
                        )
                    )
                    profiles["temperature"] = tion

    return composition


def _get_profile(ids_struct, name):
    if hasattr(ids_struct, name):
        arr = getattr(ids_struct, name)
        if len(arr):
            return np.asarray_chkfinite(arr)
