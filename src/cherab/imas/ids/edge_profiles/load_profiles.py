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
"""Module for loading edge-profile-related data from IMAS IDS structures."""

import numpy as np

from imas.ids_structure import IDSStructure

from ..common.species import (
    get_element_list,
    get_ion,
    get_ion_state,
    get_neutral,
    get_neutral_state,
)

__all__ = ["load_edge_profiles", "load_edge_species"]


def load_edge_profiles(
    species_struct: IDSStructure,
    grid_subset_index: int = 5,
    backup_species_struct: IDSStructure | None = None,
) -> dict[str, np.ndarray | None]:
    """Load edge profiles from a given species structure.

    The profiles are taken from the arrays with the given grid and subset indices
    (e.g. ``ggd[i1].electrons``, ``ggd[i1].ion[i2].states[i3]``).

    The returned dictionary values for missing profiles are None.

    Parameters
    ----------
    species_struct
        The ids structure containing the profiles for a single species.
    grid_subset_index
        Identifier index of the grid subset, by default 5 (``"Cells"``).
    backup_species_struct
        The backup ids structure that is used if the profile is missing in
        species_struct, by default None.

    Returns
    -------
    dict[str, np.ndarray | None]
        Dictionary with the following keys: ``density``, ``density_fast``, ``temperature``,
        ``velocity_radial``, ``velocity_parallel``, ``velocity_poloidal``, ``velocity_phi``,
        ``velocity_r``, ``velocity_z``, ``z_average``.
    """
    profiles = {
        "density": None,
        "density_fast": None,
        "temperature": None,
        "velocity_radial": None,
        "velocity_parallel": None,
        "velocity_poloidal": None,
        "velocity_phi": None,
        "velocity_r": None,
        "velocity_z": None,
        "z_average": None,
    }

    scalar_profiles = ("density", "density_fast", "temperature", "z_average")

    for name in scalar_profiles:
        profiles[name] = _get_profile(species_struct, name, grid_subset_index)
        if profiles[name] is None and backup_species_struct is not None:
            profiles[name] = _get_profile(backup_species_struct, name, grid_subset_index)

    # velocity
    velocity_profiles = ("radial", "parallel", "poloidal", "phi", "r", "z")
    for s in species_struct.velocity:
        if s.grid_subset_index == grid_subset_index:
            for name in velocity_profiles:
                prof = getattr(s, name)
                profiles["velocity_" + name] = np.asarray_chkfinite(prof) if len(prof) else None
            break
    if (
        all(profiles["velocity_" + name] is None for name in velocity_profiles)
        and backup_species_struct is not None
    ):
        for s in backup_species_struct.velocity:
            if s.grid_subset_index == grid_subset_index:
                for name in velocity_profiles:
                    prof = getattr(s, name)
                    profiles["velocity_" + name] = np.asarray_chkfinite(prof) if len(prof) else None
                break

    return profiles


def load_edge_species(
    ggd_struct: IDSStructure, grid_subset_index: int = 5
) -> dict[str, dict[str, np.ndarray | None]]:
    """Load edge plasma species and their profiles from a given GGD structure.

    The profiles are taken from the arrays with the given grid and subset indices
    (e.g. ``ggd[i1].electrons``, ``ggd[i1].ion[i2].states[i3]``).

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
    ggd_struct
        The ggd ids structure containing the profiles.
    grid_subset_index
        Identifier index of the grid subset, by default 5 (``"Cells"``).

    Returns
    -------
    dict[str, dict[str, ndarray | None]]
        Dictionary with plasma profiles.

    Raises
    ------
    RuntimeError
        If unable to determine the species due to missing element information.
    """
    species_types = ("molecule", "molecular_bundle", "ion", "ion_bundle")
    composition = {species_type: {} for species_type in species_types}

    composition["electron"] = load_edge_profiles(ggd_struct.electrons, grid_subset_index)

    # ions
    ion_elements = []
    for ion in ggd_struct.ion:
        elements = tuple(get_element_list(ion.element))
        if not len(elements):
            raise RuntimeError("Unable to determine the ion species, ion.element AOS is empty.")
        ion_elements.append(elements)

        if len(ion.state):
            shared_temperature = _get_profile(ion, "temperature", grid_subset_index)
            backup_ids = None if len(ion.state) > 1 else ion
            for i, state in enumerate(ion.state):
                species_type, species_id = get_ion_state(state, i, elements, grid_subset_index)
                if species_id in composition[species_type]:
                    print(f"Warning! Skipping duplicated ion: {state.name.strip()}")
                    continue
                profiles = load_edge_profiles(state, grid_subset_index, backup_ids)
                if backup_ids is None and profiles["temperature"] is None:
                    profiles["temperature"] = shared_temperature
                composition[species_type][species_id] = profiles
        else:
            species_type, species_id = get_ion(ion, elements)
            if species_id in composition[species_type]:
                print(f"Warning! Skipping duplicated ion: {ion.name.strip()}")
            else:
                composition[species_type][species_id] = load_edge_profiles(ion, grid_subset_index)

    # neutrals
    for neutral in ggd_struct.neutral:
        elements = tuple(get_element_list(neutral.element))
        if not len(elements):
            elements = ion_elements[neutral.ion_index - 1]

        if len(neutral.state):
            shared_temperature = _get_profile(neutral, "temperature", grid_subset_index)
            backup_ids = None if len(neutral.state) > 1 else neutral
            for state in neutral.state:
                species_type, species_id = get_neutral_state(state, elements)
                if species_id in composition[species_type]:
                    print(f"Warning! Skipping duplicated neutral: {state.name.strip()}")
                    continue
                profiles = load_edge_profiles(state, grid_subset_index, backup_ids)
                if backup_ids is None and profiles["temperature"] is None:
                    profiles["temperature"] = shared_temperature
                composition[species_type][species_id] = profiles
        else:
            species_type, species_id = get_neutral(neutral, elements)
            if species_id in composition[species_type]:
                print("Warning! Skipping duplicated neutral: ")
            else:
                composition[species_type][species_id] = load_edge_profiles(
                    neutral, grid_subset_index
                )

    # Replace missing species temperature with average ion temperature
    tion = _get_profile(ggd_struct, "t_i_average", grid_subset_index)
    if tion is not None:
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


def _get_profile(species_struct, name, grid_subset_index):
    if hasattr(species_struct, name):
        for s in getattr(species_struct, name):
            if s.grid_subset_index == grid_subset_index:
                return np.asarray_chkfinite(s.values)
