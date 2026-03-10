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
from numpy.typing import NDArray

from imas.ids_primitive import IDSNumericArray
from imas.ids_structure import IDSStructArray, IDSStructure

from ..common.species import (
    ProfileData,
    SpeciesComposition,
    SpeciesType,
    VelocityData,
    get_elements,
    get_ion,
    get_ion_state,
    get_neutral,
    get_neutral_state,
)

__all__ = [
    "load_edge_profiles",
    "load_edge_species",
]


def _get_profile(
    ids_struct: IDSStructure,
    name: str,
    grid_subset_index: int,
    name2: str | None = None,
) -> NDArray[np.float64] | None:
    struct_arr = getattr(ids_struct, name, None)
    if isinstance(struct_arr, IDSStructArray):
        for sub_struct in struct_arr:
            index = getattr(sub_struct, "grid_subset_index", None)
            if index is None:
                return None
            elif index == grid_subset_index:
                data = getattr(sub_struct, "values", None)
                if isinstance(data, IDSNumericArray):
                    return np.asarray(data) if len(data) else None
                elif data is None:
                    data = getattr(sub_struct, name2, None) if name2 is not None else None
                    if isinstance(data, IDSNumericArray):
                        return np.asarray(data) if len(data) else None
                    else:
                        return None
                else:
                    return None
        return None
    else:
        return None


def load_edge_profiles(
    species_struct: IDSStructure,
    grid_subset_index: int = 5,
    backup_species_struct: IDSStructure | None = None,
) -> ProfileData:
    """Load edge profiles from a given species structure.

    Parameters
    ----------
    species_struct
        The ids structure containing the profiles for a single species.
    grid_subset_index
        Identifier index of the grid subset, by default 5 (``"Cells"``).
    backup_species_struct
        The backup ids structure that is used if the profile is missing in species_struct,
        by default None.

    Returns
    -------
    `.ProfileData`
        Instance of the `.ProfileData` dataclass containing the loaded profiles.
    """
    profiles = ProfileData()
    velocities = VelocityData()

    for name in profiles.__dataclass_fields__:
        match name:
            case "velocity":
                for name2 in velocities.__dataclass_fields__:
                    data = _get_profile(species_struct, name, grid_subset_index, name2=name2)
                    if data is None and backup_species_struct is not None:
                        data = _get_profile(
                            backup_species_struct, name, grid_subset_index, name2=name2
                        )
                    setattr(velocities, name2, data)
                setattr(profiles, name, velocities)
            case "species":
                # species data is not stored in the profile structure, skip loading it here
                continue
            case _:
                data = _get_profile(species_struct, name, grid_subset_index)
                if data is None and backup_species_struct is not None:
                    data = _get_profile(backup_species_struct, name, grid_subset_index)
                setattr(profiles, name, data)

    return profiles


def load_edge_species(ggd_struct: IDSStructure, grid_subset_index: int = 5) -> SpeciesComposition:
    """Load edge plasma species and their profiles from a given GGD structure.

    Parameters
    ----------
    ggd_struct
        The ggd ids structure containing the profiles.
    grid_subset_index
        Identifier index of the grid subset, by default 5 (``"Cells"``).

    Returns
    -------
    `.SpeciesComposition`
        Instance of the `.SpeciesComposition` dataclass

    Raises
    ------
    RuntimeError
        If unable to determine the species due to missing element information.
    """
    composition = SpeciesComposition(
        electron=load_edge_profiles(ggd_struct.electrons, grid_subset_index),
    )

    # Temporary sets
    ion_elements = []
    ion_uuids = set()
    ion_bundle_uuids = set()
    neutral_uuids = set()
    neutral_bundle_uuids = set()

    # ------------
    # === Ions ===
    # ------------
    for ion in ggd_struct.ion:
        ion: IDSStructure
        elements = get_elements(ion.element)
        if not len(elements):
            raise RuntimeError("Unable to determine the ion species, ion.element AOS is empty.")
        ion_elements.append(elements)

        # ---------------
        # === Bundles ===
        # ---------------
        if len(ion.state):
            shared_temperature = _get_profile(ion, "temperature", grid_subset_index)

            # Use the ion-level profiles as backup if there is only one state, otherwise no backup
            backup_ids = None if len(ion.state) > 1 else ion

            for i, state in enumerate(ion.state):
                uuid, species_data = get_ion_state(state, i, elements, grid_subset_index)
                if uuid in ion_bundle_uuids:
                    print(f"Warning! Skipping duplicated ion: {species_data.name}")
                    continue
                ion_bundle_uuids.add(uuid)

                profile_data = load_edge_profiles(state, grid_subset_index, backup_ids)
                profile_data.species = species_data

                if backup_ids is None and profile_data.temperature is None:
                    profile_data.temperature = shared_temperature
                if species_data.species_type == SpeciesType.ION_BUNDLE:
                    composition.ion_bundle.append(profile_data)
                elif species_data.species_type == SpeciesType.ION:
                    composition.ion.append(profile_data)
                else:
                    print(
                        f"Warning! Skipping ion with unexpected species type "
                        f"{species_data.species_type}: {species_data.name}"
                    )

        # -------------------
        # === Non-bundled ===
        # -------------------
        else:
            uuid, species_data = get_ion(ion, elements)
            if uuid in ion_uuids:
                print(f"Warning! Skipping duplicated ion: {species_data.name}")
            else:
                profile_data = load_edge_profiles(ion, grid_subset_index)
                profile_data.species = species_data
                if species_data.species_type == SpeciesType.ION:
                    composition.ion.append(profile_data)
                    ion_uuids.add(uuid)
                else:
                    print(
                        f"Warning! Skipping non-bundled ion with unexpected species type "
                        f"{species_data.species_type}: {species_data.name}"
                    )

    # ----------------------------
    # === Neutrals (molecules) ===
    # ----------------------------
    for neutral in ggd_struct.neutral:
        elements = get_elements(neutral.element)
        if not len(elements):
            elements = ion_elements[neutral.ion_index - 1]

        # ---------------
        # === Bundles ===
        # ---------------
        if len(neutral.state):
            shared_temperature = _get_profile(neutral, "temperature", grid_subset_index)

            # Use the neutral-level profiles as backup if there is only one state, otherwise no backup
            backup_ids = None if len(neutral.state) > 1 else neutral

            for state in neutral.state:
                uuid, species_data = get_neutral_state(state, elements)
                if uuid in neutral_bundle_uuids:
                    print(f"Warning! Skipping duplicated neutral: {species_data.name}")
                    continue
                neutral_bundle_uuids.add(uuid)

                profile_data = load_edge_profiles(state, grid_subset_index, backup_ids)
                profile_data.species = species_data

                if backup_ids is None and profile_data.temperature is None:
                    profile_data.temperature = shared_temperature

                if species_data.species_type == SpeciesType.MOLECULAR_BUNDLE:
                    composition.molecular_bundle.append(profile_data)
                elif species_data.species_type == SpeciesType.MOLECULE:
                    composition.molecule.append(profile_data)
                elif species_data.species_type == SpeciesType.NEUTRAL_BUNDLE:
                    composition.neutral_bundle.append(profile_data)
                elif species_data.species_type == SpeciesType.NEUTRAL:
                    composition.neutral.append(profile_data)
                else:
                    print(
                        f"Warning! Skipping neutral with unexpected species type "
                        f"{species_data.species_type}: {species_data.name}"
                    )

        # -------------------
        # === Non-bundled ===
        # -------------------
        else:
            uuid, species_data = get_neutral(neutral, elements)
            if uuid in neutral_uuids:
                print(f"Warning! Skipping duplicated neutral: {species_data.name}")
            else:
                profile_data = load_edge_profiles(neutral, grid_subset_index, None)
                profile_data.species = species_data
                if species_data.species_type == SpeciesType.MOLECULE:
                    composition.molecule.append(profile_data)
                    neutral_uuids.add(uuid)
                elif species_data.species_type == SpeciesType.NEUTRAL:
                    composition.neutral.append(profile_data)
                    neutral_uuids.add(uuid)
                else:
                    print(
                        f"Warning! Skipping non-bundled neutral with unexpected species type "
                        f"{species_data.species_type}: {species_data.name}"
                    )

    # Replace missing species temperature with average ion temperature
    t_ion = _get_profile(ggd_struct, "t_i_average", grid_subset_index)
    if t_ion is not None:
        species_types = set(composition.__dataclass_fields__.keys())
        species_types.remove("electron")
        for species_type in species_types:
            for profile in getattr(composition, species_type):
                if getattr(profile, "temperature", None) is None:
                    print(
                        "Warning! Using average ion temperature for "
                        f"the {species_type} {profile.species.name}."
                    )
                    profile.temperature = t_ion

    return composition
