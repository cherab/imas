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

from dataclasses import astuple

import numpy as np
from numpy.typing import NDArray

from cherab.core.atomic import AtomicData
from imas.ids_primitive import IDSNumericArray
from imas.ids_structure import IDSStructArray, IDSStructure

from ..common import solve_coronal_equilibrium
from ..common.species import (
    ProfileData,
    SpeciesComposition,
    SpeciesData,
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


def load_edge_species(
    ggd_struct: IDSStructure,
    grid_subset_index: int = 5,
    split_ion_bundles: bool = True,
    atomic_data: AtomicData | None = None,
) -> SpeciesComposition:
    """Load edge plasma species and their profiles from a given GGD structure.

    Parameters
    ----------
    ggd_struct
        The ggd ids structure containing the profiles.
    grid_subset_index
        Identifier index of the grid subset, by default 5 (``"Cells"``).
    split_ion_bundles
        Whether to split ion bundles into individual ion states using `.solve_coronal_equilibrium`,
        by default True.
    atomic_data
        Optional atomic data to pass to `.solve_coronal_equilibrium` when splitting ion bundles.

    Returns
    -------
    `.SpeciesComposition`
        Instance of the `.SpeciesComposition` dataclass

    Raises
    ------
    RuntimeError
        If electron temperature or density profiles are missing, which are required for determining
        the species composition and solving coronal equilibrium when splitting ion bundles.
    RuntimeError
        If unable to determine the species due to missing element information, density profiles,
        or other necessary data.
    """
    composition = SpeciesComposition(
        electron=load_edge_profiles(ggd_struct.electrons, grid_subset_index),
    )

    if composition.electron.temperature is None:
        raise RuntimeError("Electron temperature profiles are required.")

    electron_density = composition.electron.density_thermal
    if electron_density is None:
        electron_density = composition.electron.density
    if electron_density is None:
        raise RuntimeError("Electron density or density_thermal profiles are required.")

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
                if uuid in ion_bundle_uuids or uuid in ion_uuids:
                    print(f"Warning! Skipping duplicated ion: {species_data}")
                    continue

                profile_data = load_edge_profiles(state, grid_subset_index, backup_ids)
                if backup_ids is None and profile_data.temperature is None:
                    profile_data.temperature = shared_temperature
                profile_data.species = species_data

                # === Case: ION ===
                if species_data.species_type == SpeciesType.ION:
                    composition.ion.append(profile_data)
                    ion_uuids.add(uuid)

                # === Case: ION_BUNDLE ===
                elif species_data.species_type == SpeciesType.ION_BUNDLE:
                    # Split ion bundles into individual ion states
                    if split_ion_bundles:
                        # Check if the necessary data is available before attempting to solve coronal equilibrium
                        elm = species_data.element or species_data.elements[0]
                        if elm is None:
                            raise RuntimeError(
                                f"Unable to determine the element for ion {species_data}, "
                                "cannot solve coronal equilibrium to split ion bundle."
                            )
                        density = profile_data.density_thermal
                        if density is None:
                            density = profile_data.density
                        if density is None:
                            raise RuntimeError(
                                f"Missing density profiles for ion {species_data}, "
                                "cannot solve coronal equilibrium to split ion bundle."
                            )
                        try:
                            densities_per_charge = solve_coronal_equilibrium(
                                elm,
                                density,
                                electron_density,
                                composition.electron.temperature,
                                atomic_data=atomic_data,
                                z_min=species_data.z_min,
                                z_max=species_data.z_max,
                            )
                        except Exception as e:
                            print(
                                f"Skipping ion bundle {species_data} "
                                f"due to error in solving coronal equilibrium: {e}"
                            )
                            composition.ion_bundle.append(profile_data)
                            ion_bundle_uuids.add(uuid)
                            continue

                        charge_states = np.arange(
                            species_data.z_min, species_data.z_max + 1, dtype=int
                        )

                        for i_charge, charge in enumerate(charge_states):
                            species = SpeciesData(
                                element=species_data.element,
                                z_min=charge,
                                z_max=charge,
                                species_type=SpeciesType.ION,
                            )
                            composition.ion.append(
                                ProfileData(
                                    species=species,
                                    density=densities_per_charge[i_charge],
                                    temperature=profile_data.temperature,
                                    velocity=profile_data.velocity,
                                )
                            )
                            ion_uuids.add(hash(astuple(species)))

                    # Don't split ion bundles, just add the bundle as is
                    else:
                        composition.ion_bundle.append(profile_data)
                        ion_bundle_uuids.add(uuid)

                # === Case: Unexpected species type ===
                else:
                    print(f"Warning! Skipping ion with unexpected species {species_data}")

        # -------------------
        # === Non-bundled ===
        # -------------------
        else:
            uuid, species_data = get_ion(ion, elements)
            if uuid in ion_uuids:
                print(f"Warning! Skipping duplicated ion: {species_data}")
            else:
                profile_data = load_edge_profiles(ion, grid_subset_index)
                profile_data.species = species_data
                if species_data.species_type == SpeciesType.ION:
                    composition.ion.append(profile_data)
                    ion_uuids.add(uuid)
                else:
                    print(
                        f"Warning! Skipping non-bundled ion with unexpected species {species_data}"
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
                    print(f"Warning! Skipping duplicated neutral: {species_data}")
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
                    print(f"Warning! Skipping neutral with unexpected species: {species_data}")

        # -------------------
        # === Non-bundled ===
        # -------------------
        else:
            uuid, species_data = get_neutral(neutral, elements)
            if uuid in neutral_uuids:
                print(f"Warning! Skipping duplicated neutral: {species_data}")
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
                        f"Warning! Skipping non-bundled neutral with unexpected species: "
                        f"{species_data}"
                    )

    # Replace missing species temperature with average ion temperature
    t_ion = _get_profile(ggd_struct, "t_i_average", grid_subset_index)
    if t_ion is not None:
        species_types = set(composition.__dataclass_fields__.keys())
        species_types.remove("electron")
        for species_type in species_types:
            for profile in getattr(composition, species_type):
                if getattr(profile, "temperature", None) is None:
                    print(f"Warning! Using average ion temperature for the {profile.species}.")
                    profile.temperature = t_ion

    return composition
