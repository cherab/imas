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

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from imas.ids_structure import IDSStructure

from ..common.species import (
    ProfileData,
    SpeciesComposition,
    SpeciesType,
    get_elements,
    get_ion,
    get_ion_state,
    get_neutral,
    get_neutral_state,
)

__all__ = [
    "GridData",
    "load_core_grid",
    "load_core_profiles",
    "load_core_species",
]


@dataclass
class GridData:
    """Dataclass for storing grid properties of the core profiles."""

    rho_tor_norm: NDArray[np.float64] | None = None
    """Normalized toroidal flux coordinate."""
    psi: NDArray[np.float64] | None = None
    """Toroidal flux [Wb]."""
    volume: NDArray[np.float64] | None = None
    """Volume enclosed by the flux surface [m^3]."""
    area: NDArray[np.float64] | None = None
    """Area of the flux surface [m^2]."""
    surface: NDArray[np.float64] | None = None
    """Surface-averaged value of the profile on the flux surface."""


def _get_profile(ids_struct: IDSStructure, name: str):
    arr = getattr(ids_struct, name, None)
    if arr is not None and len(arr):
        return np.asarray(arr)
    else:
        return None


def load_core_grid(grid_struct: IDSStructure) -> GridData:
    """Load grid properties of the core profiles.

    The returned dictionary values for missing data are None.

    Parameters
    ----------
    grid_struct
        The IDS structure containing the grid data for 1D profiles.

    Returns
    -------
    `.GridData`
        Instance of the `GridData` dataclass containing the grid properties for the core profiles.
    """
    grid = GridData()
    for name in grid.__dataclass_fields__:
        setattr(grid, name, _get_profile(grid_struct, name))

    return grid


def load_core_profiles(
    species_struct: IDSStructure, backup_species_struct: IDSStructure | None = None
) -> ProfileData:
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
    `.ProfileData`
        Instance of the `ProfileData` dataclass containing the core profiles for the species.
    """
    profiles = ProfileData()

    for name in profiles.__dataclass_fields__:
        setattr(profiles, name, _get_profile(species_struct, name))
        if getattr(profiles, name) is None and backup_species_struct is not None:
            setattr(profiles, name, _get_profile(backup_species_struct, name))

    return profiles


def load_core_species(profile_1d: IDSStructure) -> SpeciesComposition:
    """Load core plasma species and their profiles from a given profiles IDS structure.

    Parameters
    ----------
    profile_1d
        The IDS structure containing the core profiles data.

    Returns
    -------
    `.SpeciesComposition`
        Instance of the `.SpeciesComposition` dataclass

    Raises
    ------
    RuntimeError
        If unable to determine the ion species due to missing element information.
    """
    composition = SpeciesComposition(
        electron=load_core_profiles(profile_1d.electrons),
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
    for ion in profile_1d.ion:
        ion: IDSStructure
        elements = get_elements(ion.element)
        if not len(elements):
            raise RuntimeError("Unable to determine the ion species, ion.element AOS is empty.")
        ion_elements.append(elements)

        # ---------------
        # === Bundles ===
        # ---------------
        if len(ion.state):
            shared_temperature = _get_profile(ion, "temperature")

            # Use the ion-level profiles as backup if there is only one state, otherwise no backup
            backup_ids = None if len(ion.state) > 1 else ion

            for i, state in enumerate(ion.state):
                uuid, species_data = get_ion_state(state, i, elements)
                if uuid in ion_bundle_uuids:
                    print(f"Warning! Skipping duplicated ion: {species_data.name}")
                    continue
                ion_bundle_uuids.add(uuid)

                profile_data = load_core_profiles(state, backup_ids)
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
                profile_data = load_core_profiles(ion)
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
    for neutral in profile_1d.neutral:
        elements = get_elements(neutral.element)
        if not len(elements):
            elements = ion_elements[neutral.ion_index - 1]

        # ---------------
        # === Bundles ===
        # ---------------
        if len(neutral.state):
            shared_temperature = _get_profile(neutral, "temperature")

            # Use the neutral-level profiles as backup if there is only one state, otherwise no backup
            backup_ids = None if len(neutral.state) > 1 else neutral

            for state in neutral.state:
                uuid, species_data = get_neutral_state(state, elements)
                if uuid in neutral_bundle_uuids:
                    print(f"Warning! Skipping duplicated neutral: {species_data.name}")
                    continue
                neutral_bundle_uuids.add(uuid)

                profile_data = load_core_profiles(state, backup_ids)
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
                profile_data = load_core_profiles(neutral)
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
    t_ion = profile_1d.t_i_average
    if len(t_ion):
        species_types = set(composition.__dataclass_fields__.keys())
        species_types.remove("electron")
        for species_type in species_types:
            for profile in getattr(composition, species_type):
                if getattr(profile, "temperature", None) is None:
                    print(
                        "Warning! Using average ion temperature for "
                        f"the {species_type} {profile.name}."
                    )
                    profile.temperature = t_ion

    return composition
