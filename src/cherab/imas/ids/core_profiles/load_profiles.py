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

from dataclasses import astuple, dataclass, fields

import numpy as np
from numpy.typing import NDArray

from cherab.core.atomic import AtomicData
from imas.ids_primitive import IDSNumericArray
from imas.ids_structure import IDSStructure

from ..common import solve_coronal_equilibrium
from ..common.species import (
    ProfileData,
    SpeciesComposition,
    SpeciesData,
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


def _get_profile(ids_struct: IDSStructure, name: str, name2: str | None = None):
    data = getattr(ids_struct, name, None)
    if isinstance(data, IDSNumericArray):
        if len(data):
            return np.asarray(data)
        else:
            return None
    elif isinstance(data, IDSStructure):
        # Try to search lower-level structure
        return _get_profile(data, name2) if name2 is not None else None
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
        Instance of the `.GridData` dataclass containing the grid properties for the core profiles.
    """
    grid = GridData()
    for field in fields(grid):
        setattr(grid, field.name, _get_profile(grid_struct, field.name))

    return grid


def load_core_profiles(
    species_struct: IDSStructure,
    species: SpeciesData,
    backup_species_struct: IDSStructure | None = None,
) -> ProfileData:
    """Load core profiles from a given species structure.

    Parameters
    ----------
    species_struct
        IDS structure containing the profiles for a single species.
    species
        The species data for the profiles to be loaded.
    backup_species_struct
        The backup ids structure that is used if the profile is missing in species_struct,
        by default None.

    Returns
    -------
    `.ProfileData`
        Instance of the `ProfileData` dataclass containing the core profiles for the species.
    """
    profiles = ProfileData(species)

    for field in fields(profiles):
        setattr(profiles, field.name, _get_profile(species_struct, field.name))
        if getattr(profiles, field.name) is None and backup_species_struct is not None:
            setattr(profiles, field.name, _get_profile(backup_species_struct, field.name))

    return profiles


def load_core_species(
    profile_1d: IDSStructure,
    split_ion_bundles: bool = True,
    atomic_data: AtomicData | None = None,
) -> SpeciesComposition:
    """Load core plasma species and their profiles from a given profiles IDS structure.

    Parameters
    ----------
    profile_1d
        The IDS structure containing the core profiles data.
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
        electron=load_core_profiles(profile_1d.electrons, SpeciesData(z_min=-1, z_max=-1)),
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
                if uuid in ion_bundle_uuids or uuid in ion_uuids:
                    print(f"Warning! Skipping duplicated ion: {species_data}")
                    continue

                profile_data = load_core_profiles(
                    state,
                    species_data,
                    backup_species_struct=backup_ids,
                )

                if backup_ids is None and profile_data.temperature is None:
                    profile_data.temperature = shared_temperature

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
                                    density_thermal=densities_per_charge[i_charge],
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
                profile_data = load_core_profiles(ion, species_data)
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
                    print(f"Warning! Skipping duplicated neutral: {species_data}")
                    continue
                neutral_bundle_uuids.add(uuid)

                profile_data = load_core_profiles(
                    state, species_data, backup_species_struct=backup_ids
                )

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
                    print(f"Warning! Skipping neutral with unexpected species {species_data}")

        # -------------------
        # === Non-bundled ===
        # -------------------
        else:
            uuid, species_data = get_neutral(neutral, elements)
            if uuid in neutral_uuids:
                print(f"Warning! Skipping duplicated neutral: {species_data}")
            else:
                profile_data = load_core_profiles(neutral, species_data)
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
    t_ion = _get_profile(profile_1d, "t_i_average")
    if t_ion is not None and len(t_ion):
        species_types = {field.name for field in fields(composition)}
        species_types.remove("electron")
        for species_type in species_types:
            for profile in getattr(composition, species_type):
                if getattr(profile, "temperature", None) is None:
                    print(f"Warning! Using average ion temperature for {profile.species}.")
                    profile.temperature = t_ion

    return composition
