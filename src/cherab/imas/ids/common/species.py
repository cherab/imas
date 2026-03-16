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

from __future__ import annotations

from dataclasses import astuple, dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from cherab.core import Isotope
from cherab.core.atomic.elements import Element, lookup_isotope
from imas.ids_defs import EMPTY_FLOAT, EMPTY_INT
from imas.ids_struct_array import IDSStructArray
from imas.ids_structure import IDSStructure

__all__ = [
    "SpeciesType",
    "SpeciesData",
    "ProfileData",
    "SpeciesComposition",
    "VelocityData",
    "get_ion_state",
    "get_neutral_state",
    "get_ion",
    "get_neutral",
    "get_elements",
]


class SpeciesType(StrEnum):
    """Enumeration of species types in IMAS."""

    ION = "ion"
    """Single ion state, with a specific ionization state (z_min == z_max)"""
    ION_BUNDLE = "ion_bundle"
    """Bundle of ion states, with a range of ionization states (z_min != z_max)"""
    NEUTRAL = "neutral"
    """Single neutral state"""
    NEUTRAL_BUNDLE = "neutral_bundle"
    """Bundle of neutral states"""
    MOLECULE = "molecule"
    """Single molecule state"""
    MOLECULAR_BUNDLE = "molecular_bundle"
    """Bundle of molecular states"""


@dataclass
class SpeciesData:
    """Dataclass to represent the data of a species in IMAS."""

    name: str
    """Name of the state"""
    z_min: float
    """Minimum ionization state of the species"""
    z_max: float
    """Maximum ionization state of the species"""
    element: Element | None = None
    """Element that makes up the species, if it is a single particle"""
    elements: tuple[Element, ...] = field(default_factory=tuple)
    """Elements that make up the species, if it is a molecule"""
    species_type: SpeciesType | None = None
    """Type of species"""
    electron_configuration: str | None = None
    """Electron configuration of the species"""
    vibrational_mode: str | None = None
    """Vibrational mode of the species, if it is a molecule"""
    vibrational_level: float | None = None
    """Vibrational level of the species, if it is a molecule"""


@dataclass
class VelocityData:
    """Dataclass for storing the bulk velocity data of a species."""

    radial: NDArray[np.float64] | None = None
    """Radial velocity [m/s]."""
    parallel: NDArray[np.float64] | None = None
    """Parallel velocity [m/s]."""
    poloidal: NDArray[np.float64] | None = None
    """Poloidal velocity [m/s]."""
    r: NDArray[np.float64] | None = None
    """Radial velocity along the major radius axis [m/s]."""
    phi: NDArray[np.float64] | None = None
    """Toroidal velocity [m/s]."""
    z: NDArray[np.float64] | None = None
    """Vertical velocity along the height axis [m/s]."""


@dataclass
class ProfileData:
    """Dataclass for storing the profile data of a species."""

    species: SpeciesData | None = None
    """Data of the species."""
    density: NDArray[np.float64] | None = None
    """Density (thermal+non-thermal) [m^-3]."""
    density_thermal: NDArray[np.float64] | None = None
    """Density (thermal) [m^-3]."""
    density_fast: NDArray[np.float64] | None = None
    """Density of fast (non-thermal) particles [m^-3]."""
    temperature: NDArray[np.float64] | None = None
    """Temperature [eV]."""
    velocity: VelocityData | None = None
    """Bulk velocity data of the species."""


@dataclass
class SpeciesComposition:
    """Dataclass for storing the composition of the plasma species."""

    electron: ProfileData
    """Electron profiles."""
    ion: list[ProfileData] = field(default_factory=list)
    """Ion profiles."""
    ion_bundle: list[ProfileData] = field(default_factory=list)
    """Ion bundle profiles."""
    neutral: list[ProfileData] = field(default_factory=list)
    """Neutral particle profiles."""
    neutral_bundle: list[ProfileData] = field(default_factory=list)
    """Neutral bundle profiles."""
    molecule: list[ProfileData] = field(default_factory=list)
    """Molecule profiles."""
    molecular_bundle: list[ProfileData] = field(default_factory=list)
    """Molecular bundle profiles."""


def get_ion_state(
    state: IDSStructure,
    state_index: int,
    elements: tuple[Element, ...],
    grid_subset_index: int | None = None,
) -> tuple[int, SpeciesData]:
    """Get a unique identifier for an ion state.

    Parameters
    ----------
    state
        IDSStructure representing `.../ion[i]/state`
    state_index
        Index of the state in the list of states.
    elements
        Tuple of elements that make up the ion state.
    grid_subset_index
        The grid subset index to use for 1D profiles, by default None.

    Returns
    -------
    uuid : int
        Unique identifier for the ion state, generated by hashing the `SpeciesData` dataclass.
    species_data : `.SpeciesData`
        Instance of the `SpeciesData` dataclass representing the ion state.
    """
    if state.z_min == EMPTY_FLOAT or state.z_max == EMPTY_FLOAT:
        if grid_subset_index is None:  # 1D profiles
            z_average = state.z_average_1d
        else:
            for s in state.z_average:
                if s.grid_subset_index == grid_subset_index:
                    z_average = s.values
                    break
            else:
                z_average = []
        if len(z_average):  # probably, a bundle
            z_min = float(state.z_min) if state.z_min != EMPTY_FLOAT else float(min(z_average))
            z_max = float(state.z_max) if state.z_max != EMPTY_FLOAT else float(max(z_average))
        else:  # probably, a single ion
            z_min = float(state.z_min) if state.z_min != EMPTY_FLOAT else state_index + 1.0
            z_max = float(state.z_max) if state.z_max != EMPTY_FLOAT else z_min
    else:
        z_min = float(state.z_min)
        z_max = float(state.z_max)

    # Initialize the state species dataclass
    species_data = SpeciesData(
        name=getattr(state, "name", "").strip(),
        z_min=z_min,
        z_max=z_max,
        electron_configuration=str(getattr(state, "electron_configuration", "")).strip()
        if len(getattr(state, "electron_configuration", "")) > 0
        else None,
    )

    if len(elements) > 1:  # molecular ions and bundles
        species_data.elements = elements
        if z_min == z_max == 0.0:
            species_data.species_type = SpeciesType.NEUTRAL_BUNDLE
        elif z_min == z_max:
            species_data.species_type = SpeciesType.MOLECULE
            species_data.vibrational_mode = (
                str(state.vibrational_mode) if len(state.vibrational_mode) else None
            )
            species_data.vibrational_level = (
                state.vibrational_level if state.vibrational_level != EMPTY_FLOAT else None
            )
        else:
            species_data.species_type = SpeciesType.MOLECULAR_BUNDLE
    else:  # ions and bundles
        species_data.element = elements[0]
        if z_min == z_max == 0.0:
            species_data.species_type = SpeciesType.NEUTRAL
        elif z_min == z_max:
            species_data.species_type = SpeciesType.ION
        else:
            species_data.species_type = SpeciesType.ION_BUNDLE

    return hash(astuple(species_data)), species_data


def get_neutral_state(
    state: IDSStructure, elements: tuple[Element, ...]
) -> tuple[int, SpeciesData]:
    """Get a unique identifier for a neutral state.

    Parameters
    ----------
    state
        The neutral_state structure from IMAS.
    elements
        Tuple of elements that make up the neutral state.

    Returns
    -------
    uuid : int
        Unique identifier for the neutral state, generated by hashing the `SpeciesData` dataclass.
    species_data : `.SpeciesData`
        Instance of the `SpeciesData` dataclass representing the neutral state.
    """
    # Initialize the state species dataclass
    species_data = SpeciesData(
        name=getattr(state, "name", "").strip(),
        z_min=0.0,
        z_max=0.0,
        electron_configuration=str(getattr(state, "electron_configuration", "")).strip()
        if len(getattr(state, "electron_configuration", "")) > 0
        else None,
    )
    if len(elements) > 1:  # molecules and bundles
        species_data.elements = elements
        if (
            getattr(state, "vibrational_mode", None)
            and getattr(state, "vibrational_level", EMPTY_FLOAT) != EMPTY_FLOAT
        ):
            species_data.species_type = SpeciesType.MOLECULE
            species_data.vibrational_mode = (
                str(getattr(state, "vibrational_mode", "")).strip()
                if len(getattr(state, "vibrational_mode", ""))
                else None
            )
            species_data.vibrational_level = (
                getattr(state, "vibrational_level", EMPTY_FLOAT)
                if getattr(state, "vibrational_level", EMPTY_FLOAT) != EMPTY_FLOAT
                else None
            )
        else:
            species_data.species_type = SpeciesType.NEUTRAL_BUNDLE
    else:  # neutrals
        species_data.element = elements[0]
        species_data.species_type = SpeciesType.NEUTRAL

    return hash(astuple(species_data)), species_data


def get_ion(ion: IDSStructure, elements: tuple[Element, ...]) -> tuple[int, SpeciesData]:
    """Get a unique identifier for an ion or molecule.

    Parameters
    ----------
    ion
        The ion structure from IMAS.
    elements
        Tuple of elements that make up the ion.

    Returns
    -------
    uuid : int
        Unique identifier for the ion or molecule, generated by hashing the `SpeciesData` dataclass.
    species_data : `.SpeciesData`
        Instance of the `SpeciesData` dataclass representing the ion or molecule.
    """
    z_ion = int(ion.z_ion) if ion.z_ion != EMPTY_FLOAT else elements[0].atomic_number
    species_data = SpeciesData(
        name=getattr(ion, "name", "").strip(),
        z_min=z_ion,
        z_max=z_ion,
        element=elements[0] if len(elements) == 1 else None,
        elements=elements if len(elements) > 1 else tuple(),
        species_type=SpeciesType.MOLECULE if len(elements) > 1 else SpeciesType.ION,
    )

    return hash(astuple(species_data)), species_data


def get_neutral(neutral: IDSStructure, elements: tuple[Element, ...]) -> tuple[int, SpeciesData]:
    """Get a unique identifier for a neutral or molecule.

    Parameters
    ----------
    neutral
        The neutral structure from IMAS.
    elements
        Tuple of elements that make up the neutral state.

    Returns
    -------
    uuid : int
        Unique identifier for the neutral or molecule, generated by hashing the `SpeciesData` dataclass.
    species_data : `.SpeciesData`
        Instance of the `SpeciesData` dataclass representing the neutral or molecule.
    """
    species_data = SpeciesData(
        name=getattr(neutral, "name", "").strip(),
        z_min=0.0,
        z_max=0.0,
        element=elements[0] if len(elements) == 1 else None,
        elements=elements if len(elements) > 1 else tuple(),
        species_type=SpeciesType.MOLECULE if len(elements) > 1 else SpeciesType.NEUTRAL,
    )

    return hash(astuple(species_data)), species_data


def get_elements(elements_aos: IDSStructArray) -> tuple[Element | Isotope, ...]:
    """Get a tuple of elements from an IDS array of structures.

    Parameters
    ----------
    elements_aos
        Element IDS array of structures

    Returns
    -------
    tuple[`~cherab.core.atomic.elements.Element` | `~cherab.core.atomic.elements.Isotope`, ...]
        Tuple of elements that make up the species, with isotopes preferred over elements when possible.
    """
    elements = []
    for element in elements_aos:
        mass_number = int(round(element.a))
        zn = int(round(element.z_n))
        isotope = lookup_isotope(zn, number=mass_number)
        if int(round(isotope.element.atomic_weight)) == mass_number:
            # Prefer element over isotope
            isotope = isotope.element

        if getattr(element, "atoms_n", EMPTY_INT) == EMPTY_INT:
            atoms_n = 1
        else:
            atoms_n = int(round(getattr(element, "atoms_n", EMPTY_INT)))

        for _ in range(atoms_n):
            elements.append(isotope)

    return tuple(elements)
