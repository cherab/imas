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

from colorsys import hsv_to_rgb
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import get_args, get_origin, get_type_hints
from zlib import crc32

import numpy as np
from numpy.typing import NDArray
from rich.text import Text
from rich.tree import Tree

from cherab.core.atomic.elements import Element, Isotope, lookup_isotope
from imas.ids_defs import EMPTY_FLOAT, EMPTY_INT
from imas.ids_struct_array import IDSStructArray
from imas.ids_structure import IDSStructure

__all__ = [
    "SpeciesType",
    "SpeciesData",
    "ProfileData",
    "SpeciesComposition",
    "VelocityData",
    "select_profile_data",
    "get_ion_state",
    "get_neutral_state",
    "get_ion",
    "get_neutral",
    "get_elements",
]


def _element_style(symbol: str) -> str:
    """Return a stable bright style derived directly from an element symbol.

    The hue excludes the red sector and does not depend on the size or ordering of a palette.

    Returns
    -------
    str
        Rich style containing a deterministic true-color foreground.
    """
    hash_fraction = (crc32(symbol.encode()) >> 20) / 0xFFF
    hue = (35.0 + 290.0 * hash_fraction) / 360.0
    rgb = tuple(round(channel * 255) for channel in hsv_to_rgb(hue, 0.6, 1.0))
    return f"bold #{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


class SpeciesType(Enum):
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
    """Single molecular state; neutral or charged according to ``z_min`` and ``z_max``"""
    MOLECULAR_BUNDLE = "molecular_bundle"
    """Bundle of molecular states or charge states"""


@dataclass
class SpeciesData:
    """Dataclass to represent the data of a species in IMAS."""

    z_min: int
    """Minimum ionization state of the species"""
    z_max: int
    """Maximum ionization state of the species"""
    element: Element | Isotope | None = None
    """Element that makes up the species, if it is a single particle"""
    elements: tuple[Element | Isotope, ...] = field(default_factory=tuple)
    """Elements that make up the species, if it is a molecule"""
    species_type: SpeciesType | None = None
    """Type of species"""
    electron_configuration: str | None = None
    """Electron configuration of the species"""
    vibrational_mode: str | None = None
    """Vibrational mode of the species, if it is a molecule"""
    vibrational_level: float | None = None
    """Vibrational level of the species, if it is a molecule"""

    def __str__(self) -> str:
        """Return a string representation of the species data.

        Returns
        -------
        str
            String representation of the species data.
        """
        if self.species_type in {SpeciesType.ION, SpeciesType.NEUTRAL}:
            if self.element is not None:
                return f"{self.element.symbol} {self.species_type.value} (z=+{self.z_min})"
            else:
                return f"{self.species_type.value} (z=+{self.z_min})"
        elif self.species_type in {SpeciesType.ION_BUNDLE, SpeciesType.NEUTRAL_BUNDLE}:
            if self.element is not None:
                return (
                    f"{self.element.symbol} {self.species_type.value} (z={self.z_min}-{self.z_max})"
                )
            else:
                return f"{self.species_type.value} (z={self.z_min}-{self.z_max})"
        elif self.species_type == SpeciesType.MOLECULE:
            molecule = f"{'-'.join(el.symbol for el in self.elements)} {self.species_type.value}"
            return f"{molecule} (z=+{self.z_min})" if self.z_min else molecule
        elif self.species_type == SpeciesType.MOLECULAR_BUNDLE:
            return f"{'-'.join(el.symbol for el in self.elements)} {self.species_type.value} (z={self.z_min}-{self.z_max})"
        else:
            return "Unknown species type"

    def compact_label(self) -> str:
        """Return a compact label for display beneath a species-type group.

        Returns
        -------
        str
            Elemental or molecular symbol followed by charge information when non-zero.
        """
        symbol = self._symbol_label()
        if symbol is None:
            return "Unknown"

        if self.z_min != self.z_max:
            minimum = self._charge_label(self.z_min, omit_unit=False)
            maximum = self._charge_label(self.z_max, omit_unit=False)
            return f"{symbol} {minimum}–{maximum}"
        if self.z_min:
            return f"{symbol} {self._charge_label(self.z_min)}"
        return symbol

    def _symbol_label(self) -> str | None:
        """Return the elemental or molecular symbol used at the start of display labels.

        Returns
        -------
        str or None
            Elemental or molecular symbol, if available.
        """
        if self.element is not None:
            return self.element.symbol
        if self.elements:
            return "-".join(element.symbol for element in self.elements)
        return None

    @staticmethod
    def _charge_label(charge: int, *, omit_unit: bool = True) -> str:
        """Format a charge number as a linearized ionic charge.

        Returns
        -------
        str
            Charge magnitude followed by its sign, omitting a unit magnitude when requested.
        """
        magnitude = abs(charge)
        sign = "+" if charge >= 0 else "-"
        return sign if omit_unit and magnitude == 1 else f"{magnitude}{sign}"

    def rich_label(self, compact: bool = False) -> Text:
        """Return a Rich label colored consistently by element symbol.

        Parameters
        ----------
        compact
            Use the compact context-aware label instead of the standalone description.

        Returns
        -------
        `rich.text.Text`
            Species label with an isotope-derived color when element data is available.
        """
        label = self.compact_label() if compact else str(self)
        element = self.element or (self.elements[0] if self.elements else None)
        if element is None:
            return Text(label)
        return Text(label, style=_element_style(element.symbol))

    def __rich__(self) -> Text:
        """Return a Rich label colored consistently by element symbol.

        Charge states of the same element or isotope share a color, while different isotopes use
        distinct symbols and therefore different colors. Homonuclear molecules use the color of
        their constituent; heteronuclear molecules use the first constituent.

        Returns
        -------
        `rich.text.Text`
            Species label with an element-derived color when element data is available.
        """
        return self.rich_label()


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

    species: SpeciesData
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

    def array_shapes(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Return paths and shapes for every array stored in this profile.

        The dataclass hierarchy is traversed dynamically, so newly added array fields are
        included without changing this method.

        Returns
        -------
        tuple[tuple[str, tuple[int, ...]], ...]
            Field paths and corresponding array shapes.
        """
        shapes: list[tuple[str, tuple[int, ...]]] = []

        def collect(value: object, path: str = "") -> None:
            if isinstance(value, np.ndarray):
                shapes.append((path, value.shape))
            elif is_dataclass(value) and not isinstance(value, type):
                for data_field in fields(value):
                    child_path = f"{path}.{data_field.name}" if path else data_field.name
                    collect(getattr(value, data_field.name), child_path)

        collect(self)
        return tuple(shapes)


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
    """Neutral and charged molecule profiles."""
    molecular_bundle: list[ProfileData] = field(default_factory=list)
    """Molecular bundle profiles."""

    def _profile_groups(self) -> tuple[tuple[str, tuple[ProfileData, ...], bool], ...]:
        """Return profile groups discovered from the dataclass type annotations.

        Returns
        -------
        tuple[tuple[str, tuple[ProfileData, ...], bool], ...]
            Field names, stored profiles, and whether the field is a profile collection.
        """
        type_hints = get_type_hints(type(self))
        groups = []
        for data_field in fields(self):
            annotation = type_hints[data_field.name]
            value = getattr(self, data_field.name)
            is_profile = isinstance(annotation, type) and issubclass(annotation, ProfileData)
            list_args = get_args(annotation) if get_origin(annotation) is list else ()
            is_profile_list = (
                len(list_args) == 1
                and isinstance(list_args[0], type)
                and issubclass(list_args[0], ProfileData)
            )
            if is_profile:
                profiles = (value,)
                is_collection = False
            elif is_profile_list:
                profiles = tuple(value)
                is_collection = True
            else:
                continue
            groups.append((data_field.name, profiles, is_collection))

        return tuple(groups)

    def __str__(self) -> str:
        """Return a concise summary containing species names and profile shapes.

        Returns
        -------
        str
            Multiline summary of species names and array shapes grouped by type.
        """
        lines = [self.__class__.__name__]
        for group_name, profiles, is_collection in self._profile_groups():
            if not profiles:
                continue
            count = f" ({len(profiles)})" if len(profiles) > 1 else ""
            lines.append(f"  {group_name}{count}")
            for profile in profiles:
                if is_collection:
                    lines.append(f"    - {profile.species.compact_label()}")
                indent = "      " if is_collection else "    "
                lines.extend(
                    f"{indent}{path}: shape={shape}" for path, shape in profile.array_shapes()
                )

        return "\n".join(lines)

    def __rich__(self) -> Tree:
        """Return a Rich tree containing species names and profile shapes.

        Returns
        -------
        `rich.tree.Tree`
            Tree representation rendered by Rich-enabled output.
        """
        tree = Tree(Text(self.__class__.__name__, style="bold cyan"))
        for group_name, profiles, is_collection in self._profile_groups():
            if not profiles:
                continue
            count = f" ({len(profiles)})" if len(profiles) > 1 else ""
            branch = tree.add(Text(f"{group_name}{count}", style="bold"))
            for profile in profiles:
                parent = (
                    branch.add(profile.species.rich_label(compact=True))
                    if is_collection
                    else branch
                )
                for path, shape in profile.array_shapes():
                    parent.add(Text(f"{path}: shape={shape}", style="dim"))

        return tree


def select_profile_data(
    composition: SpeciesComposition,
    indices: NDArray[np.intp],
    source_size: int,
) -> None:
    """Select valid source entries from all one-dimensional profile arrays.

    Parameters
    ----------
    composition
        Species composition whose profile arrays are updated in place.
    indices
        Positions of valid source entries in the selected grid subset.
    source_size
        Number of entries in the source grid subset before invalid cells are removed.
    """
    profiles = [composition.electron]
    for group_name in (
        "ion",
        "ion_bundle",
        "neutral",
        "neutral_bundle",
        "molecule",
        "molecular_bundle",
    ):
        profiles.extend(getattr(composition, group_name))

    def select_profile(profile: ProfileData) -> None:
        for profile_field in fields(profile):
            value = getattr(profile, profile_field.name)
            if isinstance(value, np.ndarray) and value.ndim == 1 and value.size == source_size:
                setattr(profile, profile_field.name, value[indices])
            elif isinstance(value, VelocityData):
                for velocity_field in fields(value):
                    velocity = getattr(value, velocity_field.name)
                    if (
                        isinstance(velocity, np.ndarray)
                        and velocity.ndim == 1
                        and velocity.size == source_size
                    ):
                        setattr(value, velocity_field.name, velocity[indices])

    for profile in profiles:
        select_profile(profile)


def get_ion_state(
    state: IDSStructure,
    state_index: int,
    elements: tuple[Element, ...],
    grid_subset_index: int | None = None,
) -> SpeciesData:
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
    `.SpeciesData`
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
        if len(z_average):
            warning_msg = (
                f"Warning: z_min or z_max is EMPTY_FLOAT for state index {state_index}. "
                f"Using z_average to determine z_min and z_max."
            )
            z_min = (
                int(state.z_min) if state.z_min != EMPTY_FLOAT else int(np.floor(min(z_average)))
            )
            z_max = int(state.z_max) if state.z_max != EMPTY_FLOAT else int(np.ceil(max(z_average)))
        else:
            warning_msg = (
                f"Warning: z_min or z_max is EMPTY_FLOAT for state index {state_index}. "
                f"z_average is also empty. Using state_index + 1 as z_min and z_max."
            )
            z_min = int(state.z_min) if state.z_min != EMPTY_FLOAT else state_index + 1
            z_max = int(state.z_max) if state.z_max != EMPTY_FLOAT else z_min

        print(warning_msg)

    else:
        z_min = int(state.z_min)
        z_max = int(state.z_max)

    # Initialize the state species dataclass
    species_data = SpeciesData(
        z_min=z_min,
        z_max=z_max,
        electron_configuration=str(getattr(state, "electron_configuration", "")).strip()
        if len(getattr(state, "electron_configuration", "")) > 0
        else None,
    )

    if len(elements) > 1:  # molecular ions and bundles
        species_data.elements = elements
        if z_min == z_max:
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
            species_data.species_type = SpeciesType.MOLECULAR_BUNDLE
    else:  # ions and bundles
        species_data.element = elements[0]
        if z_min == z_max == 0:
            species_data.species_type = SpeciesType.NEUTRAL
        elif z_min == z_max:
            species_data.species_type = SpeciesType.ION
        else:
            species_data.species_type = SpeciesType.ION_BUNDLE

    return species_data


def get_neutral_state(state: IDSStructure, elements: tuple[Element, ...]) -> SpeciesData:
    """Get a unique identifier for a neutral state.

    Parameters
    ----------
    state
        The neutral_state structure from IMAS.
    elements
        Tuple of elements that make up the neutral state.

    Returns
    -------
    `.SpeciesData`
        Instance of the `SpeciesData` dataclass representing the neutral state.
    """
    # Initialize the state species dataclass
    species_data = SpeciesData(
        z_min=0,
        z_max=0,
        electron_configuration=str(getattr(state, "electron_configuration", "")).strip()
        if len(getattr(state, "electron_configuration", "")) > 0
        else None,
    )
    if len(elements) > 1:  # molecules
        species_data.elements = elements
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
    else:  # neutrals
        species_data.element = elements[0]
        species_data.species_type = SpeciesType.NEUTRAL

    return species_data


def get_ion(ion: IDSStructure, elements: tuple[Element, ...]) -> SpeciesData:
    """Get a unique identifier for an ion or molecule.

    Parameters
    ----------
    ion
        The ion structure from IMAS.
    elements
        Tuple of elements that make up the ion.

    Returns
    -------
    `.SpeciesData`
        Instance of the `SpeciesData` dataclass representing the ion or molecule.
    """
    z_ion = int(ion.z_ion) if ion.z_ion != EMPTY_FLOAT else elements[0].atomic_number
    species_data = SpeciesData(
        z_min=z_ion,
        z_max=z_ion,
        element=elements[0] if len(elements) == 1 else None,
        elements=elements if len(elements) > 1 else tuple(),
        species_type=SpeciesType.MOLECULE if len(elements) > 1 else SpeciesType.ION,
    )

    return species_data


def get_neutral(neutral: IDSStructure, elements: tuple[Element, ...]) -> SpeciesData:
    """Get a unique identifier for a neutral or molecule.

    Parameters
    ----------
    neutral
        The neutral structure from IMAS.
    elements
        Tuple of elements that make up the neutral state.

    Returns
    -------
    `.SpeciesData`
        Instance of the `SpeciesData` dataclass representing the neutral or molecule.
    """
    species_data = SpeciesData(
        z_min=0,
        z_max=0,
        element=elements[0] if len(elements) == 1 else None,
        elements=elements if len(elements) > 1 else tuple(),
        species_type=SpeciesType.MOLECULE if len(elements) > 1 else SpeciesType.NEUTRAL,
    )

    return species_data


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
        if round(isotope.element.atomic_weight) == mass_number:
            # Prefer element over isotope
            isotope = isotope.element

        if getattr(element, "atoms_n", EMPTY_INT) == EMPTY_INT:
            atoms_n = 1
        else:
            atoms_n = round(getattr(element, "atoms_n", EMPTY_INT), ndigits=None)

        for _ in range(atoms_n):
            elements.append(isotope)

    return tuple(elements)
