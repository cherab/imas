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
"""Module for common utilities to load profiles."""

from dataclasses import dataclass
from typing import Literal

from raysect.core.math import Vector3D
from raysect.core.math.function.float import Function2D, Function3D
from raysect.core.math.function.vector3d import Constant3D as ConstantVector3D
from raysect.core.math.function.vector3d import Function2D as VectorFunction2D
from raysect.core.math.function.vector3d import Function3D as VectorFunction3D

from ..ids.common.species import SpeciesComposition

__all__ = [
    "ProfileInterporater",
    "warn_unsupported_species",
    "get_subset_name_index",
]


ZERO_VELOCITY = ConstantVector3D(Vector3D(0, 0, 0))


@dataclass
class ProfileInterporater:
    """Dataclass to hold the interpolators for profiles."""

    density: Function2D | Function3D | None = None
    """Interpolating function for the density profile. Prefer ``density_thermal`` if available."""
    density_thermal: Function2D | Function3D | None = None
    """Interpolating function for the thermal density profile."""
    temperature: Function2D | Function3D | None = None
    """Interpolating function for the temperature profile."""
    velocity: VectorFunction2D | VectorFunction3D | None = None
    """Interpolating function for the velocity profile."""


def warn_unsupported_species(
    composition: SpeciesComposition,
    species_type: Literal["ion_bundle", "molecule", "molecular_bundle"],
) -> None:
    """Warn if species of a given type are present in the composition dictionary.

    Parameters
    ----------
    composition
        Instance of the `.SpeciesComposition` dataclass
    species_type
        Type of species to check for (e.g., 'ion_bundle', 'molecule', 'molecular_bundle').
    """
    if hasattr(composition, species_type) and len(getattr(composition, species_type)) > 0:
        names: list[str] = []
        for profile_data in getattr(composition, species_type):
            name: str | None = getattr(profile_data.species, "name", None)
            if name is None:
                element = getattr(profile_data.species, "element", None)
                if element is None:
                    elements = getattr(profile_data.species, "elements", None)
                    if elements is None or len(elements) == 0:
                        name = "Unknown"
                    else:
                        name = "".join([e.name for e in elements])
                else:
                    name = element.name

            names.append(name)

        print(
            f"Warning! Species of type '{species_type}' are currently not supported.\n"
            + f"The following species will be skipped: {'; '.join(names)}"
        )


def get_subset_name_index(subset_id_dict: dict, grid_subset_id: int | str) -> tuple[str, int]:
    """Get the name and index of a grid subset from its identifier.

    Parameters
    ----------
    subset_id_dict
        Dictionary with grid subset indices.
    grid_subset_id
        Identifier of the grid subset. Either index or name.

    Returns
    -------
    grid_subset_name
        Name of the grid subset.
    grid_subset_index
        Index of the grid subset.

    Raises
    ------
    ValueError
        If the grid subset with the given identifier is not found.
    """
    subset_id = subset_id_dict.copy()
    subset_id.update({value: key for key, value in subset_id.items()})

    try:
        grid_subset_index = int(grid_subset_id)
        grid_subset_name = subset_id[grid_subset_index]
    except ValueError as err1:
        try:
            grid_subset_name = str(grid_subset_id)
            grid_subset_index = subset_id[grid_subset_name]
        except KeyError:
            raise ValueError(f"Unable to find a grid subset with ID {grid_subset_id}.") from err1
    except KeyError as err2:
        raise ValueError(f"Unable to find a grid subset with ID {grid_subset_id}.") from err2

    return grid_subset_name, grid_subset_index
