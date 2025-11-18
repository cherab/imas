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
"""Module for utility to load profiles."""

__all__ = ["warn_unsupported_species", "get_subset_name_index"]


def warn_unsupported_species(composition: dict[str, dict], species_type: str) -> None:
    """Warn if species of a given type are present in the composition dictionary.

    Parameters
    ----------
    composition
        Dictionary with species composition.
    species_type
        Type of species to check for (e.g., 'ion_bundle', 'molecular_bundle').
    """
    if species_type in composition and len(composition[species_type]):
        print(
            f"Warning! Species of type '{species_type}' are currently not supported. "
            + "The following species will be skipped:"
        )
        names = []
        for species_id in composition[species_type]:
            d = {first: second for first, second in species_id}
            names.append(d["name"])
        print("; ".join(names))


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
