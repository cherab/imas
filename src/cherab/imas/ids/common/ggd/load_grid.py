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
"""Module for loading GGD grids from IMAS grid_ggd IDS structures."""

from __future__ import annotations

from typing import Literal, cast, overload

from numpy import int32
from numpy.typing import NDArray

from imas.db_entry import DBEntry
from imas.ids_struct_array import IDSStructArray
from imas.ids_structure import IDSStructure
from imas.ids_toplevel import IDSToplevel

from ....ggd.unstruct_2d_extend_mesh import UnstructGrid2DExtended
from ....ggd.unstruct_2d_mesh import UnstructGrid2D
from ..path import load_ids_path_reference, resolve_ids_path_reference
from .load_unstruct_2d import load_unstruct_grid_2d
from .load_unstruct_3d import load_unstruct_grid_2d_extended

__all__ = ["load_grid"]


@overload
def load_grid(
    grid_ggd: IDSStructure,
    with_subsets: Literal[False] = False,
    num_toroidal: int | None = None,
    *,
    entry: DBEntry | None = None,
) -> UnstructGrid2D | UnstructGrid2DExtended: ...


@overload
def load_grid(
    grid_ggd: IDSStructure,
    with_subsets: Literal[True],
    num_toroidal: int | None = None,
    *,
    entry: DBEntry | None = None,
) -> tuple[UnstructGrid2D, dict[str, NDArray[int32]], dict[str, int]]: ...


def load_grid(
    grid_ggd: IDSStructure,
    with_subsets: bool = False,
    num_toroidal: int | None = None,
    *,
    entry: DBEntry | None = None,
) -> (
    UnstructGrid2D
    | tuple[UnstructGrid2D, dict[str, NDArray[int32]], dict[str, int]]
    | UnstructGrid2DExtended
):
    """Load grid from the ``grid_ggd`` structure.

    The ``grid_ggd`` structure is expected to follow the IMAS GGD grid definition.
    Please see: https://imas-data-dictionary.readthedocs.io/en/latest/ggd_guide/doc.html#the-grid-ggd-aos

    .. warning::
        This function currently supports only unstructured 2D grids and unstructured 2D grids
        extended in 3D (mainly used in JOREK).
        Loading of structured grids and unstructured 3D grids will be implemented in the future.

    Parameters
    ----------
    grid_ggd
        The ``grid_ggd`` structure.
    with_subsets
        Read grid subset data, by default is False.
    num_toroidal
        Number of toroidal points, by default None.
    entry
        Open IMAS data entry used to resolve ``grid_ggd.path`` references to external IDSs.

    Returns
    -------
    grid : `.UnstructGrid2D` | `.UnstructGrid2DExtended`
        Grid object that depends on the grid type (2D/2D-extended/3D).
    subsets : `dict[str, NDArray[int32]]`
        Dictionary with grid subsets for each subset name containing the indices of the cells from
        that subset. Note that 'Cells' subset is included only if cell indices are specified.
    subset_id : `dict[str, int]`
        Dictionary with grid subset indices.

    Raises
    ------
    RuntimeError
        If the grid type is unsupported or if no spaces are found in the ``grid_ggd`` structure.
    NotImplementedError
        If the grid type is not yet implemented.

    Examples
    --------
    .. code-block:: python

        from imas import DBEntry
        from cherab.imas.ids.common import get_ids_time_slice
        from cherab.imas.ids.common.ggd import load_grid

        with DBEntry("imas:hdf5?path=/work/imas/shared/imasdb/ITER/4/123356/1", "r") as entry:
            ids = get_ids_time_slice(entry, "edge_profiles", 0)

        grid, subsets, subset_id = load_grid(ids.grid_ggd[0], with_subsets=True)
    """
    grid_source = _resolve_grid_source(grid_ggd, entry=entry)
    spaces = get_standard_spaces(grid_source)

    if not len(spaces):
        raise RuntimeError("GGD grid contain no spaces.")

    if len(spaces) == 1:  # simple unstructured grids
        if len(spaces[0].objects_per_dimension) == 3:  # 2D case
            return load_unstruct_grid_2d(grid_source, 0, with_subsets=with_subsets)
        if len(spaces[0].objects_per_dimension) == 4:  # 3D case
            raise NotImplementedError(
                "Loading unstructured 3D grids will be implemented in the future."
            )

        raise RuntimeError("Unsupported grid type.")

    if len(spaces) == 2:  # 2D structured grid or 2D unstructured grid extended in 3D
        if len(spaces[0].objects_per_dimension) == 3 and len(spaces[1].objects_per_dimension) < 3:
            return load_unstruct_grid_2d_extended(
                grid_source, with_subsets=with_subsets, num_toroidal=num_toroidal
            )
        if len(spaces[0].objects_per_dimension) < 3 and len(spaces[1].objects_per_dimension) < 3:
            raise NotImplementedError(
                "Loading structured 2D grids will be implemented in the future."
            )

        raise RuntimeError("Unsupported grid type.")

    if len(spaces) == 3:  # 3D structured grid
        if (
            len(spaces[0].objects_per_dimension) < 3
            and len(spaces[1].objects_per_dimension) < 3
            and len(spaces[2].objects_per_dimension) < 3
        ):
            raise NotImplementedError(
                "Loading structured 3D grids will be implemented in the future."
            )

        raise RuntimeError("Unsupported grid type.")

    raise RuntimeError("Unsupported grid type.")


def _resolve_grid_source(grid_ggd: IDSStructure, entry: DBEntry | None = None) -> IDSStructure:
    if len(grid_ggd.space):
        return grid_ggd

    if not len(grid_ggd.path):
        return grid_ggd

    path = str(grid_ggd.path).strip()
    if not path:
        return grid_ggd

    resolved: IDSToplevel | IDSStructure | IDSStructArray

    if "#" in path:
        if entry is None:
            raise RuntimeError(
                "Unable to resolve grid_ggd.path without a DBEntry. "
                + "Pass an open entry via load_grid(..., entry=entry)."
            )
        resolved = load_ids_path_reference(entry, path)
    else:
        root = _find_toplevel(grid_ggd)
        ids_name = root.metadata.name
        if path.startswith("/"):
            resolved = resolve_ids_path_reference(root, f"#{ids_name}{path}")
        else:
            resolved = resolve_ids_path_reference(root, path)

    if isinstance(resolved, IDSStructArray):
        if not len(resolved):
            raise RuntimeError(f"Resolved grid reference '{path}' points to an empty array.")
        resolved = resolved[0]

    if not hasattr(resolved, "space"):
        raise RuntimeError(
            f"Resolved grid reference '{path}' does not point to a grid_ggd structure."
        )

    return cast(IDSStructure, resolved)


def _find_toplevel(node: IDSStructure) -> IDSToplevel:
    current = node
    while hasattr(current, "_parent"):
        parent = current._parent
        if isinstance(parent, IDSToplevel):
            return parent
        if not hasattr(parent, "_parent"):
            break
        current = parent

    raise RuntimeError(
        "Unable to resolve local grid_ggd.path: cannot find the parent IDS toplevel object."
    )


def get_standard_spaces(grid_ggd: IDSStructure) -> list[IDSStructure]:
    """Get a list of standard non-empty spaces from the grid_ggd structure.

    Parameters
    ----------
    grid_ggd
        The grid_ggd structure.

    Returns
    -------
    list[IDSStructure]
        List of standard spaces.

    Raises
    ------
    RuntimeError
        If no spaces are defined in the grid_ggd structure.
    """
    if not len(grid_ggd.space):
        raise RuntimeError("Unable to read the grid. Grid space is not defined.")

    # Get list of standard spaces:
    spaces = []
    for space in grid_ggd.space:
        # JINTRAC confuses geometry_type with identifier, uncomment this check when the bug is fixed
        # if space.geometry_type.index == 0 or space.geometry_type.index == EMPTY_INT:
        if len(space.objects_per_dimension) and len(space.objects_per_dimension[0].object) > 2:
            spaces.append(space)

    return spaces
