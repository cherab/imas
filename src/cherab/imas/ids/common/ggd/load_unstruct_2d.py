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
"""Module for loading unstructured 2D grids from IMAS grid_ggd IDS structure."""

from enum import IntEnum
from typing import Final, Literal, cast, overload

import numpy as np
from numpy.typing import NDArray

from imas.ids_structure import IDSStructArray, IDSStructure

from ....ggd.unstruct_2d_mesh import UnstructGrid2D

__all__ = ["load_unstruct_grid_2d"]

_KNOWN_2D_SUBSET_IDS: Final[frozenset[int]] = frozenset({5, 22, 23, 24, 25, 38, 39, 40})
"""Set of known 2D grid subset indices in IMAS grid_ggd structure.

The detailed description of each subset can be found in the IMAS data dictionary:
https://imas-data-dictionary.readthedocs.io/en/latest/generated/identifier/ggd_subset_identifier.html
"""


class DIMENSION(IntEnum):
    """Enumeration for grid dimensions."""

    VERTEX = 0
    EDGE = 1
    FACE = 2


@overload
def load_unstruct_grid_2d(
    grid_ggd: IDSStructure, space_index: int = 0, with_subsets: Literal[False] = False
) -> UnstructGrid2D: ...


@overload
def load_unstruct_grid_2d(
    grid_ggd: IDSStructure,
    space_index: int = 0,
    *,
    with_subsets: Literal[True],
) -> tuple[
    UnstructGrid2D, dict[str, tuple[NDArray[np.int32], NDArray[np.bool_]]], dict[str, int]
]: ...


def load_unstruct_grid_2d(
    grid_ggd: IDSStructure, space_index: int = 0, with_subsets: bool = False
) -> (
    UnstructGrid2D
    | tuple[UnstructGrid2D, dict[str, tuple[NDArray[np.int32], NDArray[np.bool_]]], dict[str, int]]
):
    """Load unstructured 2D grid from the grid_ggd structure.

    Parameters
    ----------
    grid_ggd
        The grid_ggd structure.
    space_index
        The index of the grid space, by default 0.
    with_subsets
        Read grid subset data, by default False.

    Returns
    -------
    grid : `.UnstructGrid2D`
        Unstructured 2D grid object.
    subsets : `dict[str, tuple[NDArray[np.int32], NDArray[np.bool_]]]`
        Dictionary with grid subsets for each subset name containing a tuple with the indices of the
        cells from that subset and a boolean array indicating the validity of each index.
        Note that 'Cells' subset is included only if cell indices are specified.
    subset_id : `dict[str, int]`
        Dictionary with grid subset indices.

    Raises
    ------
    ValueError
        If the specified space does not contain a 2D grid.
    """
    space = grid_ggd.space[space_index]

    # Check if the grid is 2D
    if len(space.objects_per_dimension) != 3:
        raise ValueError("The load_unstruct_grid_2d() supports only unstructured 2D grids.")

    grid_name = str(grid_ggd.identifier.name)

    # Reading vertices
    num_vert = len(space.objects_per_dimension[DIMENSION.VERTEX].object)
    vertices = np.empty((num_vert, 2), dtype=np.float64)
    for i in range(num_vert):
        vertices[i] = space.objects_per_dimension[DIMENSION.VERTEX].object[i].geometry[:2]

    # Reading polygonal cells
    faces = cast(IDSStructArray, space.objects_per_dimension[DIMENSION.FACE].object)
    num_faces = len(faces)
    cells = []
    # ``cells`` is compact (invalid faces are omitted), while GGD subset
    # references are expressed in the original face numbering.
    face_to_cell = np.full(num_faces, -1, dtype=np.int32)
    valid_face = np.ones(num_faces, dtype=bool)
    winding_ok = True

    for i_face in range(num_faces):
        face = faces[i_face]

        if not face.has_value or not face.nodes.has_value:
            valid_face[i_face] = False
            continue
        # Convert every face from Fortran to C indexing. Triangular faces are already ordered.
        # Only polygons need their winding reconstructed below.
        cell = np.asarray_chkfinite(face.nodes, dtype=np.int32) - 1
        if cell.size < 3:
            valid_face[i_face] = False
            continue

        if cell.size > 3:
            # trying to get the nodes in winding order by parsing the edges
            edge_dict: dict[int, list[int]] = {}
            for boundary in face.boundary:
                n1, n2 = (
                    space.objects_per_dimension[DIMENSION.EDGE].object[boundary.index - 1].nodes - 1
                )  # Fortran to C indexing
                if n1 not in cell or n2 not in cell:  # fail, error in the data
                    edge_dict = {}
                    break
                if n1 in edge_dict:
                    if n2 == edge_dict[n1][0]:  # fail, error in the data
                        edge_dict = {}
                        break
                    edge_dict[n1][1] = n2
                else:
                    edge_dict[n1] = [n2, -1]
                if n2 in edge_dict:
                    if n1 == edge_dict[n2][0]:  # fail, error in the data
                        edge_dict = {}
                        break
                    edge_dict[n2][1] = n1
                else:
                    edge_dict[n2] = [n1, -1]

            if len(edge_dict) == cell.size:  # success, getting the cell nodes in winding order
                cell1 = np.empty(len(edge_dict), dtype=np.int32)
                cell1[0] = cell[0]
                pair = edge_dict[cell1[0]]
                cell1[1] = cell[1] if cell[1] in pair else pair[0]
                for i in range(2, cell1.size):
                    pair = edge_dict[cell1[i - 1]]
                    cell1[i] = pair[1] if cell1[i - 2] == pair[0] else pair[0]
                cell = cell1
            else:
                winding_ok = False

        face_to_cell[i_face] = len(cells)
        cells.append(cell)

    if not winding_ok:
        print("Warning! Unable to verify that the cell nodes are in the winding order.")

    grid = UnstructGrid2D(vertices, cells, valid_face, name=grid_name)

    if not with_subsets:
        return grid

    # Reading grid subsets (2D only)
    subsets: dict[str, tuple[NDArray[np.int32], NDArray[np.bool_]]] = {}
    subset_id: dict[str, int] = {}
    for subset in grid_ggd.grid_subset:
        subset_index: int = subset.identifier.index.value
        dimension_is_2d: bool = subset.dimension == DIMENSION.FACE + 1  # C to Fortran indexing
        known_subset_id: bool = subset_index in _KNOWN_2D_SUBSET_IDS
        if (dimension_is_2d or known_subset_id) and len(subset.element):
            name = str(subset.identifier.name)
            num_elm = len(subset.element)
            indices = np.empty(num_elm, dtype=np.int32)
            valid_subset = np.ones_like(indices, dtype=bool)
            for i_elm, element in enumerate(subset.element):
                if len(element.object) > 1:
                    print(
                        f"Warning! Skipping grid subset {name}, "
                        + "because it includes cells not present in the original grid."
                    )
                    break
                face_index = element.object[0].index.value - 1  # Fortran to C indexing
                if face_index < 0 or face_index >= num_faces:
                    valid_subset[i_elm] = False
                    indices[i_elm] = -1
                    continue
                indices[i_elm] = face_to_cell[face_index]
                if indices[i_elm] < 0:
                    valid_subset[i_elm] = False
            else:
                subsets[name] = (indices[valid_subset], valid_subset)
                subset_id[name] = subset_index

    return grid, subsets, subset_id
