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

import numpy as np
from numpy.typing import NDArray

from imas.ids_defs import EMPTY_INT
from imas.ids_structure import IDSStructure

from ....ggd.unstruct_2d_mesh import UnstructGrid2D

__all__ = ["load_unstruct_grid_2d"]

VERTEX_DIMENSION = 0
EDGE_DIMENSION = 1
FACE_DIMENSION = 2


def load_unstruct_grid_2d(
    grid_ggd: IDSStructure, space_index: int = 0, with_subsets: bool = False
) -> UnstructGrid2D | tuple[UnstructGrid2D, dict[str, NDArray[np.int32]], dict[str, int]]:
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
    subsets : `dict[str, NDArray[numpy.int32]]`
        Dictionary with grid subsets for each subset name containing the indices of the cells from
        that subset. Note that 'Cells' subset is included only if cell indices are specified.
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
    num_vert = len(space.objects_per_dimension[VERTEX_DIMENSION].object)
    vertices = np.empty((num_vert, 2), dtype=np.float64)
    for i in range(num_vert):
        vertices[i] = space.objects_per_dimension[VERTEX_DIMENSION].object[i].geometry[:2]

    # Reading polygonal cells
    cells = []
    winding_ok = True
    for object in space.objects_per_dimension[FACE_DIMENSION].object:
        # getting cell from nodes
        cell = np.asarray_chkfinite(object.nodes, dtype=np.int32) - 1  # Fortran to C indexing
        if cell.size > 3:
            # trying to get the nodes in winding order by parsing the edges
            edge_dict = {}
            for boundary in object.boundary:
                n1, n2 = (
                    space.objects_per_dimension[EDGE_DIMENSION].object[boundary.index - 1].nodes - 1
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

            cells.append(cell)

    if not winding_ok:
        print("Warning! Unable to verify that the cell nodes are in the winging order.")

    grid = UnstructGrid2D(vertices, cells, name=grid_name)

    if not with_subsets:
        return grid

    # Reading grid subsets (2D only)
    cell_subset_ids = (5, 22, 23, 24, 25, 38, 39, 40)
    subsets = {}
    subset_id = {}
    for subset in grid_ggd.grid_subset:
        dimension_is_2d = subset.dimension == FACE_DIMENSION + 1  # C to Fortran indexing
        known_subset_id = (
            subset.dimension != EMPTY_INT and subset.identifier.index in cell_subset_ids
        )
        if (dimension_is_2d or known_subset_id) and len(subset.element):
            name = str(subset.identifier.name)
            indices = np.empty(len(subset.element), dtype=np.int32)
            for i, element in enumerate(subset.element):
                if len(element.object) > 1:
                    print(
                        f"Warning! Skipping grid subset {name}, "
                        + "because it includes cells not present in the original grid."
                    )
                    break
                indices[i] = element.object[0].index.value
            subsets[name] = indices - 1  # Fortran to C indexing
            subset_id[name] = subset.identifier.index.value

    return grid, subsets, subset_id
