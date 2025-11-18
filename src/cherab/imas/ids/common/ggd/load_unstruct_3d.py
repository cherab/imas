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
"""Module for loading unstructured 3D grids from IMAS grid_ggd IDS structure."""

import numpy as np

from imas.ids_struct_array import IDSStructArray
from imas.ids_structure import IDSStructure

from ....ggd import UnstructGrid2DExtended

__all__ = ["load_unstruct_grid_2d_extended", "load_unstruct_grid_3d"]

# Constants for space indices and dimensions
SPACE_RZ = 0
SPACE_FOURIER = 1
VERTEX_DIMENSION = 0
EDGE_DIMENSION = 1
FACE_DIMENSION = 2
TETRA_IN_CELL = 5
NUM_TOROIDAL = 64


def load_unstruct_grid_2d_extended(
    grid_ggd: IDSStructure, with_subsets: bool = False, num_toroidal: int = NUM_TOROIDAL
) -> UnstructGrid2DExtended:
    """Load unstructured 2D grid extended in 3D from the ``grid_ggd`` structure.

    Parameters
    ----------
    grid_ggd
        The ``grid_ggd`` structure.
    with_subsets
        Read grid subset data if True.
    num_toroidal
        Number of toroidal points.
        If specifying more than 1, the grid will be extended in 3D by repeating the 2D
        cross-section around the torus with evenly spaced toroidal angles.

    Returns
    -------
    `.UnstructGrid2DExtended`
        The unstructured 2D grid extended in 3D.

    Raises
    ------
    ValueError
        If the number of toroidal points is less than 1.
        If the grid is not an unstructured extended 2D grid.
    """
    # Validate num_toroidal
    if num_toroidal < 1:
        raise ValueError("The number of toroidal points must be greater than 0.")

    # Get the R-Z space
    space: IDSStructArray = grid_ggd.space[SPACE_RZ]

    # Check if the grid is 2D
    if len(space.objects_per_dimension) != 3:
        raise ValueError(
            "The load_unstruct_grid_2d_extended() supports only unstructured extended 2D grids."
        )

    grid_name = str(grid_ggd.identifier.name)

    # =========================================
    # Reading vertices (poloidal and toroidal)
    # =========================================
    num_poloidal = len(space.objects_per_dimension[VERTEX_DIMENSION].object)
    num_vert = num_poloidal * num_toroidal
    vertices_rpz = np.empty((num_vert, 3), dtype=float)

    for i_phi, i_pol in np.ndindex(num_toroidal, num_poloidal):
        phi = 2.0 * np.pi * i_phi / num_toroidal
        vertices_rpz[i_pol + i_phi * num_poloidal] = (
            space.objects_per_dimension[VERTEX_DIMENSION].object[i_pol].geometry[0],
            phi,
            space.objects_per_dimension[VERTEX_DIMENSION].object[i_pol].geometry[1],
        )
    # Convert to cartesian coordinates
    vertices = np.empty((num_vert, 3), dtype=float)
    p = vertices_rpz[:, 1]
    vertices[:, 0] = vertices_rpz[:, 0] * np.cos(p)
    vertices[:, 1] = vertices_rpz[:, 0] * np.sin(p)
    vertices[:, 2] = vertices_rpz[:, 2]

    # =========================================
    # Reading cells indices
    # =========================================
    faces: IDSStructArray = space.objects_per_dimension[FACE_DIMENSION].object
    num_faces = len(faces)
    cells = np.zeros((num_faces * num_toroidal, 8), dtype=np.int32)
    i_cell = 0
    for i_phi, i_face in np.ndindex(num_toroidal, num_faces):
        i_phi_next = i_phi + 1 if i_phi + 1 < num_toroidal else 0
        cells[i_cell] = np.hstack(
            [
                (faces[i_face].nodes - 1) + i_phi * num_poloidal,
                (faces[i_face].nodes - 1) + i_phi_next * num_poloidal,
            ]
        )
        i_cell += 1

    grid = UnstructGrid2DExtended(
        vertices, cells, num_faces, num_poloidal, num_toroidal, name=grid_name
    )

    if not with_subsets:
        return grid
    else:
        raise NotImplementedError("Reading grid subsets is not implemented yet.")


def load_unstruct_grid_3d(grid_ggd: IDSStructure, space_index: int = 0, with_subsets: bool = False):
    """Load unstructured 3D grid from the ``grid_ggd`` structure.

    .. warning::
        This function is a placeholder for future implementation.

    Parameters
    ----------
    grid_ggd
        The ``grid_ggd`` structure.
    space_index
        The index of the space to read, by default 0.
    with_subsets
        Read grid subset data if True, by default False.

    Returns
    -------
    `.UnstructGrid3D`
        The unstructured 3D grid.
    """
    raise NotImplementedError("Loading unstructured 3D grids will be implemented in the future.")
