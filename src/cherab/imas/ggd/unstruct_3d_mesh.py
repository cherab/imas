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
"""Module defining unstructured 3D mesh class and related methods."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
from numpy.typing import ArrayLike, NDArray
from raysect.core.math.vector import Vector3D

from ..math import UnstructGridFunction3D, UnstructGridVectorFunction3D
from ..math.tetrahedralize import calculate_tetra_volume, cell_to_5tetra
from .base_mesh import CellSelection, GGDGrid, as_index_array

__all__ = ["UnstructGrid3D"]

ZERO_VECTOR = Vector3D(0, 0, 0)


class UnstructGrid3D(GGDGrid):
    """Unstructured 3D grid object.

    The grid cells are polyhedra. Vertices may be shared with neighbouring cells.

    To use Raysect's KDtree accelerator, each polyhedral cell is tetrahedralized.

    Parameters
    ----------
    vertices
        Array-like of shape ``(N, 3)`` containing coordinates of the polyhedron vertices.
    cells
        List of ``(N,)``-shaped arrays containing the vertex indices in clockwise or
        counterclockwise order for each polyhedral cell in the list (the starting vertex must not be
        included twice).
    name
        Name of the grid, by default ``'Cells'``.
    """

    def __init__(
        self,
        vertices: ArrayLike,
        cells: ArrayLike,
        name: str = "Cells",
    ) -> None:
        vertices = np.array(vertices, dtype=np.float64)
        vertices.setflags(write=False)
        cells = np.array(cells, dtype=np.int32)
        cells.setflags(write=False)

        if vertices.ndim != 2:
            raise ValueError(
                "Attribute 'vertices' must be a 2D array-like. "
                + f"The number of dimensions in 'vertices' is {vertices.ndim}."
            )

        if vertices.shape[1] != 3:
            raise ValueError(
                "Attribute 'vertices' must be a (N, 3) array-like. "
                + f"The shape of 'vertices' is {vertices.shape}."
            )

        if cells.ndim != 2:
            raise ValueError(
                "Attribute 'cells' must be a 2D array-like. "
                + f"The number of dimensions in 'cells' is {cells.ndim}."
            )

        if cells.shape[1] != 8:
            raise ValueError(
                "Attribute 'cells' must be a (M, 8) array-like. "
                + f"The shape of 'cells' is {cells.shape}."
            )

        self._vertices: NDArray[np.float64] = vertices
        self._cells: NDArray[np.int32] = cells

        super().__init__(name=name, dimension=3, coordinate_system="cartesian")

    @override
    def _initial_setup(self) -> None:
        self._interpolator = None

        self._num_cell = len(self._cells)

        self._mesh_extent = {
            "xmin": self._vertices[:, 0].min(),
            "xmax": self._vertices[:, 0].max(),
            "ymin": self._vertices[:, 1].min(),
            "ymax": self._vertices[:, 1].max(),
            "zmin": self._vertices[:, 2].min(),
            "zmax": self._vertices[:, 2].max(),
        }

        # Tetrahedralize cells
        self._tetrahedra = cell_to_5tetra(self._cells)
        self._tetra_to_cell_map = np.repeat(np.arange(len(self._cells), dtype=np.int32), 5)
        self._cell_to_tetra_map = np.column_stack(
            (np.arange(len(self._cells), dtype=np.int32) * 5, np.full(len(self._cells), 5))
        )

        self._tetrahedra.setflags(write=False)
        self._cell_to_tetra_map.setflags(write=False)
        self._tetra_to_cell_map.setflags(write=False)

        # Calculate cell volume and centroid
        self._cell_volume = np.empty(len(self._cells), dtype=np.float64)
        self._cell_centre = np.empty((len(self._cells), 3), dtype=np.float64)

        for i, cell in enumerate(self._cells):
            vertices = self._vertices[cell]
            self._cell_centre[i] = vertices.mean(axis=0)
            itet, ntet = self._cell_to_tetra_map[i]
            self._cell_volume[i] = calculate_tetra_volume(
                self._vertices, self._tetrahedra[itet : itet + ntet]
            )

        self._cell_volume.setflags(write=False)
        self._cell_centre.setflags(write=False)

    @property
    def vertices(self) -> NDArray[np.float64]:
        """Mesh vertex coordinates as ``(N, 3)`` array in the (X, Y, Z) space."""
        return self._vertices

    @property
    def cells(self) -> NDArray[np.int32]:
        """Mesh cells as ``(M, 8)`` array."""
        return self._cells

    @property
    def tetrahedra(self) -> NDArray[np.int32]:
        """Mesh tetrahedra as ``(5M, 4)`` array."""
        return self._tetrahedra

    @property
    def tetra_to_cell_map(self) -> NDArray[np.int32]:
        """Array of shape ``(5M,)`` mapping every tetrahedral index to a grid cell ID."""
        return self._tetra_to_cell_map

    @property
    def cell_to_tetra_map(self) -> NDArray[np.int32]:
        """Array of shape ``(M, 2)`` mapping every grid cell index to tetrahedral IDs.

        The first column is the index of the first tetrahedron forming the cell.
        The second column is the number of tetrahedra forming the cell.

        >>> itet, ntet = mesh.cell_to_tetra_map[icell]
        >>> tet_cell = mesh.tetrahedra[itet : itet + ntet]
        """
        return self._cell_to_tetra_map

    @override
    def subset(self, indices: CellSelection, name: str | None = None) -> UnstructGrid3D:
        """Create a subset UnstructGrid3D from this instance.

        Parameters
        ----------
        indices
            Indices of the cells of the original grid in the subset.
        name
            Name of the grid subset. Default is ``instance.name + " subset"``.

        Returns
        -------
        `.UnstructGrid3D`
            Subset instance.
        """
        index_array = as_index_array(indices)

        grid = UnstructGrid3D.__new__(UnstructGrid3D)

        grid._name = name or self._name + " subset"
        grid._coordinate_system = self._coordinate_system
        grid._dimension = self._dimension
        grid._interpolator = None

        cells_original = self._cells[
            index_array
        ]  # all cells in this subset but with original vertex indices
        vert_index, inv_index = np.unique(
            cells_original, return_inverse=True
        )  # all unique vertex indices in this subset
        grid._vertices = np.array(self._vertices[vert_index])  # vertices in this subset
        grid._vertices.setflags(write=False)

        # renumerating vertex indices
        cells = []  # and split
        ist = 0
        for cell in cells_original:
            cells.append(inv_index[ist : ist + len(cell)])
            ist += len(cell)
        grid._cells = np.vstack(cells, dtype=np.int32)
        grid._cells.setflags(write=False)
        grid._num_cell = len(grid._cells)

        # cell volume and centres of this subset
        grid._cell_volume = np.array(self._cell_volume[index_array])
        grid._cell_volume.setflags(write=False)
        grid._cell_centre = np.array(self._cell_centre[index_array])
        grid._cell_centre.setflags(write=False)

        # mesh extent of this subset
        grid._mesh_extent = {
            "xmin": grid._vertices[:, 0].min(),
            "xmax": grid._vertices[:, 0].max(),
            "ymin": grid._vertices[:, 1].min(),
            "ymax": grid._vertices[:, 1].max(),
            "zmin": grid._vertices[:, 2].min(),
            "zmax": grid._vertices[:, 2].max(),
        }

        # Tetrahedralize cells and maps of this subset
        grid._tetrahedra = cell_to_5tetra(grid._cells)
        grid._tetra_to_cell_map = np.repeat(np.arange(len(grid._cells), dtype=np.int32), 5)
        grid._cell_to_tetra_map = np.column_stack(
            (
                np.arange(len(grid._cells), dtype=np.int32) * 5,
                np.full(len(grid._cells), 5, dtype=np.int32),
            )
        )

        grid._tetrahedra.setflags(write=False)
        grid._cell_to_tetra_map.setflags(write=False)
        grid._tetra_to_cell_map.setflags(write=False)

        return grid

    @override
    def interpolator(
        self, grid_data: NDArray[np.float64], fill_value: float = 0
    ) -> UnstructGridFunction3D:
        """Return an UnstructGridFunction3D interpolator instance for the data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance
        of the previously created interpolator sharing the same KDtree structure.

        Parameters
        ----------
        grid_data
            Array containing data in the grid cells.
        fill_value
            Value returned outside the grid, by default 0.

        Returns
        -------
        `.UnstructGridFunction3D`
            Interpolator instance.

        Raises
        ------
        TypeError
            If the existing interpolator is not compatible with the provided grid data.
        """
        if self._interpolator is None:
            self._interpolator = UnstructGridFunction3D(
                self._vertices, self._tetrahedra, self._tetra_to_cell_map, grid_data, fill_value
            )
            return self._interpolator
        elif not isinstance(self._interpolator, UnstructGridFunction3D):
            raise TypeError(
                "The existing interpolator is not compatible with the provided grid data. "
                + "Please create a new interpolator instance with the appropriate grid data."
            )
        else:
            return UnstructGridFunction3D.instance(self._interpolator, grid_data, fill_value)

    @override
    def vector_interpolator(
        self, grid_vectors: NDArray[np.float64], fill_vector: Vector3D = ZERO_VECTOR
    ) -> UnstructGridVectorFunction3D:
        """Return an `UnstructGridVectorFunction3D` interpolator instance for the vector data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance
        of the previously created interpolator sharing the same KDtree structure.

        Parameters
        ----------
        grid_vectors
            ``(3, L)`` Array containing 3D vectors in the grid cells.
        fill_vector
            3D vector returned outside the grid.

        Returns
        -------
        `.UnstructGridVectorFunction3D`
            Interpolator instance.

        Raises
        ------
        TypeError
            If the existing interpolator is not compatible with the provided grid vectors.
        """
        if self._interpolator is None:
            self._interpolator = UnstructGridVectorFunction3D(
                self._vertices,
                self._tetrahedra,
                self._tetra_to_cell_map,
                grid_vectors,
                fill_vector,
            )
            return self._interpolator

        elif not isinstance(self._interpolator, UnstructGridVectorFunction3D):
            raise TypeError(
                "The existing interpolator is not compatible with the provided grid vectors. "
                + "Please create a new interpolator instance with the appropriate grid vectors."
            )
        else:
            return UnstructGridVectorFunction3D.instance(
                self._interpolator, grid_vectors, fill_vector
            )

    @override
    def __getstate__(self):
        """Get the state of the UnstructGrid3D instance for serialization.

        Returns
        -------
        Dictionary with the instance attributes.
        """
        state = {
            "name": self._name,
            "dimension": self._dimension,
            "coordinate_system": self._coordinate_system,
            "vertices": self._vertices,
            "cells": self._cells,
            "index_flags": self.index_flags if hasattr(self, "index_flags") else None,
        }
        return state

    def __setstate__(self, state):
        """Restore the state of the UnstructGrid3D instance."""
        self._name = state["name"]
        self._dimension = state["dimension"]
        self._coordinate_system = state["coordinate_system"]
        self._vertices = state["vertices"]
        self._vertices.setflags(write=False)
        self._cells = state["cells"]
        self._cells.setflags(write=False)
        if "index_flags" in state:
            # If index_flags exists, it is a subset of the original grid
            self.index_flags = state["index_flags"]

        self._initial_setup()
