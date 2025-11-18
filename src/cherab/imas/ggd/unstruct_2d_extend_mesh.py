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
"""Module defining unstructured 2D-extended mesh class and related methods."""

from __future__ import annotations

import sys
from typing import Literal

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override  # pyright: ignore[reportUnreachable]

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation
from numpy.typing import ArrayLike, NDArray
from raysect.core.math.vector import Vector3D

from ..math import UnstructGridFunction3D, UnstructGridVectorFunction3D
from ..math.tetrahedralize import calculate_tetra_volume, cell_to_5tetra
from .base_mesh import GGDGrid

__all__ = ["UnstructGrid2DExtended"]

ZERO_VECTOR = Vector3D(0, 0, 0)


class UnstructGrid2DExtended(GGDGrid):
    """Unstructured 2D grid with toroidal extension.

    The grid cells are voxels (cubes). Vertices may be shared with neighbouring cells.

    To use Raysect's KDtree accelerator, each cell is tetrahedralized.

    Parameters
    ----------
    vertices
        Array-like of shape ``(N, 3)`` containing coordinates of the cell vertices in the (X, Y, Z)
        space.
    cells
        Array-like of shape ``(M, 8)`` containing the vertex indices in clockwise or
        counterclockwise order for each cubic cell in the list
        (the starting vertex must not be included twice).
    num_faces
        Number of faces at the poloidal plane.
    num_poloidal
        Number of poloidal points.
    num_toroidal
        Number of toroidal points.
    name
        Name of the grid, by default ``"Cells"``.
    coordinate_system
        Coordinate system of the grid, by default ``"cylindrical"``.
    """

    def __init__(
        self,
        vertices: ArrayLike,
        cells: ArrayLike,
        num_faces: int,
        num_poloidal: int,
        num_toroidal: int,
        name: str = "Cells",
        coordinate_system: Literal["cylindrical", "cartesian"] = "cylindrical",
    ):
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
        self._num_faces: int = num_faces
        self._num_poloidal: int = num_poloidal
        self._num_toroidal: int = num_toroidal

        super().__init__(name, 3, coordinate_system)

    @override
    def _initial_setup(self):
        self._interpolator = None

        self._num_cell: int = self._cells.shape[0]

        # Extract grid points at the first poloidal plane
        r = self._vertices[: self._num_poloidal, 0]
        z = self._vertices[: self._num_poloidal, 2]

        # Work out the extent of the mesh (both Cartesian and cylindrical).
        self._mesh_extent = {
            "xmin": self._vertices[:, 0].min(),
            "xmax": self._vertices[:, 0].max(),
            "ymin": self._vertices[:, 1].min(),
            "ymax": self._vertices[:, 1].max(),
            "rmin": r.min(),
            "rmax": r.max(),
            "zmin": z.min(),
            "zmax": z.max(),
        }

        # Tetrahedralize cells
        self._tetrahedra = cell_to_5tetra(self._cells)
        self._tetra_to_cell_map = np.repeat(np.arange(len(self._cells), dtype=np.int32), 5)
        self._cell_to_tetra_map = np.column_stack(
            (np.arange(len(self._cells)) * 5, np.full(len(self._cells), 5))
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
    def num_poloidal(self) -> int:
        """Number of poloidal grid points."""
        return self._num_poloidal

    @property
    def num_toroidal(self) -> int:
        """Number of toroidal grid points."""
        return self._num_toroidal

    @property
    def num_faces(self) -> int:
        """Number of faces at the poloidal plane."""
        return self._num_faces

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

    def subset_faces(self, indices: ArrayLike, name: str | None = None) -> UnstructGrid2DExtended:
        """Create a subset UnstructGrid2DExtended from this instance.

        The subset is defined by the indices of the faces at the poloidal plane.
        Thus, the subset mesh is periodic in the toroidal direction.

        See Also
        --------
        subset : For a method that creates a subset from the original grid cells.

        Parameters
        ----------
        indices
            Indices of the faces of the original grid in the subset.
        name
            Name of the grid subset. Default is ``instance.name + " subset"``.

        Returns
        -------
        `.UnstructGrid2DExtended`
            Subset instance.

        Raises
        ------
        ValueError
            If any of the indices of the faces is out of range.
        """
        grid = UnstructGrid2DExtended.__new__(UnstructGrid2DExtended)

        grid._name = name or self._name + " subset"
        grid._coordinate_system = self._coordinate_system
        grid._dimension = self._dimension
        grid._interpolator = None
        if np.amax(indices) >= self._num_faces:
            raise ValueError(
                "The indices of the faces must be less than the number of faces in the grid."
            )
        indices = np.unique(indices)
        grid._num_faces = len(indices)
        grid._num_toroidal = self._num_toroidal

        index_flags = np.zeros(self._num_faces, dtype=bool)
        index_flags[indices] = True
        index_flags = np.tile(index_flags, self._num_toroidal)
        index_flags.setflags(write=False)

        # TODO: Require to implement index_flags as a property neatly?
        grid.index_flags = index_flags

        cells_original = self.cells[
            index_flags
        ]  # all cells in this subset but with original vertex indices
        vert_index, inv_index = np.unique(
            cells_original, return_inverse=True
        )  # all unique vertex indices in this subset
        grid._vertices = np.array(self.vertices[vert_index])  # vertices in this subset
        grid._vertices.setflags(write=False)

        # renumerating vertex indices
        cells = []  # and split
        ist = 0
        for cell in cells_original:
            cells.append(inv_index[ist : ist + len(cell)])
            ist += len(cell)
        grid._cells = np.array(cells, dtype=np.int32)
        grid._cells.setflags(write=False)
        grid._num_cell = len(grid._cells)
        grid._num_poloidal = grid._vertices.shape[0] // self._num_toroidal

        # cell volume and centres of this subset
        grid._cell_volume = np.array(self._cell_volume[index_flags])
        grid._cell_volume.setflags(write=False)
        grid._cell_centre = np.array(self._cell_centre[index_flags])
        grid._cell_centre.setflags(write=False)

        # Extract grid points at the first poloidal plane
        r = np.hypot(grid._vertices[:, 0], grid._vertices[:, 1])
        z = grid._vertices[:, 2]

        # mesh extent of this subset
        grid._mesh_extent = {
            "xmin": grid._vertices[:, 0].min(),
            "xmax": grid._vertices[:, 0].max(),
            "ymin": grid._vertices[:, 1].min(),
            "ymax": grid._vertices[:, 1].max(),
            "rmin": r.min(),
            "rmax": r.max(),
            "zmin": z.min(),
            "zmax": z.max(),
        }

        # Tetrahedralize cells and maps of this subset
        grid._tetrahedra = cell_to_5tetra(grid._cells)
        grid._tetra_to_cell_map = np.repeat(np.arange(len(grid._cells), dtype=np.int32), 5)
        grid._cell_to_tetra_map = np.column_stack(
            (np.arange(len(grid._cells)) * 5, np.full(len(grid._cells), 5))
        )

        grid._tetrahedra.setflags(write=False)
        grid._cell_to_tetra_map.setflags(write=False)
        grid._tetra_to_cell_map.setflags(write=False)

        return grid

    @override
    def subset(self, indices: ArrayLike, name: str | None = None) -> UnstructGrid2DExtended:
        """Create a subset UnstructGrid2DExtended from this instance.

        .. warning::
            The subset loses the range of cylindrical coordinates
            because the extracted vertex data is not necessarily periodic in the toroidal direction.

        Parameters
        ----------
        indices
            Indices of the cells of the original grid in the subset.
        name
            Name of the grid subset. Default is ``instance.name + " subset"``.

        Returns
        -------
        `.UnstructGrid2DExtended`
            Subset instance.
        """
        grid = UnstructGrid2DExtended.__new__(UnstructGrid2DExtended)

        grid._name = name or self._name + " subset"
        grid._coordinate_system = self._coordinate_system
        grid._dimension = self._dimension
        grid._interpolator = None
        grid._num_faces = None
        grid._num_poloidal = None
        grid._num_toroidal = None

        cells_original = self.cells[
            indices
        ]  # all cells in this subset but with original vertex indices
        vert_index, inv_index = np.unique(
            cells_original, return_inverse=True
        )  # all unique vertex indices in this subset
        grid._vertices = np.array(self.vertices[vert_index])  # vertices in this subset
        grid._vertices.setflags(write=False)

        # renumerating vertex indices
        cells = []  # and split
        ist = 0
        for cell in cells_original:
            cells.append(inv_index[ist : ist + len(cell)])
            ist += len(cell)
        grid._cells = np.array(cells, dtype=np.int32)
        grid._cells.setflags(write=False)
        grid._num_cell = len(grid._cells)

        # cell volume and centres of this subset
        grid._cell_volume = np.array(self._cell_volume[indices])
        grid._cell_volume.setflags(write=False)
        grid._cell_centre = np.array(self._cell_centre[indices])
        grid._cell_centre.setflags(write=False)

        # mesh extent of this subset
        grid._mesh_extent = {
            "xmin": grid._vertices[:, 0].min(),
            "xmax": grid._vertices[:, 0].max(),
            "ymin": grid._vertices[:, 1].min(),
            "ymax": grid._vertices[:, 1].max(),
        }

        # Tetrahedralize cells and maps of this subset
        grid._tetrahedra = cell_to_5tetra(grid._cells)
        grid._tetra_to_cell_map = np.repeat(np.arange(len(grid._cells), dtype=np.int32), 5)
        grid._cell_to_tetra_map = np.column_stack(
            (np.arange(len(grid._cells)) * 5, np.full(len(grid._cells), 5))
        )

        grid._tetrahedra.setflags(write=False)
        grid._cell_to_tetra_map.setflags(write=False)
        grid._tetra_to_cell_map.setflags(write=False)

        return grid

    @override
    def interpolator(self, grid_data: ArrayLike, fill_value: float = 0) -> UnstructGridFunction3D:
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
        """
        if self._interpolator is None:
            self._interpolator = UnstructGridFunction3D(
                self._vertices, self._tetrahedra, self._tetra_to_cell_map, grid_data, fill_value
            )
            return self._interpolator

        return UnstructGridFunction3D.instance(self._interpolator, grid_data, fill_value)

    @override
    def vector_interpolator(
        self, grid_vectors: ArrayLike, fill_vector: Vector3D = ZERO_VECTOR
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

        return UnstructGridVectorFunction3D.instance(self._interpolator, grid_vectors, fill_vector)

    @override
    def __getstate__(self):
        """Get the state of the UnstructGrid2DExtended instance for serialization.

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
            "num_poloidal": self._num_poloidal,
            "num_toroidal": self._num_toroidal,
            "num_faces": self._num_faces,
            "index_flags": self.index_flags if hasattr(self, "index_flags") else None,
        }
        return state

    def __setstate__(self, state):
        """Restore the state of the UnstructGrid2DExtended instance."""
        self._name = state["name"]
        self._dimension = state["dimension"]
        self._coordinate_system = state["coordinate_system"]
        self._vertices = state["vertices"]
        self._vertices.setflags(write=False)
        self._cells = state["cells"]
        self._cells.setflags(write=False)
        self._num_poloidal = state["num_poloidal"]
        self._num_toroidal = state["num_toroidal"]
        self._num_faces = state["num_faces"]
        if "index_flags" in state:
            # If index_flags exists, it is a subset of the original grid
            self.index_flags = state["index_flags"]

        self._initial_setup()

    def plot_tetra_mesh(
        self, data: ArrayLike | None = None, ax: matplotlib.axes.Axes | None = None
    ) -> None:
        """Plot the tetrahedral mesh grid geometry.

        .. warning::
            Plotting of tetrahedral mesh is not implemented yet.

        Parameters
        ----------
        data
            Data array defined on the tetrahedral mesh.
        ax
            Matplotlib axes to plot the mesh. If None, a new figure is created.

        Raises
        ------
        NotImplementedError
            Plotting of tetrahedral mesh is not implemented yet.
        """
        raise NotImplementedError("Plotting of tetrahedral mesh is not implemented yet.")

    @override
    def plot_mesh(
        self,
        data: ArrayLike | None = None,
        ax: matplotlib.axes.Axes | None = None,
        **grid_styles: str | float,
    ) -> matplotlib.axes.Axes:
        """Plot the polygonal mesh grid geometry at the first poloidal plane to a matplotlib figure.

        Parameters
        ----------
        data
            Data array defined on the polygonal mesh at the poloidal plane.
        ax
            Matplotlib axes to plot the mesh. If None, a new figure is created.
        **grid_styles
            Styles for the grid lines and faces,
            by default ``{"facecolor": "none", "edgecolor": "b", "linewidth": 0.25}``.

        Returns
        -------
        `~matplotlib.axes.Axes`
            Matplotlib axes with the plotted mesh.
        """
        if ax is None:
            _, ax = plt.subplots(layout="constrained")

        # Set default grid line styles if not provided
        grid_styles.setdefault("facecolor", "none")
        grid_styles.setdefault("edgecolor", "b")
        grid_styles.setdefault("linewidth", 0.25)

        cells = self._cells[: self._num_faces, :4]
        verts = [self._vertices[cell][:, ::2] for cell in cells]

        if data is None:
            collection_mesh = PolyCollection(verts, **grid_styles)
        else:
            collection_mesh = PolyCollection(verts)
            collection_mesh.set_array(data)
        ax.add_collection(collection_mesh)
        ax.set_aspect(1)
        ax.set_xlim(self._mesh_extent["rmin"], self._mesh_extent["rmax"])
        ax.set_ylim(self._mesh_extent["zmin"], self._mesh_extent["zmax"])

        if self._coordinate_system == "cartesian":
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
        elif self._coordinate_system == "cylindrical":
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")

        return ax

    def plot_tri_mesh(
        self, data: ArrayLike, ax: matplotlib.axes.Axes | None = None, cmap: str = "viridis"
    ) -> matplotlib.axes.Axes:
        """Plot the data defined on the triangular mesh at the poloidal plane to a matplotlib figure.

        Parameters
        ----------
        data
            Data array defined on the triangular mesh at the poloidal plane.
        ax
            Matplotlib axes to plot the mesh. If None, a new figure is created.
        cmap
            Colormap to use for the data, by default 'viridis'.

        Returns
        -------
        `~matplotlib.axes.Axes`
            Matplotlib axes with the plotted mesh.

        Raises
        ------
        ValueError
            If the data array does not have the same number of faces as the grid.
        """
        data = np.asarray_chkfinite(data)
        if data.shape[0] != self._num_faces:
            raise ValueError("The data array must have the same number of faces as the grid.")
        data = np.repeat(data, 2)

        cells = self._cells[: self._num_faces]
        tri = np.empty((self._num_faces * 2, 3), dtype=np.int32)
        for i, cell in enumerate(cells):
            tri[2 * i] = [cell[0], cell[1], cell[2]]
            tri[2 * i + 1] = [cell[0], cell[2], cell[3]]
        verts = self._vertices[: self.num_poloidal, ::2]
        triangles = Triangulation(verts[:, 0], verts[:, 1], tri)

        if ax is None:
            _, ax = plt.subplots(layout="constrained")

        ax.set_aspect(1)
        ax.tripcolor(triangles, data, cmap=cmap)
        ax.set_xlim(self._mesh_extent["rmin"], self._mesh_extent["rmax"])
        ax.set_ylim(self._mesh_extent["zmin"], self._mesh_extent["zmax"])

        if self._coordinate_system == "cartesian":
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
        elif self._coordinate_system == "cylindrical":
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")

        return ax
