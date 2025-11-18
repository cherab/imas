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
"""Module defining unstructured 2D mesh class and related methods."""

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
from numpy.typing import ArrayLike, NDArray
from raysect.core.math.polygon import triangulate2d
from raysect.core.math.vector import Vector3D

from ..math import UnstructGridFunction2D, UnstructGridVectorFunction2D
from .base_mesh import GGDGrid

__all__ = ["UnstructGrid2D"]

ZERO_VECTOR = Vector3D(0, 0, 0)


class UnstructGrid2D(GGDGrid):
    """Unstructured 2D grid object.

    The grid cells are polygons. Vertices may be shared with neighbouring cells.

    To use Raysect's KDtree accelerator, each polygonal cell is triangulated.

    Parameters
    ----------
    vertices
        Array-like of shape ``(N, 2)`` containing coordinates of the polygon vertices.
    cells
        List of ``(N,)``-shaped arrays containing the vertex indices in clockwise or
        counterclockwise order for each polygonal cell in the list (the starting vertex must not be
        included twice).
    name
        Name of the grid, by default ``'Cells'``.
    coordinate_system
        Coordinate system of the grid, by default ``'cylindrical'``.
    """

    def __init__(
        self,
        vertices: ArrayLike,
        cells: list[ArrayLike],
        name: str = "Cells",
        coordinate_system: Literal["cylindrical", "cartesian"] = "cylindrical",
    ):
        vertices = np.asarray_chkfinite(vertices, dtype=np.float64)
        vertices.setflags(write=False)

        if vertices.ndim != 2:
            raise ValueError(
                "Attribute 'vertices' must be a 2D array-like. "
                + f"The number of dimensions in 'vertices' is {vertices.ndim}."
            )

        if vertices.shape[1] != 2:
            raise ValueError(
                "Attribute 'vertices' must be a (N, 2) array-like. "
                + f"The shape of 'vertices' is {vertices.shape}."
            )

        if not len(cells):
            raise ValueError("The list of cells must contain at least one element.")

        for cell in cells:
            if len(cell) < 3:
                raise ValueError(f"Cell {np.array2string(cell)} is not a polygon.")

        self._vertices: NDArray[np.float64] = vertices
        self._cells: tuple[ArrayLike, ...] = tuple(cells)

        super().__init__(name, 2, coordinate_system)

    @override
    def _initial_setup(self) -> None:
        self._interpolator = None

        self._num_cell: int = len(self._cells)

        x = self._vertices[:, 0]
        y = self._vertices[:, 1]

        ntri_total = 0
        for cell in self._cells:
            ntri_total += len(cell) - 2

        # Work out the extent of the mesh (both Cartesian and cylindrical).
        self._mesh_extent = {
            "xmin": x.min(),
            "xmax": x.max(),
            "ymin": y.min(),
            "ymax": y.max(),
            "rmin": x.min(),
            "rmax": x.max(),
            "zmin": y.min(),
            "zmax": y.max(),
        }

        # Triangulate cells
        self._triangles = np.empty((ntri_total, 3), dtype=np.int32)
        self._cell_to_triangle_map = np.empty((len(self._cells), 2), dtype=np.int32)
        self._triangle_to_cell_map = np.empty(ntri_total, dtype=np.int32)

        itri = 0
        for i, cell in enumerate(self._cells):
            ntri = len(cell) - 2
            if ntri == 1:
                self._triangles[i] = cell
            else:
                vert = self._vertices[cell]
                tri = triangulate2d(vert)
                self._triangles[itri : itri + ntri] = cell[tri]
            self._cell_to_triangle_map[i] = [itri, ntri]
            self._triangle_to_cell_map[itri : itri + ntri] = i
            itri += ntri

        self._triangles.setflags(write=False)
        self._cell_to_triangle_map.setflags(write=False)
        self._triangle_to_cell_map.setflags(write=False)

        # Calculate cell area and centroid
        self._cell_centre = np.empty((len(self._cells), 2), dtype=np.float64)
        self._cell_area = np.empty(len(self._cells), dtype=np.float64)

        vx = x[self._triangles]
        vy = y[self._triangles]
        area = 0.5 * np.abs(
            (vx[:, 0] - vx[:, 2]) * (vy[:, 1] - vy[:, 2])
            - (vx[:, 1] - vx[:, 2]) * (vy[:, 0] - vy[:, 2])
        )

        for i, cell in enumerate(self._cells):
            self._cell_centre[i] = self._vertices[cell].mean(0)
            i_start, ntri = self._cell_to_triangle_map[i]
            self._cell_area[i] = area[i_start : i_start + ntri].sum()

        self._cell_centre.setflags(write=False)
        self._cell_area.setflags(write=False)

        if self._coordinate_system == "cylindrical":
            self._cell_volume = 0.5 * np.pi * self._cell_centre[:, 0] * self._cell_area
            self._cell_volume.setflags(write=False)

    @property
    def vertices(self) -> NDArray[np.float64]:
        """Mesh vertex coordinates as ``(N, 2)`` array."""
        return self._vertices

    @property
    def cells(self) -> tuple[ArrayLike, ...]:
        """List of ``K`` polygonal cells."""
        return self._cells

    @property
    def triangles(self) -> NDArray[np.int32]:
        """Mesh triangles as ``(M, 3)`` array."""
        return self._triangles

    @property
    def triangle_to_cell_map(self) -> NDArray[np.int32]:
        """Array of shape ``(M,)`` mapping every triangle index to a grid cell ID."""
        return self._triangle_to_cell_map

    @property
    def cell_to_triangle_map(self) -> NDArray[np.int32]:
        """Array of shape ``(K, 2)`` mapping every grid cell index to triangle IDs.

        The first column is the index of the first triangle forming the cell.
        The second column is the number of triangles forming the cell.

        >>> itri, ntri = mesh.cell_to_triangle_map[icell]
        >>> tri_cell = mesh.triangles[itri : itri + ntri]
        """
        return self._cell_to_triangle_map

    @override
    def subset(self, indices: ArrayLike, name: str | None = None) -> UnstructGrid2D:
        """Create a subset UnstructGrid2D from this instance.

        Parameters
        ----------
        indices
            Indices of the cells of the original grid in the subset.
        name
            Name of the grid subset. Default is ``instance.name + " subset"``.

        Returns
        -------
        `.UnstructGrid2D`
            Subset instance.
        """
        grid = UnstructGrid2D.__new__(UnstructGrid2D)

        grid._name = name or self.name + " subset"
        grid._coordinate_system = self._coordinate_system
        grid._dimension = self._dimension
        grid._interpolator = None

        cells_original = tuple(
            self.cells[i] for i in indices
        )  # all cells in this subset but with original vertex indices
        cells_all = np.concatenate(
            cells_original
        )  # all vertex indices in this subset with repetitions
        vert_index, inv_index = np.unique(
            cells_all, return_inverse=True
        )  # all unique vertex indices in this subset
        grid._vertices = np.array(self.vertices[vert_index])  # vertices in this subset
        grid._vertices.setflags(write=False)

        # renumerating vertex indices
        cells = []  # and split
        i_start = 0
        for cell in cells_original:
            cells.append(inv_index[i_start : i_start + len(cell)])
            i_start += len(cell)
        grid._cells = tuple(cells)
        grid._num_cell = len(grid._cells)
        ntri_total = i_start - 2 * len(cells_original)

        # cell area and centres of this subset
        grid._cell_area = np.array(self.cell_area[indices])
        grid._cell_area.setflags(write=False)
        grid._cell_centre = np.array(self.cell_centre[indices])
        grid._cell_centre.setflags(write=False)

        # mesh extent of this subset
        xmin, ymin = grid._vertices.min(0)
        xmax, ymax = grid._vertices.max(0)
        grid._mesh_extent = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "rmin": xmin,
            "rmax": xmax,
            "zmin": ymin,
            "zmax": ymax,
        }

        # triangles and maps of this subset
        grid._triangles = np.empty((ntri_total, 3), dtype=np.int32)
        grid._cell_to_triangle_map = np.empty((len(cells), 2), dtype=np.int32)
        grid._triangle_to_cell_map = np.empty(ntri_total, dtype=np.int32)

        c2t_map = self.cell_to_triangle_map[indices]  # map with original triangle indices
        # maps original vertices to the subset, -1 if not in the subset
        subset_vertex_map = -1 * np.ones(self.vertices.shape[0], dtype=np.int32)
        subset_vertex_map[vert_index] = np.arange(vert_index.size, dtype=np.int32)

        itri = 0
        for i, cell in enumerate(cells):
            ntri = len(cell) - 2
            if ntri == 1:
                grid._triangles[i] = cell
            else:
                c2t = c2t_map[i]
                tri = self.triangles[c2t[0] : c2t[0] + c2t[1]]
                grid._triangles[itri : itri + ntri] = subset_vertex_map[tri]
            grid._cell_to_triangle_map[i] = [itri, ntri]
            grid._triangle_to_cell_map[itri : itri + ntri] = i
            itri += ntri

        grid._triangles.setflags(write=False)
        grid._cell_to_triangle_map.setflags(write=False)
        grid._triangle_to_cell_map.setflags(write=False)

        return grid

    @override
    def interpolator(self, grid_data: ArrayLike, fill_value: float = 0) -> UnstructGridFunction2D:
        """Return an `UnstructGridFunction2D` interpolator instance for the data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance of the
        previously created interpolator sharing the same KDtree structure.

        Parameters
        ----------
        grid_data
            Array containing data in the grid cells.
        fill_value
            Value returned outside the grid, by default is 0.

        Returns
        -------
        `.UnstructGridFunction2D`
            Interpolator instance.
        """
        if self._interpolator is None:
            self._interpolator = UnstructGridFunction2D(
                self._vertices, self._triangles, self._triangle_to_cell_map, grid_data, fill_value
            )
            return self._interpolator

        return UnstructGridFunction2D.instance(self._interpolator, grid_data, fill_value)

    @override
    def vector_interpolator(
        self, grid_vectors: ArrayLike, fill_vector: Vector3D = ZERO_VECTOR
    ) -> UnstructGridVectorFunction2D:
        """Return an `UnstructGridVectorFunction2D` interpolator instance for the vector data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance of the
        previously created interpolator sharing the same KDtree structure.

        Parameters
        ----------
        grid_vectors
            ``(3, K)`` Array containing 3D vectors in the grid cells.
        fill_vector
            3D vector returned outside the grid.

        Returns
        -------
        `.UnstructGridVectorFunction2D`
            Interpolator instance.
        """
        if self._interpolator is None:
            self._interpolator = UnstructGridVectorFunction2D(
                self._vertices,
                self._triangles,
                self._triangle_to_cell_map,
                grid_vectors,
                fill_vector,
            )
            return self._interpolator

        return UnstructGridVectorFunction2D.instance(self._interpolator, grid_vectors, fill_vector)

    @override
    def __getstate__(self):
        """Serialize the state of the UnstructGrid2D instance for pickling.

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
        }
        return state

    def __setstate__(self, state):
        """Restore the state of the UnstructGrid2D instance from the serialized state."""
        self._name = state["name"]
        self._dimension = state["dimension"]
        self._coordinate_system = state["coordinate_system"]
        self._vertices = state["vertices"]
        self._vertices.setflags(write=False)
        self._cells = state["cells"]

        self._initial_setup()

    def plot_triangle_mesh(
        self, data: ArrayLike | None = None, ax: matplotlib.axes.Axes | None = None
    ) -> matplotlib.axes.Axes:
        """Plot the triangle mesh grid geometry to a matplotlib figure.

        Parameters
        ----------
        data
            Data array defined on the polygonal mesh.
        ax
            Matplotlib axes to plot on. If None, a new figure and axes are created.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes with the plotted mesh.
        """
        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        verts = self._vertices[self._triangles]
        if data is None:
            collection_mesh = PolyCollection(
                [verts], facecolor="none", edgecolor="b", linewidth=0.25
            )
        else:
            collection_mesh = PolyCollection([verts])
            collection_mesh.set_array(data[self._triangle_to_cell_map])
        ax.add_collection(collection_mesh)
        ax.set_aspect(1)
        ax.set_xlim(self._mesh_extent["xmin"], self._mesh_extent["xmax"])
        ax.set_ylim(self._mesh_extent["ymin"], self._mesh_extent["ymax"])

        if self._coordinate_system == "cartesian":
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
        elif self._coordinate_system == "cylindrical":
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")

        return ax

    @override
    def plot_mesh(
        self,
        data: ArrayLike | None = None,
        ax: matplotlib.axes.Axes | None = None,
        **grid_styles: str | float,
    ) -> matplotlib.axes.Axes:
        """Plot the polygonal mesh grid geometry to a matplotlib figure.

        Parameters
        ----------
        data
            Data array defined on the polygonal mesh.
        ax
            Matplotlib axes to plot on. If None, a new figure and axes are created.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes with the plotted mesh.
        """
        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        # Set default grid line styles if not provided
        grid_styles.setdefault("facecolor", "none")
        grid_styles.setdefault("edgecolor", "b")
        grid_styles.setdefault("linewidth", 0.25)

        verts = [self._vertices[cell] for cell in self._cells]
        if data is None:
            collection_mesh = PolyCollection(verts, **grid_styles)
        else:
            collection_mesh = PolyCollection(verts)
            collection_mesh.set_array(data)
        ax.add_collection(collection_mesh)
        ax.set_aspect(1)
        ax.set_xlim(self._mesh_extent["xmin"], self._mesh_extent["xmax"])
        ax.set_ylim(self._mesh_extent["ymin"], self._mesh_extent["ymax"])

        if self._coordinate_system == "cartesian":
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
        elif self._coordinate_system == "cylindrical":
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")

        return ax
