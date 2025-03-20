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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation
from raysect.core.math import Vector3D

from ..math import UnstructGridFunction3D, UnstructGridVectorFunction3D
from ..math.tetrahedralize import cell_to_5tetra, calculate_tetra_volume
from .base_mesh import GGDGrid


class UnstructGrid2DExtended(GGDGrid):
    """
    Unstructured 2D grid with toroidal extension.

    The grid cells are voxels (cubes). Vertices may be shared with neighbouring cells.

    To use Raysect's KDtree accelerator, each cell is tetrahedralized.

    :param object vertices: Array-like of shape (N, 3) containing coordinates of the cell vertices
        in the (X, Y, Z) space.
    :param object cells: Array-like of shape (M, 8) containing the vertex indices
        in clockwise or counterclockwise order for each cubic cell in the list
        (the starting vertex must not be included twice).
    :param int num_faces: Number of faces at the poloidal plane.
    :param int num_poloidal: Number of poloidal points.
    :param int num_toroidal: Number of toroidal points.
    :param str name: A name of the grid. Default is 'Cells'.
    """
    def __init__(self, vertices, cells, num_faces, num_poloidal, num_toroidal, name="Cells", coordinate_system="cylindrical"):
        vertices = np.array(vertices, dtype=np.float64)
        vertices.setflags(write=False)
        cells = np.array(cells, dtype=np.int32)
        cells.setflags(write=False)

        if vertices.ndim != 2:
            raise ValueError("Attribute 'vertices' must be a 2D array-like. The number of dimensions in 'vertices' is {}.".format(vertices.ndim))

        if vertices.shape[1] != 3:
            raise ValueError("Attribute 'vertices' must be a (N, 3) array-like. The shape of 'vertices' is {}.".format(vertices.shape))

        if cells.ndim != 2:
            raise ValueError("Attribute 'cells' must be a 2D array-like. The number of dimensions in 'cells' is {}.".format(cells.ndim))

        if cells.shape[1] != 8:
            raise ValueError("Attribute 'cells' must be a (M, 8) array-like. The shape of 'cells' is {}.".format(cells.shape))

        self._vertices = vertices
        self._cells = cells
        self._num_faces = num_faces
        self._num_poloidal = num_poloidal
        self._num_toroidal = num_toroidal

        super().__init__(name, 3, coordinate_system)

    def _initial_setup(self):
        self._interpolator = None

        self._num_cell = self._cells.shape[0]

        # Extract grid points at the first poloidal plane
        r = self._vertices[:self._num_poloidal, 0]
        z = self._vertices[:self._num_poloidal, 2]

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
        self._cell_to_tetra_map = np.column_stack((np.arange(len(self._cells)) * 5, np.full(len(self._cells), 5)))

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
            self._cell_volume[i] = calculate_tetra_volume(self._vertices, self._tetrahedra[itet : itet + ntet])

        self._cell_volume.setflags(write=False)
        self._cell_centre.setflags(write=False)

    @property
    def num_poloidal(self):
        """Number of poloidal grid points."""
        return self._num_poloidal

    @property
    def num_toroidal(self):
        """Number of toroidal grid points."""
        return self._num_toroidal

    @property
    def num_faces(self):
        """Number of faces at the poloidal plane."""
        return self._num_faces

    @property
    def vertices(self):
        """Mesh vertex coordinates as (N, 3) array in the (X, Y, Z) space."""
        return self._vertices

    @property
    def cells(self):
        """Mesh cells as (M, 8) array."""
        return self._cells

    @property
    def tetrahedra(self):
        """Mesh tetrahedra as (5M, 4) array."""
        return self._tetrahedra

    @property
    def tetra_to_cell_map(self):
        """
        Array of shape (5M,) mapping every tetrahedral index to a grid cell ID.
        """
        return self._tetra_to_cell_map

    @property
    def cell_to_tetra_map(self):
        """
        Array of shape (M, 2) mapping every grid cell index to tetrahedral IDs.
        The first column is the index of the first tetrahedron forming the cell.
        The second column is the number of tetrahedra forming the cell.

        .. code-block:: pycon

            >>> itet, ntet = mesh.cell_to_tetra_map[icell]
            >>> tet_cell = mesh.tetrahedra[itet:itet + ntet]
        """
        return self._cell_to_tetra_map

    def subset(self, indices, name=None):
        """
        Creates a subset UnstructGrid2DExtended from this instance.

        .. warning::
            The subset loses the range of cylindrical coordinates
            because the extracted vertex data is not necessarily periodic in the toroidal direction.

        :param indices: Indices of the cells of the original grid in the subset.
        :param name: Name of the grid subset. Default is instance.name + ' subset'.
        """

        grid = UnstructGrid2DExtended.__new__(UnstructGrid2DExtended)

        grid._name = name or self._name + " subset"
        grid._coordinate_system = self._coordinate_system
        grid._dimension = self._dimension
        grid._interpolator = None
        grid._num_faces = None
        grid._num_poloidal = None
        grid._num_toroidal = None

        cells_original = self.cells[indices]  # all cells in this subset but with original vertex indices
        vert_indx, inv_indx = np.unique(cells_original, return_inverse=True)  # all unique vertex indices in this subset
        grid._vertices = np.array(self.vertices[vert_indx])  # vertices in this subset
        grid._vertices.setflags(write=False)

        # renumerating vertex indices
        cells = []  # and split
        ist = 0
        for cell in cells_original:
            cells.append(inv_indx[ist : ist + len(cell)])
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
        grid._cell_to_tetra_map = np.column_stack((np.arange(len(grid._cells)) * 5, np.full(len(grid._cells), 5)))

        grid._tetrahedra.setflags(write=False)
        grid._cell_to_tetra_map.setflags(write=False)
        grid._tetra_to_cell_map.setflags(write=False)

        return grid

    def interpolator(self, grid_data, fill_value=0):
        """
        Returns an UnstructGridFunction3D interpolator instance for the data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance
        of the previously created interpolator sharing the same KDtree structure.

        :param grid_data: An array containing data in the grid cells.
        :param fill_value: A value returned outside the gird. Default is 0.

        :returns: UnstructGridFunction3D interpolator
        """
        if self._interpolator is None:
            self._interpolator = UnstructGridFunction3D(
                self._vertices, self._tetrahedra, self._tetra_to_cell_map, grid_data, fill_value
            )
            return self._interpolator

        return UnstructGridFunction3D.instance(self._interpolator, grid_data, fill_value)

    def vector_interpolator(self, grid_vectors, fill_vector=Vector3D(0, 0, 0)):
        """
        Returns an UnstructGridVectorFunction3D interpolator instance for the vector data
        defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance
        of the previously created interpolator sharing the same KDtree structure.

        :param grid_vectors: A (3,K) array containing 3D vectors in the grid cells.
        :param fill_vector: A 3D vector returned outside the grid. Default is (0, 0, 0).

        :returns: UnstructGridVectorFunction3D interpolator
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

    def __getstate__(self):
        state = {
            "name": self._name,
            "dimension": self._dimension,
            "coordinate_system": self._coordinate_system,
            "vertices": self._vertices,
            "cells": self._cells,
            "num_poloidal": self._num_poloidal,
            "num_toroidal": self._num_toroidal,
            "num_faces": self._num_faces,
        }
        return state

    def __setstate__(self, state):
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

        self._initial_setup()

    def plot_tetra_mesh(self, data=None, ax=None):
        """
        Plot the tetrahedral mesh grid geometry.

        :param data: Data array defined on the tetrahedral mesh
        """
        raise NotImplementedError("Plotting of tetrahedral mesh is not implemented yet.")

    def plot_mesh_poloidal(self, data=None, ax=None):
        """
        Plot the polygonal mesh grid geometry at the first poloidal plane to a matplotlib figure.

        :param data: Data array defined on the polygonal mesh at the poloidal plane
        """
        if ax is None:
            _, ax = plt.subplots(layout="constrained")

        cells = self._cells[: self._num_faces, :4]
        verts = [self._vertices[cell][:, ::2] for cell in cells]

        if data is None:
            collection_mesh = PolyCollection(verts, facecolor="none", edgecolor="b", linewidth=0.25)
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

    def plot_tri_mesh(self, data, ax=None, cmap="viridis"):
        """
        Plot the data defined on the triangular mesh at the poloidal plane to a matplotlib figure.

        :param data: Data array defined on the cells at the poloidal plane
        """
        data = np.asarray(data)
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