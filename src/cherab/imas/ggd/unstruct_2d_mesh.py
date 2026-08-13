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
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation
from numpy.typing import ArrayLike, NDArray
from raysect.core.math.polygon import triangulate2d
from raysect.core.math.vector import Vector3D

from ..math import UnstructGridFunction2D, UnstructGridVectorFunction2D
from ..math.polygon import calculate_2d_cell_geometry
from .base_mesh import (
    CellConnectivity,
    CellData,
    CellSelection,
    GGDGrid,
    InterpolatorCacheMode,
    as_index_array,
)

__all__ = ["UnstructGrid2D"]

ZERO_VECTOR = Vector3D(0, 0, 0)


def _as_cell_data(data: CellData, valid_data_mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Return validated scalar cell data as a one-dimensional float array.

    Parameters
    ----------
    data
        Scalar values defined on grid cells.
    valid_data_mask
        Boolean array indicating valid cell data.

    Returns
    -------
    NDArray[numpy.float64]
        Validated one-dimensional cell data.

    Raises
    ------
    ValueError
        If the data is not one-dimensional or does not match the valid data mask.
    """
    data_array = np.asarray_chkfinite(data, dtype=np.float64)
    if data_array.ndim != 1:
        raise ValueError("Cell data must be one-dimensional.")
    num_valid = int(np.count_nonzero(valid_data_mask))
    if data_array.size == num_valid:
        return data_array
    if data_array.size == valid_data_mask.size:
        return data_array[valid_data_mask]
    raise ValueError(
        "Cell data must contain either the number of valid cells or the number of "
        f"source entries. Data size: {data_array.size}, valid cells: {num_valid}, "
        f"source entries: {valid_data_mask.size}."
    )


class UnstructGrid2D(GGDGrid):
    """Unstructured 2D grid object.

    The grid cells are polygons. Vertices may be shared with neighbouring cells.

    To use Raysect's KDtree accelerator, each polygonal cell is triangulated.

    Parameters
    ----------
    vertices
        Array-like of shape ``(N, 2)`` containing coordinates of the polygon vertices.
    cells
        An ``(N, 4)`` integer array or a list/tuple of 1-D integer index arrays containing
        the vertex indices in clockwise or counterclockwise order for each polygonal cell
        (the starting vertex must not be included twice).
    valid_data_mask
        Boolean mask over the source face/data entries. Its number of ``True``
        values must equal the number of cells retained by this grid. Data passed
        to plotting/interpolation may therefore be either source-sized or already
        compacted to the valid cells.
    name
        Name of the grid, by default ``'Cells'``.
    coordinate_system
        Coordinate system of the grid, by default ``'cylindrical'``.
    """

    def __init__(
        self,
        vertices: ArrayLike,
        cells: CellConnectivity,
        valid_data_mask: NDArray[np.bool_] | Sequence[bool] | None = None,
        name: str = "Cells",
        coordinate_system: Literal["cylindrical", "cartesian"] = "cylindrical",
    ) -> None:
        vertices = np.ascontiguousarray(np.asarray_chkfinite(vertices, dtype=np.float64))
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

        normalized_cells: list[NDArray[np.intp]] = []
        for cell in cells:
            cell_array = np.asarray(cell, dtype=np.intp)
            if len(cell_array) < 3:
                raise ValueError(f"Cell {np.array2string(cell_array)} is not a polygon.")
            normalized_cells.append(cell_array)

        self._vertices: NDArray[np.float64] = vertices
        self._cells: tuple[NDArray[np.intp], ...] = tuple(normalized_cells)

        if valid_data_mask is None:
            valid_data_mask = np.ones(len(self._cells), dtype=np.bool_)
        else:
            valid_data_mask = np.asarray(valid_data_mask, dtype=np.bool_)
        if valid_data_mask.ndim != 1:
            raise ValueError("valid_data_mask must be one-dimensional.")
        if (n := np.count_nonzero(valid_data_mask)) != len(self._cells):
            raise ValueError(
                f"The number of valid data mask entries ({n})"
                f" must match the number of cells ({len(self._cells)})."
            )
        self._valid_data_mask = np.array(valid_data_mask, dtype=np.bool_, copy=True)
        self._valid_data_mask.setflags(write=False)

        super().__init__(name, 2, coordinate_system)

    @override
    def _initial_setup(self) -> None:
        self._scalar_interpolator = None
        self._vector_interpolator = None
        self._triangulation: Triangulation | None = None

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
                self._triangles[itri] = cell
            else:
                vert = self._vertices[cell]
                tri = triangulate2d(cast(Any, vert))
                self._triangles[itri : itri + ntri] = cell[tri]
            self._cell_to_triangle_map[i] = [itri, ntri]
            self._triangle_to_cell_map[itri : itri + ntri] = i
            itri += ntri

        self._triangles.setflags(write=False)
        self._cell_to_triangle_map.setflags(write=False)
        self._triangle_to_cell_map.setflags(write=False)

        # Calculate cell areas and area centroids in Cython.
        self._cell_centre, self._cell_area = calculate_2d_cell_geometry(
            self._vertices, self._triangles, self._cell_to_triangle_map
        )
        self._cell_centre.setflags(write=False)
        self._cell_area.setflags(write=False)

        if self._coordinate_system == "cylindrical":
            self._cell_volume = np.multiply(self._cell_centre[:, 0], self._cell_area)
            np.multiply(self._cell_volume, 2.0 * np.pi, out=self._cell_volume)
            self._cell_volume.setflags(write=False)

    @property
    def vertices(self) -> NDArray[np.float64]:
        """Mesh vertex coordinates as ``(N, 2)`` array."""
        return self._vertices

    @property
    def cells(self) -> tuple[NDArray[np.intp], ...]:
        """List of ``K`` polygonal cells as 1-D integer index arrays."""
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

    @property
    def valid_data_mask(self) -> NDArray[np.bool_]:
        """Boolean mask over source data entries retained by this grid."""
        return self._valid_data_mask

    @override
    def subset(
        self,
        indices: CellSelection,
        name: str | None = None,
        *,
        valid_data_mask: NDArray[np.bool_] | Sequence[bool] | None = None,
    ) -> UnstructGrid2D:
        """Create a subset UnstructGrid2D from this instance.

        Parameters
        ----------
        indices
            Indices of the cells of the original grid in the subset.
        valid_data_mask
            Boolean array indicating which cells in the subset have valid data.
        name
            Name of the grid subset. Default is ``instance.name + " subset"``.

        Returns
        -------
        `.UnstructGrid2D`
            Subset instance.

        Raises
        ------
        ValueError
            If the validity mask is not one-dimensional or does not select exactly
            one valid entry per subset cell.
        """
        # ``load_unstruct_grid_2d(..., with_subsets=True)`` returns the index
        # array and its source validity mask together. Accept that pair directly
        # for convenience while retaining the normal ``indices`` API.
        if valid_data_mask is None and isinstance(indices, tuple) and len(indices) == 2:
            candidate_mask = np.asarray(indices[1])
            if candidate_mask.ndim == 1 and candidate_mask.dtype == np.bool_:
                indices, valid_data_mask = cast(Any, indices[0]), candidate_mask

        index_array = as_index_array(indices)

        if valid_data_mask is None:
            valid_data_mask = np.ones(index_array.size, dtype=np.bool_)
        else:
            valid_data_mask = np.asarray(valid_data_mask, dtype=np.bool_)
        if valid_data_mask.ndim != 1:
            raise ValueError("valid_data_mask must be one-dimensional.")
        if np.count_nonzero(valid_data_mask) != index_array.size:
            raise ValueError(
                "The number of valid data entries must match the number of subset cells."
            )

        grid = UnstructGrid2D.__new__(UnstructGrid2D)

        grid._name = name or self.name + " subset"
        grid._coordinate_system = self._coordinate_system
        grid._dimension = self._dimension
        grid._scalar_interpolator = None
        grid._vector_interpolator = None
        grid._triangulation = None
        grid._valid_data_mask = np.array(valid_data_mask, dtype=np.bool_, copy=True)
        grid._valid_data_mask.setflags(write=False)

        index_list = [int(i) for i in index_array]
        cells_original: tuple[NDArray[np.intp], ...] = tuple(
            self.cells[i] for i in index_list
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
        cells: list[NDArray[np.intp]] = []  # and split
        i_start = 0
        for cell in cells_original:
            num_vertices = int(cell.shape[0])
            cells.append(inv_index[i_start : i_start + num_vertices].astype(np.intp))
            i_start += num_vertices
        grid._cells = tuple(cells)
        grid._num_cell = len(grid._cells)
        ntri_total = sum(len(cell) - 2 for cell in cells)

        # cell area and centres of this subset
        grid._cell_area = np.array(self.cell_area[index_array])
        grid._cell_area.setflags(write=False)
        grid._cell_centre = np.array(self.cell_centre[index_array])
        grid._cell_centre.setflags(write=False)
        if self._coordinate_system == "cylindrical":
            grid._cell_volume = np.array(self.cell_volume[index_array])
            grid._cell_volume.setflags(write=False)

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

        c2t_map = self.cell_to_triangle_map[index_array]  # map with original triangle indices
        # maps original vertices to the subset, -1 if not in the subset
        subset_vertex_map = -1 * np.ones(self.vertices.shape[0], dtype=np.int32)
        subset_vertex_map[vert_index] = np.arange(vert_index.size, dtype=np.int32)

        itri = 0
        for i, cell in enumerate(cells):
            ntri = len(cell) - 2
            if ntri == 1:
                grid._triangles[itri] = cell
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
    def interpolator(
        self,
        grid_data: CellData,
        fill_value: float = 0,
        *,
        interpolator_cache: InterpolatorCacheMode = "memory",
        interpolator_cache_dir: str | Path | None = None,
        interpolator_cache_namespace: str = "ggd",
    ) -> UnstructGridFunction2D:
        """Return an `UnstructGridFunction2D` interpolator instance for the data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance of the
        previously created interpolator sharing the same KDtree structure.

        Parameters
        ----------
        grid_data
            Array containing data in the grid cells.
        fill_value
            Value returned outside the grid, by default 0.0.
        interpolator_cache
            Cache mode for the interpolator, by default ``"memory"``.
            The cache mode is described in the `.InterpolatorCacheMode` type alias.
        interpolator_cache_dir
            Directory used when ``interpolator_cache="disk"``, by default None
            (uses the system cache directory, e.g., ``~/.cache/cherab/imas/interpolators``).
        interpolator_cache_namespace
            Namespace prefix to avoid cache-key collisions, by default ``"ggd"``.

        Returns
        -------
        `.UnstructGridFunction2D`
            Interpolator instance.
        """
        grid_data = _as_cell_data(grid_data, self._valid_data_mask)
        return self._build_cached_interpolator(
            interpolator_cls=UnstructGridFunction2D,
            template_builder=lambda: UnstructGridFunction2D(
                self._vertices,
                self._triangles,
                self._triangle_to_cell_map,
                np.zeros(self._num_cell, dtype=np.float64),
                0.0,
            ),
            data=grid_data,
            fill=fill_value,
            template_data=np.zeros(self._num_cell, dtype=np.float64),
            template_fill=0.0,
            cached_slot="_scalar_interpolator",
            mode=interpolator_cache,
            cache_dir=interpolator_cache_dir,
            namespace=interpolator_cache_namespace,
        )

    @override
    def vector_interpolator(
        self,
        grid_vectors: NDArray[np.float64],
        fill_vector: Vector3D = ZERO_VECTOR,
        *,
        interpolator_cache: InterpolatorCacheMode = "memory",
        interpolator_cache_dir: str | Path | None = None,
        interpolator_cache_namespace: str = "ggd",
    ) -> UnstructGridVectorFunction2D:
        """Return an `UnstructGridVectorFunction2D` interpolator instance for the vector data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance of the
        previously created interpolator sharing the same KDtree structure.

        Parameters
        ----------
        grid_vectors
            ``(3, K)`` Array containing 3D vectors in the grid cells.
        fill_vector
            3D vector returned outside the grid, by default `Vector3D(0, 0, 0)`.
        interpolator_cache
            Cache mode for the interpolator, by default ``"memory"``.
            The cache mode is described in the `.InterpolatorCacheMode` type alias.
        interpolator_cache_dir
            Directory used when ``interpolator_cache="disk"``, by default None
            (uses the system cache directory, e.g., ``~/.cache/cherab/imas/interpolators``).
        interpolator_cache_namespace
            Namespace prefix to avoid cache-key collisions, by default ``"ggd"``.

        Returns
        -------
        `.UnstructGridVectorFunction2D`
            Interpolator instance.
        """
        return self._build_cached_interpolator(
            interpolator_cls=UnstructGridVectorFunction2D,
            template_builder=lambda: UnstructGridVectorFunction2D(
                self._vertices,
                self._triangles,
                self._triangle_to_cell_map,
                np.zeros((3, self._num_cell), dtype=np.float64),
                ZERO_VECTOR,
            ),
            data=grid_vectors,
            fill=fill_vector,
            template_data=np.zeros((3, self._num_cell), dtype=np.float64),
            template_fill=ZERO_VECTOR,
            cached_slot="_vector_interpolator",
            mode=interpolator_cache,
            cache_dir=interpolator_cache_dir,
            namespace=interpolator_cache_namespace,
        )

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
            "valid_data_mask": self._valid_data_mask,
        }
        return state

    def __setstate__(self, state):
        """Restore the state of the UnstructGrid2D instance from the serialized state."""
        self._name = state["name"]
        self._dimension = state["dimension"]
        self._coordinate_system = state["coordinate_system"]
        self._vertices = state["vertices"]
        self._vertices.setflags(write=False)
        self._cells = tuple(np.asarray(cell, dtype=np.intp) for cell in state["cells"])
        self._valid_data_mask = np.asarray(
            state.get("valid_data_mask", np.ones(len(self._cells), dtype=np.bool_)),
            dtype=np.bool_,
        )
        self._valid_data_mask.setflags(write=False)

        self._initial_setup()

    def plot_tri_mesh(
        self,
        data: CellData,
        ax: matplotlib.axes.Axes | None = None,
        cmap: str = "viridis",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot cell data on the triangular mesh using Matplotlib's tripcolor.

        Parameters
        ----------
        data
            Data array defined on the polygonal mesh. Each cell value is assigned to all
            triangles forming that cell.
        ax
            Matplotlib axes to plot on. If None, a new figure and axes are created.
        cmap
            Colormap to use for the data, by default ``"viridis"``.
        **kwargs
            Additional keyword arguments passed to `~matplotlib.axes.Axes.tripcolor`.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes with the plotted mesh.
        """
        data_array = _as_cell_data(data, self._valid_data_mask)
        triangle_data = data_array[self._triangle_to_cell_map]

        if self._triangulation is None:
            self._triangulation = Triangulation(
                self._vertices[:, 0], self._vertices[:, 1], self._triangles
            )

        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        ax.set_aspect(1)
        ax.tripcolor(
            self._triangulation,
            facecolors=triangle_data,
            cmap=cmap,
            **kwargs,
        )
        ax.set_xlim(self._mesh_extent["xmin"], self._mesh_extent["xmax"])
        ax.set_ylim(self._mesh_extent["ymin"], self._mesh_extent["ymax"])

        if self._coordinate_system == "cartesian":
            ax.set_xlabel("$X$ [m]")
            ax.set_ylabel("$Y$ [m]")
        elif self._coordinate_system == "cylindrical":
            ax.set_xlabel("$R$ [m]")
            ax.set_ylabel("$Z$ [m]")

        return ax

    @override
    def plot_mesh(
        self,
        data: CellData | None = None,
        ax: matplotlib.axes.Axes | None = None,
        **grid_styles,
    ) -> matplotlib.axes.Axes:
        """Plot the polygonal mesh grid geometry to a matplotlib figure.

        Parameters
        ----------
        data
            Data array defined on the polygonal mesh.
        ax
            Matplotlib axes to plot on. If None, a new figure and axes are created.
        **grid_styles
            Styles for the grid lines and faces,
            by default ``{"facecolor": "none", "edgecolor": "b", "linewidth": 0.25}``.
            If data is provided, the styles are not applied to the grid lines and faces to allow
            the data colormap to be visible.

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
            collection_mesh.set_array(_as_cell_data(data, self._valid_data_mask))
        ax.add_collection(collection_mesh)
        ax.set_aspect(1)
        ax.set_xlim(self._mesh_extent["xmin"], self._mesh_extent["xmax"])
        ax.set_ylim(self._mesh_extent["ymin"], self._mesh_extent["ymax"])

        if self._coordinate_system == "cartesian":
            ax.set_xlabel("$X$ [m]")
            ax.set_ylabel("$Y$ [m]")
        elif self._coordinate_system == "cylindrical":
            ax.set_xlabel("$R$ [m]")
            ax.set_ylabel("$Z$ [m]")

        return ax
