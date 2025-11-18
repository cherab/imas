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
"""Module defining the base class for general grids (GGD)."""

from __future__ import annotations

from typing import Literal

import matplotlib.axes
import numpy as np
from numpy.typing import ArrayLike, NDArray
from raysect.core.math.function.float import Function2D, Function3D
from raysect.core.math.function.vector3d.function2d import Function2D as VectorFunction2D
from raysect.core.math.function.vector3d.function3d import Function3D as VectorFunction3D
from raysect.core.math.vector import Vector3D

ZEROVECTOR = Vector3D(0, 0, 0)


__all__ = ["GGDGrid"]


class GGDGrid:
    """Base class for general grids (GGD).

    Parameters
    ----------
    name
        Name of the grid.
    dimension
        Grid dimensions, by default 1.
    coordinate_system
        Coordinate system, by default ``"cartesian"``.
    """

    def __init__(
        self,
        name: str = "",
        dimension: int = 1,
        coordinate_system: Literal["cylindrical", "cartesian"] = "cartesian",
    ):
        dimension = int(dimension)
        if dimension < 1:
            raise ValueError("Attribute dimension must be >= 1.")

        self._dimension: int = dimension
        self._name: str = str(name)
        self._coordinate_system: str = str(coordinate_system)

        self._interpolator: object | None = None
        self._cell_centre: NDArray[np.float64] | None = None
        self._cell_area: NDArray[np.float64] | None = None
        self._cell_volume: NDArray[np.float64] | None = None
        self._mesh_extent: dict[str, float] | None = None
        self._num_cell: int = 0

        self._initial_setup()

    def _initial_setup(self) -> None:
        raise NotImplementedError("To be defined in subclass.")

    @property
    def name(self) -> str:
        """Grid name."""
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = str(value)

    @property
    def dimension(self) -> int:
        """Grid dimension."""
        return self._dimension

    @property
    def num_cell(self) -> int:
        """Number of grid cells."""
        return self._num_cell

    @property
    def coordinate_system(self) -> str:
        """Coordinate system."""
        return self._coordinate_system

    @property
    def cell_centre(self) -> NDArray[np.float64] | None:
        """Coordinate of cell centres as ``(num_cell, dimension)`` array."""
        return self._cell_centre

    @property
    def cell_area(self) -> NDArray[np.float64] | None:
        """Cell areas as ``(num_cell,)`` array."""
        return self._cell_area

    @property
    def cell_volume(self) -> NDArray[np.float64] | None:
        """Cell volume as ``(num_cell,)`` array."""
        return self._cell_volume

    @property
    def mesh_extent(self) -> dict[str, float] | None:
        """Extent of the mesh.

        A dictionary with xmin, xmax, ymin and ymax, ... keys.
        """
        return self._mesh_extent

    def subset(self, indices: ArrayLike, name: str | None = None) -> GGDGrid:
        """Create a subset grid from this instance.

        Parameters
        ----------
        indices
            Indices of the cells of the original grid in the subset.
        name
            Name of the grid subset. Default is ``instance.name + " subset"``.

        Returns
        -------
        GGDGrid
            Subset grid instance.
        """
        raise NotImplementedError("To be defined in subclass.")

    def interpolator(
        self, grid_data: ArrayLike, fill_value: float = 0.0
    ) -> Function2D | Function3D:
        """Return an Function interpolator instance for the data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance of the
        previously created interpolator.

        Parameters
        ----------
        grid_data
            Array containing data in the grid cells.
        fill_value
            A value returned outside the grid, by default is 0.0.

        Returns
        -------
        `Function2D` | `Function3D`
            Interpolator instance.
        """
        raise NotImplementedError("To be defined in subclass.")

    def vector_interpolator(
        self, grid_vectors: ArrayLike, fill_vector: Vector3D = ZEROVECTOR
    ) -> VectorFunction2D | VectorFunction3D:
        """Return a VectorFunction interpolator instance for the vector data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance of the
        previously created interpolator.

        Parameters
        ----------
        grid_vectors
            ``(3, num_cell)`` Array containing 3D vectors in the grid cells.
        fill_vector
            3D vector returned outside the grid, by default ``Vector3D(0, 0, 0)``.

        Returns
        -------
        `VectorFunction2D` | `VectorFunction3D`
            Interpolator instance.
        """
        raise NotImplementedError("To be defined in subclass.")

    def plot_mesh(self, data: ArrayLike | None = None, ax: matplotlib.axes.Axes | None = None):
        """Plot the grid geometry to a matplotlib figure.

        Parameters
        ----------
        data
            Data array defined on the grid.
        ax
            Matplotlib axes to plot on. If None, a new figure and axes are created.
        """
        raise NotImplementedError("To be defined in subclass.")
