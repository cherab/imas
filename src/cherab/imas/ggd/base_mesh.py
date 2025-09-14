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

from raysect.core.math import Vector3D

ZEROVECTOR = Vector3D(0, 0, 0)


__all__ = ["GGDGrid"]


class GGDGrid:
    """Base class for general grids (GGD).

    Parameters
    ----------
    name : str
        Name of the grid, by default ''.
    dimension : int
        Grid dimensions, by default 1.
    coordinate_system : str
        Coordinate system, by default 'cartesian'.
    """

    def __init__(self, name="", dimension=1, coordinate_system="cartesian"):
        dimension = int(dimension)
        if dimension < 1:
            raise ValueError("Attribute dimension must be >= 1.")

        self._dimension = dimension
        self._name = str(name)
        self._coordinate_system = str(coordinate_system)

        self._interpolator = None
        self._cell_centre = None
        self._cell_area = None
        self._cell_volume = None
        self._mesh_extent = None
        self._num_cell = 0

        self._initial_setup()

    def _initial_setup(self):
        raise NotImplementedError("To be defined in subclass.")

    @property
    def name(self):
        """Grid name."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def dimension(self):
        """Grid dimension."""
        return self._dimension

    @property
    def num_cell(self):
        """Number of grid cells."""
        return self._num_cell

    @property
    def coordinate_system(self):
        """Coordinate system."""
        return self._coordinate_system

    @property
    def cell_centre(self):
        """Coordinate of cell centres as (num_cell, dimension) array."""
        return self._cell_centre

    @property
    def cell_area(self):
        """Cell areas as (num_cell,) array."""
        return self._cell_area

    @property
    def cell_volume(self):
        """Cell volume as (num_cell,) array."""
        return self._cell_volume

    @property
    def mesh_extent(self):
        """Extent of the mesh.

        A dictionary with xmin, xmax, ymin and ymax, ... keys.
        """
        return self._mesh_extent

    def subset(self, indices, name=None):
        """Create a subset grid from this instance.

        Parameters
        ----------
        indices : array_like
            Indices of the cells of the original grid in the subset.
        name : str, optional
            Name of the grid subset. Default is instance.name + ' subset'.
        """

        raise NotImplementedError("To be defined in subclass.")

    def interpolator(self, grid_data, fill_value=0.0):
        """Return an Function interpolator instance for the data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance of the
        previously created interpolator.

        Parameters
        ----------
        grid_data : array_like
            Array containing data in the grid cells.
        fill_value : float, optional
            A value returned outside the grid, by default is 0.0.

        Returns
        -------
        Function2D | Function3D
            Interpolator instance.
        """

        raise NotImplementedError("To be defined in subclass.")

    def vector_interpolator(self, grid_vectors, fill_vector=ZEROVECTOR):
        """Return a VectorFunction interpolator instance for the vector data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance of the
        previously created interpolator.

        Parameters
        ----------
        grid_vectors : (3, num_cell) array_like
            Array containing 3D vectors in the grid cells.
        fill_vector : Vector3D, optional
            3D vector returned outside the grid, by default Vector3D(0, 0, 0).

        Returns
        -------
        VectorFunction2D | VectorFunction3D
            Interpolator instance.
        """

        raise NotImplementedError("To be defined in subclass.")

    def plot_mesh(self, data=None, ax=None):
        """Plot the grid geometry to a matplotlib figure.

        Parameters
        ----------
        data : array_like, optional
            Data array defined on the grid.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on. If None, a new figure and axes are created.
        """

        raise NotImplementedError("To be defined in subclass.")
