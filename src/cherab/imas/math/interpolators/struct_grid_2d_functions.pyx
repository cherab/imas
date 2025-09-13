# cython: language_level=3

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
"""Module defining simple interpolators for data defined on a 2D structured grid."""
import numpy as np

from raysect.core.math.vector cimport new_vector3d
from raysect.core.math.cython.utility cimport find_index

cimport cython

__all__ = ["StructGridFunction2D", "StructGridVectorFunction2D"]


cdef class StructGridFunction2D(Function2D):
    """Simple interpolator for the data defined on the 2D structured grid.

    Find the cell containing the point (x, y).
    Return the data value for this cell or the `fill_value` if the points lies outside the grid.

    Parameters
    ----------
    x : (L,) array_like
        The corners of the quadrilateral cells along x axis.
    y : (M,) array_like
        The corners of the quadrilateral cells along y axis.
    grid_data : (L-1, M-1) ndarray
        Array containing data in the grid cells.
    fill_value : float, optional
        Value returned outside the gird, by default 0.
    """

    def __init__(
        self,
        object x not None,
        object y not None,
        np.ndarray grid_data not None,
        double fill_value=0,
    ):

        self._x = np.array(x, dtype=np.float64)
        self._y = np.array(y, dtype=np.float64)

        if self._x.ndim != 1:
            raise ValueError("Argument 'x' must be 1D array.")
        if self._y.ndim != 1:
            raise ValueError("Argument 'y' must be 1D array.")

        if self._x.size < 2:
            raise ValueError("Array 'x' must have at least 2 elements.")
        if self._y.size < 2:
            raise ValueError("Array 'y' must have at least 2 elements.")

        # NOTE: Attention!!! Do not copy grid_data!
        # Attribute self._grid_data must point to the original data array,
        # so as not to re-initialize the interpolator if the user changes data values.

        # populate internal attributes
        if grid_data.shape[0] != self._x.size - 1 or grid_data.shape[1] != self._y.size - 1:
            raise ValueError(
                "The shape of the grid_data array does not match the shape of the grid."
            )

        self._grid_data = grid_data
        self._fill_value = fill_value

        self._x_mv = self._x
        self._y_mv = self._y
        self._grid_data_mv = self._grid_data

    def __getstate__(self):
        return self._grid_data, self._fill_value, self._x, self._y

    def __setstate__(self, state):
        self._grid_data, self._fill_value, self._x, self._y = state
        self._x_mv = self._x
        self._y_mv = self._y
        self._grid_data_mv = self._grid_data

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double x, double y) except? -1e999:

        cdef int i_x = find_index(self._x_mv, x)
        cdef int i_y = find_index(self._y_mv, y)

        if -1 < i_x < self._x_mv.shape[0] and -1 < i_y < self._y_mv.shape[0]:
            return self._grid_data_mv[i_x, i_y]

        return self._fill_value


cdef class StructGridVectorFunction2D(VectorFunction2D):
    """Simple vector interpolator for the data defined on the 2D structured grid.

    Find the cell containing the point (x, y).
    Return the 3D vector value this cell or the `fill_vector` if the points lies outside the grid.

    Parameters
    ----------
    x : (L,) array_like
        The corners of the quadrilateral cells along x axis.
    y : (M,) array_like
        The corners of the quadrilateral cells along y axis.
    grid_vectors : (3, L-1, M-1) ndarray
        Array containing 3D vectors in the grid cells.
    fill_vector : Vector3D
        3D vector returned outside the gird, by default `Vector3D(0, 0, 0)`.
    """

    def __init__(
        self,
        object x not None,
        object y not None,
        np.ndarray grid_vectors not None,
        Vector3D fill_vector=Vector3D(0, 0, 0),
    ):

        self._x = np.array(x, dtype=np.float64)
        self._y = np.array(y, dtype=np.float64)

        if self._x.ndim != 1:
            raise ValueError("Argument 'x' must be 1D array.")
        if self._y.ndim != 1:
            raise ValueError("Argument 'y' must be 1D array.")

        if self._x.size < 2:
            raise ValueError("Array 'x' must have at least 2 elements.")
        if self._y.size < 2:
            raise ValueError("Array 'y' must have at least 2 elements.")

        # NOTE: Attention!!! Do not copy grid_vectors!
        # Attribute self._grid_vectors must point to the original data array,
        # so as not to re-initialize the interpolator if the user changes data values.

        # populate internal attributes
        if (
            grid_vectors.shape[0] != 3
            or grid_vectors.shape[1] != self._x.size - 1
            or grid_vectors.shape[2] != self._y.size - 1
        ):
            raise ValueError(
                "The shape of the grid_vectors array does not match the shape of the grid."
            )

        self._grid_vectors = grid_vectors
        self._fill_vector = fill_vector

        self._x_mv = self._x
        self._y_mv = self._y
        self._grid_vectors_mv = self._grid_vectors

    def __getstate__(self):
        return self._grid_vectors, self._fill_vector, self._x, self._y

    def __setstate__(self, state):
        self._grid_vectors, self._fill_vector, self._x, self._y = state
        self._x_mv = self._x
        self._y_mv = self._y
        self._grid_vectors_mv = self._grid_vectors

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef Vector3D evaluate(self, double x, double y):

        cdef int i_x = find_index(self._x_mv, x)
        cdef int i_y = find_index(self._y_mv, y)
        cdef double vx, vy, vz

        if -1 < i_x < self._x_mv.shape[0] and -1 < i_y < self._y_mv.shape[0]:

            vx = self._grid_vectors_mv[0, i_x, i_y]
            vy = self._grid_vectors_mv[1, i_x, i_y]
            vz = self._grid_vectors_mv[2, i_x, i_y]

            return new_vector3d(vx, vy, vz)

        return self._fill_vector
