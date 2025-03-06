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

import numpy as np

from raysect.core.math.vector cimport new_vector3d
from raysect.core.math.point cimport new_point3d

cimport cython


cdef class UnstructGridFunction3D(Function3D):
    """
    A simple interpolator for the data defined on the 3D unstructured grid.
    Finds the cell containing the point (x, y, z) using the KDtree algorithm.
    Returns the data value for this cell or the `fill_value` if the grid
    does not contain the point.

    :param object vertex_coords: 3D (N,3) array-like with the vertex coordinates of tetrahedra.
    :param object tetrahedra: 3D (M,4) integer array-like with the vertex indices forming
        the tetrahedra.
    :param object tetra_to_cell_map: 1D (M,) integer array-like with the indices of
        the grid cells (cube) containing the tetrahedra.
    :param ndarray grid_data: An array containing data in the grid cells.
    :param double fill_value: A value returned outside the gird. Default is 0.
    """

    def __init__(self, object vertex_coords not None, object tetrahedra not None, object tetra_to_cell_map not None,
                 np.ndarray grid_data not None, double fill_value=0):

        vertex_coords = np.array(vertex_coords, dtype=np.float64)
        tetrahedra = np.array(tetrahedra, dtype=np.int32)
        tetra_to_cell_map = np.array(tetra_to_cell_map, dtype=np.int32)

        # build kdtree
        self._kdtree = MeshKDTree3D(vertex_coords, tetrahedra)

        self._tetra_to_cell_map = tetra_to_cell_map
        self._tetra_to_cell_map_mv = self._tetra_to_cell_map

        # Attention!!! Do not copy grid_data! Attribute self._grid_data must point to the original data array,
        # so as not to re-initialize the interpolator if the user changes data values.

        # populate internal attributes
        self._grid_data = grid_data
        self._fill_value = fill_value

        self._grid_data_mv = self._grid_data

    def __getstate__(self):
        return self._grid_data, self._fill_value, self._tetra_to_cell_map, self._kdtree

    def __setstate__(self, state):
        self._grid_data, self._fill_value, self._tetra_to_cell_map, self._kdtree = state
        self._tetra_to_cell_map_mv = self._tetra_to_cell_map
        self._grid_data_mv = self._grid_data

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @classmethod
    def instance(cls, object instance not None, np.ndarray grid_data=None, object fill_value=None):
        """
        Creates a new interpolator instance from an existing UnstructGridFunction3D
        or UnstructGridVectorFunction3D instance.
        The new interpolator instance will share the same internal acceleration
        data as the original interpolator. The grid_data of the new instance can
        be redefined.
        This method should be used if the user has multiple datasets
        that lie on the same mesh geometry. Using this methods avoids the
        repeated rebuilding of the mesh acceleration structures by sharing the
        geometry data between multiple interpolator objects.

        If created from the UnstructGridVectorFunction3D instance,
        the grid_data and the fill_value must not be None.

        :param object instance: UnstructGridFunction3D or UnstructGridVectorFunction3D object.
        :param ndarray grid_data: An array containing data in the grid cells.
        :param object fill_value: A value returned outside the gird.
        :return: A UnstructGridFunction3D object.
        :rtype: UnstructGridFunction3D
        """

        cdef UnstructGridFunction3D m, inst
        cdef UnstructGridVectorFunction3D instvec

        m = UnstructGridFunction3D.__new__(UnstructGridFunction3D)

        if isinstance(instance, UnstructGridFunction3D):
            inst = instance
            # copy source data
            m._kdtree = inst._kdtree
            m._tetra_to_cell_map = inst._tetra_to_cell_map

            # replace grid data and fill value
            m._grid_data = inst._grid_data if grid_data is None else grid_data
            m._fill_value = inst._fill_value if fill_value is None else <double>fill_value
        elif isinstance(instance, UnstructGridVectorFunction3D):
            instvec = instance
            m._kdtree = instvec._kdtree
            m._tetra_to_cell_map = instvec._tetra_to_cell_map

            if grid_data is None:
                raise ValueError("Argument 'grid_data' must not be None if the new instant UnstructGridFunction3D is created from the UnstructGridVectorFunction3D instance.")
            if fill_value is None:
                raise ValueError("Argument 'fill_value' must not be None if the new instant UnstructGridFunction3D is created from the UnstructGridVectorFunction3D instance.")
            m._grid_data = grid_data
            m._fill_value = <double>fill_value
        else:
            raise TypeError("Argument 'instance' must be either UnstructGridFunction3D or UnstructGridVectorFunction3D instance.")

        m._tetra_to_cell_map_mv = m._tetra_to_cell_map
        m._grid_data_mv = m._grid_data

        return m

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef:
            np.int32_t tetrahedra_id, icell

        if self._kdtree.is_contained(new_point3d(x, y, z)):

            tetrahedra_id = self._kdtree.tetrahedra_id
            icell = self._tetra_to_cell_map_mv[tetrahedra_id]
            return self._grid_data_mv[icell]

        return self._fill_value

cdef class UnstructGridVectorFunction3D(VectorFunction3D):
    """
    A simple vector interpolator for the data defined on the 3D unstructured grid.
    Finds the cell containing the point (x, y, z) using the KDtree algorithm.
    Returns the 3D vector value for this cell or the `fill_vector` if the grid
    does not contain the point.

    :param object vertex_coords: 3D (N,3) array-like with the vertex coordinates of tetrahedra.
    :param object tetrahedra: 3D (M,4) integer array-like with the vertex indices forming
        the tetrahedra.
    :param object tetra_to_cell_map: 1D (M,) integer array-like with the indices of
        the grid cells (cube) containing the tetrahedra.
    :param ndarray grid_vectors: A (3,K) array containing 3D vectors in the grid cells.
    :param Vector3D fill_vector: A 3D vector returned outside the gird. Default is (0, 0, 0).
    """

    def __init__(self, object vertex_coords not None, object tetrahedra not None, object tetra_to_cell_map not None,
                 np.ndarray grid_vectors not None, Vector3D fill_vector=Vector3D(0, 0, 0)):

        vertex_coords = np.array(vertex_coords, dtype=np.float64)
        tetrahedra = np.array(tetrahedra, dtype=np.int32)
        tetra_to_cell_map = np.array(tetra_to_cell_map, dtype=np.int32)

        # build kdtree
        self._kdtree = MeshKDTree3D(vertex_coords, tetrahedra)

        self._tetra_to_cell_map = tetra_to_cell_map
        self._tetra_to_cell_map_mv = self._tetra_to_cell_map

        # Attention!!! Do not copy grid_vectors! Attribute self._grid_vectors must point to the original data array,
        # so as not to re-initialize the interpolator if the user changes data values.

        # populate internal attributes
        self._grid_vectors = grid_vectors
        self._fill_vector = fill_vector

        self._grid_vectors_mv = self._grid_vectors

    def __getstate__(self):
        return self._grid_vectors, self._fill_vector, self._tetra_to_cell_map, self._kdtree

    def __setstate__(self, state):
        self._grid_vectors, self._fill_vector, self._tetra_to_cell_map, self._kdtree = state
        self._grid_vectors_mv = self._grid_vectors
        self._tetra_to_cell_map_mv = self._tetra_to_cell_map

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @classmethod
    def instance(cls, object instance not None, np.ndarray grid_vectors=None,
                 Vector3D fill_vector=None):
        """
        Creates a new interpolator instance from an existing UnstructGridVectorFunction3D
        or UnstructGridFunction3D instance.
        The new interpolator instance will share the same internal acceleration
        data as the original interpolator. The grid_vectors of the new instance can
        be redefined.
        This method should be used if the user has multiple datasets
        that lie on the same mesh geometry. Using this methods avoids the
        repeated rebuilding of the mesh acceleration structures by sharing the
        geometry data between multiple interpolator objects.

        If created from the UnstructGridFunction3D instance,
        the grid_vectors and the fill_vector must not be None.

        :param object instance: UnstructGridVectorFunction3D or UnstructGridFunction3D object.
        :param ndarray grid_vectors: An array containing vector grid data.
        :return: UnstructGridVectorFunction3D object.
        :rtype: UnstructGridVectorFunction3D
        """

        cdef UnstructGridVectorFunction3D m, instvec
        cdef UnstructGridFunction3D inst

        m = UnstructGridVectorFunction3D.__new__(UnstructGridVectorFunction3D)

        if isinstance(instance, UnstructGridVectorFunction3D):
            instvec = instance
            # copy source data
            m._kdtree = instvec._kdtree
            m._tetra_to_cell_map = instvec._tetra_to_cell_map

            # replace grid vector and fill vector
            m._grid_vectors = instvec._grid_vectors if grid_vectors is None else grid_vectors
            m._fill_vector = instvec._fill_vector if fill_vector is None else fill_vector
        elif isinstance(instance, UnstructGridFunction3D):
            inst = instance
            m._kdtree = inst._kdtree
            m._tetra_to_cell_map = inst._tetra_to_cell_map

            if grid_vectors is None:
                raise ValueError("Argument 'grid_vectors' must not be None if the new instant UnstructGridVectorFunction3D is created from the UnstructGridFunction3D instance.")
            if fill_vector is None:
                raise ValueError("Argument 'fill_vector' must not be None if the new instant UnstructGridVectorFunction3D is created from the UnstructGridFunction3D instance.")
            m._grid_vectors = grid_vectors
            m._fill_vector = fill_vector
        else:
            raise TypeError("Argument 'instance' must be either UnstructGridFunction3D or UnstructGridVectorFunction3D instance.")

        m._tetra_to_cell_map_mv = m._tetra_to_cell_map
        m._grid_vectors_mv = m._grid_vectors

        return m

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef Vector3D evaluate(self, double x, double y, double z):

        cdef:
            np.int32_t tetrahedra_id, icell
            double vx, vy, vz

        if self._kdtree.is_contained(new_point3d(x, y, z)):

            tetrahedra_id = self._kdtree.tetrahedra_id
            icell = self._tetra_to_cell_map_mv[tetrahedra_id]
            vx = self._grid_vectors_mv[0, icell]
            vy = self._grid_vectors_mv[1, icell]
            vz = self._grid_vectors_mv[2, icell]

            return new_vector3d(vx, vy, vz)

        return self._fill_vector
