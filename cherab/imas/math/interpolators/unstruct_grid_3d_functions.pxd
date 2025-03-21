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

cimport numpy as np
from raysect.core.math.vector cimport Vector3D
from raysect.core.math.function.float.function3d.interpolate.common cimport MeshKDTree3D
from raysect.core.math.function.float.function3d cimport Function3D
from raysect.core.math.function.vector3d.function3d cimport Function3D as VectorFunction3D


cdef class UnstructGridFunction3D(Function3D):

    cdef:
        MeshKDTree3D _kdtree
        np.ndarray _grid_data, _tetra_to_cell_map
        np.int32_t[::1] _tetra_to_cell_map_mv
        double[::1] _grid_data_mv
        double _fill_value

cdef class UnstructGridVectorFunction3D(VectorFunction3D):

    cdef:
        MeshKDTree3D _kdtree
        np.ndarray _grid_vectors, _tetra_to_cell_map
        np.int32_t[::1] _tetra_to_cell_map_mv
        double[:, ::1] _grid_vectors_mv
        Vector3D _fill_vector
