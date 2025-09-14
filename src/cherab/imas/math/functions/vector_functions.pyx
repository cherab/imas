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
"""Module defining unit vector functions."""

from raysect.core.math cimport Vector3D
from raysect.core.math.function.vector3d cimport autowrap_function1d as autowrap_vectorfunction1d
from raysect.core.math.function.vector3d cimport autowrap_function2d as autowrap_vectorfunction2d
from raysect.core.math.function.vector3d cimport autowrap_function3d as autowrap_vectorfunction3d

__all__ = ["UnitVector1D", "UnitVector2D", "UnitVector3D"]


cdef class UnitVector1D(VectorFunction1D):
    """Evaluates a unit vector for the given VectorFunction1D instance."""

    def __init__(self, object vector):
        self._vector = autowrap_vectorfunction1d(vector)

    cdef Vector3D evaluate(self, double x):

        return self._vector.evaluate(x).normalise()


cdef class UnitVector2D(VectorFunction2D):
    """Evaluates a unit vector for the given VectorFunction2D instance."""

    def __init__(self, object vector):
        self._vector = autowrap_vectorfunction2d(vector)

    cdef Vector3D evaluate(self, double x, double y):

        return self._vector.evaluate(x, y).normalise()


cdef class UnitVector3D(VectorFunction3D):
    """Evaluates a unit vector for the given VectorFunction3D instance."""

    def __init__(self, object vector):
        self._vector = autowrap_vectorfunction3d(vector)

    cdef Vector3D evaluate(self, double x, double y, double z):

        return self._vector.evaluate(x, y, z).normalise()
