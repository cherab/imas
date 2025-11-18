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
"""Interpolators for structured and unstructured grid functions.

This module provides interpolation classes for 2D and 3D grid-based functions, supporting both
structured and unstructured grids. These interpolators are designed to work with IMAS data
structures and provide efficient evaluation of scalar and vector functions defined on various grid
types.
"""

from .struct_grid_2d_functions import StructGridFunction2D, StructGridVectorFunction2D
from .struct_grid_3d_functions import StructGridFunction3D, StructGridVectorFunction3D
from .unstruct_grid_2d_functions import UnstructGridFunction2D, UnstructGridVectorFunction2D
from .unstruct_grid_3d_functions import UnstructGridFunction3D, UnstructGridVectorFunction3D

__all__ = [
    "StructGridFunction2D",
    "StructGridVectorFunction2D",
    "StructGridFunction3D",
    "StructGridVectorFunction3D",
    "UnstructGridFunction2D",
    "UnstructGridVectorFunction2D",
    "UnstructGridFunction3D",
    "UnstructGridVectorFunction3D",
]
