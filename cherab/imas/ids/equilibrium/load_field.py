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

from imas.imasdef import EMPTY_DOUBLE, EMPTY_INT


RECTANGULAR_GRID = 1


def load_magnetic_field_data(profiles_2d):
    """
    Loads 2D profiles of the magnetic field components from the profiles_2d structure of the equilibirum IDS.

    :param profiles_2d: The profiles_2d structure of the equilibirum IDS.

    :returns: A dictionary with the following keys:
        'r': A N-size array with R coordinates of rectangular grid,
        'z': A M-size array with Z coordinates of rectangular grid,
        'b_field_r': A NxM  array of shape (N, M) with R component of the magnetic field,
        'b_field_z': A NxM  array of shape (N, M) with Z component of the magnetic field,
        'b_field_tor': A NxM  array of shape (N, M) with toroidal component of the magnetic field,
    """

    rectangular_grid = False
    for prof2d in profiles_2d:
        if prof2d.grid_type.index == RECTANGULAR_GRID or prof2d.grid_type.index == EMPTY_INT:
            rectangular_grid = True
            break
    
    if not rectangular_grid:
        raise RuntimeError("Unable to read magnetic field: rectangular grid for 2D profiles is not found and other grid types are not supported.")

    b_dict = {}

    b_dict['r'] = np.array(prof2d.grid.dim1)
    b_dict['z'] = np.array(prof2d.grid.dim2)
    shape = (b_dict['r'].size, b_dict['z'].size)

    b_dict['b_field_r'] = np.array(prof2d.b_field_r)
    b_dict['b_field_tor'] = np.array(prof2d.b_field_tor)
    b_dict['b_field_z'] = np.array(prof2d.b_field_z)

    if b_dict['b_field_r'].shape != shape or b_dict['b_field_tor'].shape != shape or b_dict['b_field_z'].shape != shape:
        raise RuntimeError("Unable to read magnetic field: the shape of the magnetic field components does not match the grid shape.")

    return b_dict
