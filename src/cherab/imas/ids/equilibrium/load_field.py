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
"""Module for loading magnetic field data from the equilibrium IDS."""

import numpy as np

from imas.ids_defs import EMPTY_INT
from imas.ids_struct_array import IDSStructArray

RECTANGULAR_GRID = 1

__all__ = ["load_magnetic_field_data"]


def load_magnetic_field_data(profiles_2d: IDSStructArray) -> dict:
    """Load 2D profiles of the magnetic field components from equilibrium IDS.

    The magnetic field components are extracted from the ``profiles_2d`` IDS structure,
    assuming that the profiles are defined on a rectangular grid.

    Parameters
    ----------
    profiles_2d
        The ``profiles_2d`` structure of the equilibrium IDS.

    Returns
    -------
    Dictionary with the following keys:

        :r: ``(N,)`` ndarray with R coordinates of rectangular grid.
        :z: ``(M,)`` ndarray with Z coordinates of rectangular grid.
        :b_field_r: ``(N, M)`` ndarray with R component of the magnetic field.
        :b_field_z: ``(N, M)`` ndarray with Z component of the magnetic field.
        :b_field_phi: ``(N, M)`` ndarray with toroidal component of the magnetic field.

    Raises
    ------
    RuntimeError
        If unable to read the magnetic field due to unsupported grid type or
        mismatched array shapes.
    """
    rectangular_grid = False
    for prof2d in profiles_2d:
        if prof2d.grid_type.index == RECTANGULAR_GRID or prof2d.grid_type.index == EMPTY_INT:
            rectangular_grid = True
            break

    if not rectangular_grid:
        raise RuntimeError(
            "Unable to read magnetic field: "
            + "rectangular grid for 2D profiles is not found and other grid types are not supported."
        )

    b_dict = {}

    b_dict["r"] = np.asarray_chkfinite(prof2d.grid.dim1)
    b_dict["z"] = np.asarray_chkfinite(prof2d.grid.dim2)
    shape = (b_dict["r"].size, b_dict["z"].size)

    b_dict["b_field_r"] = np.asarray_chkfinite(prof2d.b_field_r)
    b_dict["b_field_phi"] = np.asarray_chkfinite(prof2d.b_field_phi)
    b_dict["b_field_z"] = np.asarray_chkfinite(prof2d.b_field_z)

    if (
        b_dict["b_field_r"].shape != shape
        or b_dict["b_field_phi"].shape != shape
        or b_dict["b_field_z"].shape != shape
    ):
        raise RuntimeError(
            "Unable to read magnetic field: "
            + "the shape of the magnetic field components does not match the grid shape."
        )

    return b_dict
