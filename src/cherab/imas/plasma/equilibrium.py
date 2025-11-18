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
"""Module for loading plasma equilibrium and magnetic field from the equilibrium IDS."""

import numpy as np
from raysect.core.math.function.float import Interpolator1DArray, Interpolator2DArray
from raysect.core.math.function.vector3d import FloatToVector3DFunction2D

from cherab.tools.equilibrium import EFITEquilibrium
from imas import DBEntry

from ..ids.common import get_ids_time_slice
from ..ids.equilibrium import load_equilibrium_data, load_magnetic_field_data

__all__ = ["load_equilibrium", "load_magnetic_field"]


def load_equilibrium(
    *args,
    time: float = 0,
    occurrence: int = 0,
    time_threshold: float = np.inf,
    with_psi_interpolator: bool = False,
    **kwargs,
) -> tuple[EFITEquilibrium, Interpolator1DArray | None] | EFITEquilibrium:
    """Load plasma equilibrium from the equilibrium IDS and create an `EFITEquilibrium` object.

    Parameters
    ----------
    *args
        Arguments passed to `~imas.db_entry.DBEntry`.
    time
        Time for the equilibrium, by default 0.
    occurrence
        Occurrence index of the ``equilibrium`` IDS, by default 0.
    time_threshold
        Maximum allowed difference between the requested time and the nearest available
        time, by default `numpy.inf`.
    with_psi_interpolator
        If True, returns the ``psi_norm(rho_tor_norm)`` interpolator; otherwise, returns only the
        equilibrium object.
    **kwargs
        Keyword arguments passed to `~imas.db_entry.DBEntry`.

    Returns
    -------
    equilibrium : `~cherab.tools.equilibrium.efit.EFITEquilibrium`
        The plasma equilibrium object.
    psi_interpolator : `~raysect.core.math.function.float.function1d.interpolate.Interpolator1DArray` | None
        If ``with_psi_interpolator`` is True and ``rho_tor_norm`` is available, returns the
        ``psi_norm(rho_tor_norm)`` interpolator.
        If rho_tor_norm is not available, returns None.
        Otherwise, returns only the equilibrium object.
    """
    with DBEntry(*args, **kwargs) as entry:
        equilibrium_ids = get_ids_time_slice(
            entry, "equilibrium", time=time, occurrence=occurrence, time_threshold=time_threshold
        )

    equilibrium_dict = load_equilibrium_data(equilibrium_ids)

    cocos_11to3(equilibrium_dict)

    equilibrium_dict["psi_norm"][0] = min(0, equilibrium_dict["psi_norm"][0])
    equilibrium_dict["psi_norm"][-1] = max(1.0, equilibrium_dict["psi_norm"][-1])

    f_profile = np.array([equilibrium_dict["psi_norm"], equilibrium_dict["f"]])
    q_profile = np.array([equilibrium_dict["psi_norm"], equilibrium_dict["q"]])

    equilibrium = EFITEquilibrium(
        equilibrium_dict["r"],
        equilibrium_dict["z"],
        equilibrium_dict["psi_grid"],
        equilibrium_dict["psi_axis"],
        equilibrium_dict["psi_lcfs"],
        equilibrium_dict["magnetic_axis"],
        equilibrium_dict["x_points"],
        equilibrium_dict["strike_points"],
        f_profile,
        q_profile,
        equilibrium_dict["b_vacuum_radius"],
        equilibrium_dict["b_vacuum_magnitude"],
        equilibrium_dict["lcfs_polygon"],
        None,
        equilibrium_dict["time"],
    )

    if not with_psi_interpolator:
        return equilibrium

    if equilibrium_dict["rho_tor_norm"] is None:
        return equilibrium, None

    psi_interpolator = Interpolator1DArray(
        equilibrium_dict["rho_tor_norm"], equilibrium_dict["psi_norm"], "cubic", "none", 0
    )

    return equilibrium, psi_interpolator


def load_magnetic_field(
    *args,
    time: float = 0,
    occurrence: int = 0,
    time_threshold: float = np.inf,
    **kwargs,
) -> FloatToVector3DFunction2D:
    """Load the magnetic field from the equilibrium IDS and returns a VectorFunction2D interpolator.

    Parameters
    ----------
    *args
        Arguments passed to `~imas.db_entry.DBEntry`.
    time
        Time for the equilibrium, by default 0.
    occurrence
        Occurrence index of the ``equilibrium`` IDS, by default 0.
    time_threshold
        Maximum allowed difference between the requested time and the nearest available
        time, by default `numpy.inf`.
    **kwargs
        Keyword arguments passed to `~imas.db_entry.DBEntry`.

    Returns
    -------
    `~raysect.core.math.function.vector3d.function2d.base.Function2D`
        The magnetic field interpolator.

    Raises
    ------
    RuntimeError
        If the equilibrium IDS does not have a time slice.
    """
    with DBEntry(*args, **kwargs) as entry:
        equilibrium_ids = get_ids_time_slice(
            entry, "equilibrium", time=time, occurrence=occurrence, time_threshold=time_threshold
        )

    if not len(equilibrium_ids.time_slice):
        raise RuntimeError("Equilibrium IDS does not have a time slice.")

    b_dict = load_magnetic_field_data(equilibrium_ids.time_slice[0].profiles_2d)

    br = Interpolator2DArray(b_dict["r"], b_dict["z"], b_dict["b_field_r"], "cubic", "none", 0, 0)
    btor = Interpolator2DArray(
        b_dict["r"], b_dict["z"], b_dict["b_field_phi"], "cubic", "none", 0, 0
    )
    bz = Interpolator2DArray(b_dict["r"], b_dict["z"], b_dict["b_field_z"], "cubic", "none", 0, 0)

    return FloatToVector3DFunction2D(br, btor, bz)


def cocos_11to3(equilibrium_dict: dict) -> None:
    """Convert from COCOS 11 convention used in IMAS to COCOS 3 convention used in EFIT.

    This function modifies the equilibrium_dict in place to convert the coordinates
    and other relevant data from the COCOS 11 convention to the COCOS 3 convention.

    Parameters
    ----------
    equilibrium_dict
        The equilibrium data dictionary to modify in place.
    """
    equilibrium_dict["psi_grid"] = -equilibrium_dict["psi_grid"] / (2.0 * np.pi)
    equilibrium_dict["psi_axis"] = -equilibrium_dict["psi_axis"] / (2.0 * np.pi)
    equilibrium_dict["psi_lcfs"] = -equilibrium_dict["psi_lcfs"] / (2.0 * np.pi)
    equilibrium_dict["q"] = -equilibrium_dict["q"]
    if equilibrium_dict["phi"] is not None:
        equilibrium_dict["phi"] = -equilibrium_dict["phi"]
