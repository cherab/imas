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
"""Module for loading 2D plasma equilibrium data from the equilibrium IDS."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from raysect.core.math import Point2D

from imas.ids_defs import EMPTY_FLOAT, EMPTY_INT
from imas.ids_toplevel import IDSToplevel

RECTANGULAR_GRID = 1

__all__ = ["load_equilibrium_data"]


def load_equilibrium_data(
    equilibrium_ids: IDSToplevel,
) -> dict[str, NDArray[np.float64] | float | Point2D | list[Point2D] | None]:
    """Load 2D plasma equilibrium data from the equilibrium IDS.

    Parameters
    ----------
    equilibrium_ids
        The time-slice of the equilibrium IDS.

    Returns
    -------
    dict[str, Any]
        Dictionary with the following keys and values:

        :r: ``(N, )`` ndarray with R coordinates of rectangular grid,
        :z: ``(M, )`` ndarray with Z coordinates of rectangular grid,
        :psi_grid: ``(N, M)`` ndarray with psi grid values,
        :psi_axis: The psi value at the magnetic axis,
        :psi_lcfs: The psi value at the LCFS,
        :magnetic_axis: Point2D containing the coordinates of the magnetic axis,
        :x_points: list or tuple of Point2D x-points,
        :strike_points: list or tuple of Point2D strike-points,
        :psi_norm: ``(K, )`` ndarray with the values of psi_norm,
        :f: ``(K, )`` ndarray with the current flux profile on psi_norm,
        :q: ``(K, )`` ndarray with the safety factor (q) profile on psi_norm,
        :phi: ``(K, )`` ndarray with the toroidal flux profile on psi_norm,
        :rho_tor_norm: ``(K, )`` ndarray with the normalised toroidal flux coordinate on psi_norm,
        :b_vacuum_radius: Vacuum B-field reference radius (in meters),
        :b_vacuum_magnitude: Vacuum B-Field magnitude at the reference radius,
        :lcfs_polygon: ``(2, L)`` ndarray of ``[[x0, ...], [y0, ...]]`` vertices specifying the LCFS
            boundary
        :time: The time stamp of the time-slice (in seconds).

    Raises
    ------
    RuntimeError
        If the required data is not available in the IDS.
    """
    if not len(equilibrium_ids.time_slice):
        raise RuntimeError("Equilibrium IDS does not have a time slice.")

    profiles_2d = equilibrium_ids.time_slice[0].profiles_2d

    rectangular_grid = False
    for prof2d in profiles_2d:
        if prof2d.grid_type.index == RECTANGULAR_GRID or prof2d.grid_type.index == EMPTY_INT:
            rectangular_grid = True
            break

    if not rectangular_grid:
        raise RuntimeError(
            "Unable to read equilibrium: rectangular grid for psi profile is not found and other grid types are not supported."
        )

    r = np.asarray_chkfinite(prof2d.grid.dim1)
    z = np.asarray_chkfinite(prof2d.grid.dim2)

    psi_grid = np.asarray_chkfinite(prof2d.psi)
    if psi_grid.shape != (r.size, z.size):
        raise RuntimeError(
            "Unable to read equilibrium: the shape of 'profiles_2d[i].psi' does not match the grid shape."
        )

    profiles_1d = equilibrium_ids.time_slice[0].profiles_1d

    psi1d = np.asarray_chkfinite(profiles_1d.psi)
    if not psi1d.size:
        raise RuntimeError("Unable to read equilibrium: 'profiles_1d.psi' is empty.")

    if not len(profiles_1d.f):
        raise RuntimeError("Unable to read equilibrium: 'profiles_1d.f' is empty.")

    if not len(profiles_1d.q):
        raise RuntimeError("Unable to read equilibrium: 'profiles_1d.q' is empty.")

    global_quantities = equilibrium_ids.time_slice[0].global_quantities

    psi_axis = global_quantities.psi_axis
    if psi_axis == EMPTY_FLOAT:
        raise RuntimeError("Unable to read equilibrium: 'global_quantities.psi_axis' is not set.")
    else:
        psi_axis = float(psi_axis)

    psi_lcfs = global_quantities.psi_boundary
    if psi_lcfs == EMPTY_FLOAT:
        raise RuntimeError(
            "Unable to read equilibrium: 'global_quantities.psi_boundary' is not set."
        )
    else:
        psi_lcfs = float(psi_lcfs)

    psi_norm = (psi1d - psi_axis) / (psi_lcfs - psi_axis)
    psi_norm, index = np.unique(psi_norm, return_index=True)
    f: NDArray[np.float64] = profiles_1d.f[index]
    q: NDArray[np.float64] = profiles_1d.q[index]

    # additional 1D profiles
    phi: NDArray[np.float64] | None = profiles_1d.phi[index] if len(profiles_1d.phi) else None
    rho_tor_norm: NDArray[np.float64] | None = (
        profiles_1d.rho_tor_norm[index] if len(profiles_1d.rho_tor_norm) else None
    )
    r_axis = global_quantities.magnetic_axis.r
    z_axis = global_quantities.magnetic_axis.z
    if r_axis == EMPTY_FLOAT or z_axis == EMPTY_FLOAT:
        raise RuntimeError("Unable to read equilibrium: magnetic axis is not set.")

    magnetic_axis = Point2D(r_axis.value, z_axis.value)

    boundary = equilibrium_ids.time_slice[0].boundary

    x_points = []
    if hasattr(boundary, "x_point"):  # DD4 no longer has x_point
        for x_point in boundary.x_point:
            if x_point.r != EMPTY_FLOAT and x_point.z != EMPTY_FLOAT:
                x_points.append(Point2D(x_point.r.value, x_point.z.value))

    strike_points = []
    if hasattr(boundary, "strike_point"):  # DD4 no longer has strike_point
        for strike_point in boundary.strike_point:
            if strike_point.r != EMPTY_FLOAT and strike_point.z != EMPTY_FLOAT:
                strike_points.append(Point2D(strike_point.r.value, strike_point.z.value))

    r_lcfs = boundary.outline.r
    z_lcfs = boundary.outline.z

    lcfs_polygon = np.array([r_lcfs, z_lcfs])

    # exclude the recurring points from the polygon
    if lcfs_polygon.shape[1]:
        _, index = np.unique(lcfs_polygon, return_index=True, axis=1)
        lcfs_polygon = lcfs_polygon[:, np.sort(index)]

    if lcfs_polygon.shape[1] < 3:
        raise RuntimeError(
            "Unable to read equilibrium: boundary.outline contains less than 3 unique points."
        )

    b_vacuum_radius = equilibrium_ids.vacuum_toroidal_field.r0
    if b_vacuum_radius == EMPTY_FLOAT:
        raise RuntimeError("Unable to read equilibrium: vacuum_toroidal_field.r0 is not set.")
    else:
        b_vacuum_radius = float(b_vacuum_radius)

    b_vacuum_magnitude = equilibrium_ids.vacuum_toroidal_field.b0[0]
    if b_vacuum_magnitude == EMPTY_FLOAT:
        raise RuntimeError("Unable to read equilibrium: vacuum_toroidal_field.b0 is not set.")
    else:
        b_vacuum_magnitude = float(b_vacuum_magnitude)

    time = float(equilibrium_ids.time[0])

    return {
        "r": r,
        "z": z,
        "psi_grid": psi_grid,
        "psi_axis": psi_axis,
        "psi_lcfs": psi_lcfs,
        "magnetic_axis": magnetic_axis,
        "x_points": x_points,
        "strike_points": strike_points,
        "psi_norm": psi_norm,
        "f": f,
        "q": q,
        "phi": phi,
        "rho_tor_norm": rho_tor_norm,
        "b_vacuum_radius": b_vacuum_radius,
        "b_vacuum_magnitude": b_vacuum_magnitude,
        "lcfs_polygon": lcfs_polygon,
        "time": time,
    }
