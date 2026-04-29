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
"""Module for handling lines of sight from IMAS bolometer IDS."""

import numpy as np
from numpy.typing import NDArray
from raysect.core.math import Point3D, from_cylindrical

from imas.ids_toplevel import IDSToplevel


def get_los(ids: IDSToplevel) -> dict[str, list[tuple[Point3D, Point3D]]]:
    """Extract line of sight information from the bolometer IDS.

    Parameters
    ----------
    ids
        The IMAS IDS containing the bolometer data.

    Returns
    -------
    dict[str, list[tuple[Point3D, Point3D]]]
        Dictionary of line of sight information for each bolometer camera, keyed by camera name.
    """
    los_dict: dict[str, list[tuple[Point3D, Point3D]]] = {}
    for camera in ids.camera:
        name = str(camera.name)
        los = []
        for ch in camera.channel:
            origin = from_cylindrical(
                ch.line_of_sight.first_point.r,
                ch.line_of_sight.first_point.z,
                np.rad2deg(ch.line_of_sight.first_point.phi),
            )
            terminal = from_cylindrical(
                ch.line_of_sight.second_point.r,
                ch.line_of_sight.second_point.z,
                np.rad2deg(ch.line_of_sight.second_point.phi),
            )
            los.append((origin, terminal))

        los_dict[name] = los

    return los_dict


def get_los_interp(ids: IDSToplevel, ds: float = 1.0e-3) -> dict[str, list[NDArray[np.float64]]]:
    r"""Interpolate lines of sight coordinates to :math:`(R, Z, \phi)`.

    This function returns the :math:`(R, Z, \phi)` coordinates interpolated between the origin and
    terminal points of each line of sight.

    Parameters
    ----------
    ids
        The IMAS IDS containing the bolometer data.
    ds
        Desired spacing between interpolated points, by default 1 mm.

    Returns
    -------
    dict[str, list[numpy.ndarray]]
        Dictionary of interpolated lines of sight coordinates in :math:`(R, Z, \phi)` for each camera.
        Keyed by camera name, with each value being a list of arrays corresponding to each line of sight.
        The arrays have shape (3, N) where the first dimension corresponds to :math:`(R, Z, \phi)`
        and the second dimension corresponds to the interpolated points along the line of sight.
    """
    # Get the original lines of sight from the IDS
    los_dict = get_los(ids)

    # Interpolate the lines of sight to (R, Z, phi) coordinates
    los_interp = {}
    for name, los in los_dict.items():
        interp_los_per_channel = []
        for origin, terminal in los:
            distance = origin.distance_to(terminal)
            num_points = int(np.ceil(distance / ds)) + 1
            x_points = np.linspace(origin.x, terminal.x, num_points, endpoint=True)
            y_points = np.linspace(origin.y, terminal.y, num_points, endpoint=True)
            z_points = np.linspace(origin.z, terminal.z, num_points, endpoint=True)
            r_points = np.hypot(x_points, y_points)
            phi_points = np.arctan2(y_points, x_points)
            interp_los_per_channel.append(np.vstack((r_points, z_points, phi_points)))

        los_interp[name] = interp_los_per_channel

    return los_interp
