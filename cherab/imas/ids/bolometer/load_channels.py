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

from __future__ import annotations

from typing import Any

import numpy as np
from imas import bolometer  # type: ignore
from raysect.core.math import Point3D, Vector3D

from .utility import GEOMETRY_TYPE


def load_channels(bolometer_ids: bolometer) -> dict[str, dict[str, Any]]:
    """Load bolometer channels from the bolometer IDS.

    :param bolometer_ids: The bolometer IDS.

    :returns: A dictionary with the following keys:
        'channel_name': A dictionary with the following keys:
            'foil': A dictionary with the following keys:
                'centre': A Point3D with the coordinates of the foil centre,
                'type': The geometry type of the foil,
                'basis_x': A Vector3D with the x-basis vector of the foil,
                'basis_y': A Vector3D with the y-basis vector of the foil,
                'basis_z': A Vector3D with the z-basis vector of the foil,
                'dx': The width of the foil in the x-direction [m],
                'dy': The width of the foil in the y-direction [m],
                'surface': The surface area of the foil [m^2],
                'radius': The radius of the foil [m] if the geometry type is circular,
                'coords': The outline coordinates for basis_x and basis_y if the geometry type is outline,
            'slit': A dictionary with the following keys:
                'centre': A Point3D with the coordinates of the slit centre,
                'type': The geometry type of the slit,
                'basis_x': A Vector3D with the x-basis vector of the slit,
                'basis_y': A Vector3D with the y-basis vector of the slit,
                'basis_z': A Vector3D with the z-basis vector of the slit,
                'dx': The width of the slit in the x-direction [m],
                'dy': The width of the slit in the y-direction [m],
                'surface': The surface area of the slit [m^2],
                'radius': The radius of the slit [m] if the geometry type is circular,
                'coords': The outline coordinates for basis_x and basis_y if the geometry type is outline.
    """
    if not isinstance(bolometer_ids, bolometer):
        raise ValueError("Invalid bolometer IDS")

    channels = getattr(bolometer_ids, "channel", [])
    if not len(channels):
        raise RuntimeError("No channels found in bolometer")

    bolo_data = {}

    for channel in channels:
        name = channel.name
        bolo_data[name] = {
            "foil": load_geometry(channel.detector),
            "slit": load_geometry(channel.aperture[0]),
        }

    return bolo_data


def load_geometry(sensor) -> dict[str, Any]:
    """Load the geometry of a sensor.

    :param sensor: detector or aperture structure object.

    :returns: A dictionary with the following keys:
        'centre': A Point3D with the coordinates of the sensor centre,
        'type': The geometry type of the sensor,
        'basis_x': A Vector3D with the x-basis vector of the sensor,
        'basis_y': A Vector3D with the y-basis vector of the sensor,
        'basis_z': A Vector3D with the z-basis vector of the sensor,
        'dx': The width of the sensor in the x-direction [m],
        'dy': The width of the sensor in the y-direction [m],
        'surface': The surface area of the sensor [m^2],
        'radius': The radius of the sensor [m] if the geometry type is circular,
        'coords': The outline coordinates for basis_x and basis_y if the geometry type is outline.
    """
    geometry = {}

    centre = sensor.centre
    geometry["centre"] = Point3D(*_cylin_to_cart(centre.r, centre.phi, centre.z))

    geometry_type = GEOMETRY_TYPE.from_value(sensor.geometry_type)

    if geometry_type == GEOMETRY_TYPE.RECTANGLE:
        # Geometry type: rectangle
        geometry["type"] = GEOMETRY_TYPE.RECTANGLE

        # Unit vectors
        geometry["basis_x"] = Vector3D(
            sensor.x2_unit_vector.x, sensor.x2_unit_vector.y, sensor.x2_unit_vector.z
        )
        geometry["basis_y"] = Vector3D(
            sensor.x1_unit_vector.x, sensor.x1_unit_vector.y, sensor.x1_unit_vector.z
        )
        geometry["basis_z"] = Vector3D(
            sensor.x3_unit_vector.x, sensor.x3_unit_vector.y, sensor.x3_unit_vector.z
        )

        # Check if bases are in RHS (assume that basis_z is a vector targetting the plasma)
        if geometry["basis_z"].dot(geometry["basis_x"].cross(geometry["basis_y"])) < 0:
            geometry["basis_y"] *= -1

        # Dimensions
        geometry["dx"] = sensor.x2_width
        geometry["dy"] = sensor.x1_width

        # Area
        geometry["surface"] = getattr(sensor, "surface", None)

    if geometry_type == GEOMETRY_TYPE.CIRCULAR:
        # Geometry type: circular
        geometry["type"] = GEOMETRY_TYPE.CIRCULAR

        # Radius
        radius = getattr(sensor, "radius", None)
        if radius is None or radius <= 0:
            raise ValueError("Invalid radius")

    if geometry_type == GEOMETRY_TYPE.OUTLINE:
        # Geometry type: outline
        geometry["type"] = GEOMETRY_TYPE.OUTLINE

        # Outline coordinates for basis_x and basis_y
        geometry["coords"] = np.vstack((sensor.outline.x2, sensor.outline.x1))

    return geometry


def _cylin_to_cart(r, phi, z) -> tuple[float, float, float]:
    """Convert cylindrical coordinates to cartesian coordinates.

    :param r: radial coordinate
    :param phi: azimuthal coordinate
    :param z: vertical coordinate

    :returns: A tuple with the x, y, and z coordinates.
    """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z
