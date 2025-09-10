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
from raysect.core.math import Point3D, Vector3D

from imas.ids_structure import IDSStructure
from imas.ids_toplevel import IDSToplevel

from .utility import GeometryType

__all__ = ["load_cameras"]


def load_cameras(ids: IDSToplevel) -> dict:
    """Load bolometer cameras from the bolometer IDS.

    This function retrieves the camera information from the bolometer IDS and organizes it into a
    structured format.
    The specific structure of the output dictionary is as follows:

    .. code-block:: python

        {
        'camera_name': {
            "description": "Camera description",
            "channels": [
                {
                    'foil': {
                        'centre': Point3D(...),  # Centre coordinates of the foil
                        'type': GeometryType(...),  # Geometry type of the foil
                        'basis_x': Vector3D(...),  # x-basis vector of the foil
                        'basis_y': Vector3D(...),  # y-basis vector of the foil
                        'basis_z': Vector3D(...),  # z-basis vector of the foil
                        'dx': ...,  # Width of the foil in the x-direction [m]
                        'dy': ...,  # Width of the foil in the y-direction [m]
                        'surface': ...,  # Surface area of the foil [m²]
                        'radius': ...,  # Radius of the foil [m] if GeometryType.CIRCULAR
                        'coords': np.array([...]),  # Outline coordinates if GeometryType.OUTLINE
                    },
                    'slit': [  # List of slits
                        {
                            'centre': Point3D(...),  # Centre coordinates of the slit
                            'type': GeometryType(...),  # Geometry type of the slit
                            'basis_x': Vector3D(...),  # x-basis vector of the slit
                            'basis_y': Vector3D(...),  # y-basis vector of the slit
                            'basis_z': Vector3D(...),  # z-basis vector of the slit
                            'dx': ...,  # Width of the slit in the x-direction [m]
                            'dy': ...,  # Width of the slit in the y-direction [m]
                            'surface': ...,  # Surface area of the slit [m²]
                            'radius': ...,  # Radius of the slit [m] if GeometryType.CIRCULAR
                            'coords': np.array([...]),  # Outline coordinates if GeometryType.OUTLINE
                        },
                        ...
                    ]
                },
                ...
            ],
        ],
        ...
        }

    Parameters
    ----------
    ids : IDSToplevel
        The bolometer IDS.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary with camera names as keys, and foil and slit data as values.
        Some keys in the foil and slit lists may not be present depending on the geometry type.
    """
    if not isinstance(ids, IDSToplevel):
        raise ValueError("Invalid IDS object.")

    if not ids.metadata.name == "bolometer":
        raise ValueError(f"Invalid bolometer IDS ({ids.metadata.name}).")

    cameras = getattr(ids, "camera", [])
    if not len(cameras):
        raise RuntimeError("No camera found in IDS.")

    bolo_data = {}

    for camera in cameras:
        name = camera.name.value
        description = camera.description.value

        bolo_data[name] = {
            "description": description,
            "channels": [],
        }

        # TODO: different structure for pinhole to reduce overhead?
        for channel in camera.channel:
            bolo_data[name]["channels"].append(
                {
                    "foil": load_geometry(channel.detector),
                    "slit": [load_geometry(aperture) for aperture in channel.aperture],
                }
            )

    return bolo_data


def load_geometry(sensor: IDSStructure) -> dict[str, Any]:
    """Load the geometry of a sensor or aperture from the bolometer IDS.

    Parameters
    ----------
    sensor : imas.ids_structure.IDSStructure
        Detector or aperture structure object.

    Returns
    -------
    dict[str, Any]
        Dictionary with the following keys:

        - ``'centre'``: Point3D with the coordinates of the sensor centre,
        - ``'type'``: Geometry type of the sensor,
        - ``'basis_x'``: Vector3D with the x-basis vector of the sensor,
        - ``'basis_y'``: Vector3D with the y-basis vector of the sensor,
        - ``'basis_z'``: Vector3D with the z-basis vector of the sensor,
        - ``'dx'``: Width of the sensor in the x-direction [m],
        - ``'dy'``: Width of the sensor in the y-direction [m],
        - ``'surface'``: Surface area of the sensor [m²],
        - ``'radius'``: Radius of the sensor [m] if the geometry type is circular,
        - ``'coords'``: Outline coordinates for basis_x and basis_y if the geometry type is outline.

        Some keys may not be present depending on the geometry type.
    """
    geometry = {}

    centre = sensor.centre
    geometry["centre"] = Point3D(*_cylin_to_cart(centre.r, centre.phi, centre.z))

    geometry_type = GeometryType.from_value(sensor.geometry_type.item())
    geometry["type"] = geometry_type

    match geometry_type:
        case GeometryType.RECTANGLE:
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

            # Dimensions
            geometry["dx"] = sensor.x2_width.item()
            geometry["dy"] = sensor.x1_width.item()

        case GeometryType.CIRCULAR:
            # Radius
            radius = getattr(sensor, "radius", None)
            if radius is None or radius <= 0:
                raise ValueError(f"Invalid radius ({radius}).")

            geometry["radius"] = radius.item()

            geometry["basis_z"] = Vector3D(
                sensor.x3_unit_vector.x, sensor.x3_unit_vector.y, sensor.x3_unit_vector.z
            )

        case GeometryType.OUTLINE:
            # Outline coordinates for basis_x and basis_y
            geometry["coords"] = np.vstack((sensor.outline.x2, sensor.outline.x1))

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

        case _:
            raise ValueError(f"Invalid geometry type ({geometry_type}).")

    # Surface area
    surface = getattr(sensor, "surface", None)
    geometry["surface"] = surface.item() if surface is not None else None

    return geometry


def _cylin_to_cart(r, phi, z) -> tuple[float, float, float]:
    """Convert cylindrical coordinates to cartesian coordinates.

    Parameters
    ----------
    r : float
        Radial coordinate
    phi : float
        Azimuthal coordinate
    z : float
        Vertical coordinate

    Returns
    -------
    tuple[float, float, float]
        A tuple with the x, y, and z coordinates.
    """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z
