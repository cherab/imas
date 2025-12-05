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
"""Module for loading bolometer cameras from IMAS bolometer IDS."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from raysect.core.math import Point3D, Vector3D, from_cylindrical

from imas.ids_structure import IDSStructure
from imas.ids_toplevel import IDSToplevel

from .utility import CameraType, GeometryType


@dataclass
class Geometry:
    """Represent the geometric specification of a bolometer sensor head or slit aperture.

    The Geometry describes both simple rectangular / circular sensor faces and
    arbitrary polygonal / polyline definitions via explicit coordinates.
    It encapsulates a local orthonormal (or user supplied) basis, extents, and derived surface
    properties.
    """

    centre: Point3D = field(default_factory=Point3D)
    """Geometric centre (reference point) of the sensor/slit in global 3D coordinates."""
    type: GeometryType = GeometryType.RECTANGLE
    """Enumerated shape type.

    Common values include `RECTANGLE`, `CIRCLE`, `OUTLINE`, etc.
    Defaults to `.GeometryType.RECTANGLE`.
    """
    basis_x: Vector3D | None = None
    """Local x-axis direction vector lying in the sensor/slit plane."""
    basis_y: Vector3D | None = None
    """Local y-axis direction vector lying in the sensor/slit plane."""
    basis_z: Vector3D | None = None
    """Local outward-facing normal vector perpendicular to the sensor/slit plane.

    This vector must be directed toward the radiation sources.
    When None, it can be derived as the cross product of `basis_x` and `basis_y`.
    """
    dx: float | None = None
    """Width along `basis_x` for rectangular geometry.

    None when not applicable (e.g., circular or polygonal types).
    """
    dy: float | None = None
    """Width along `basis_y` for rectangular geometry.

    None when not applicable (e.g., circular or polygonal types).
    """
    surface: float | None = None
    """Precomputed surface area of the aperture face.

    If None, may be derived from `dx`/`dy` (rectangles), `radius` (circles), or `coords` (polygons)
    during validation or runtime.
    """
    radius: float | None = None
    """Radius for circular geometry types.

    None if the geometry is not circular.
    """
    coords: NDArray[np.float64] | None = None
    """Coordinate array defining the outline in the `basis_x` and `basis_y` plane.

    This array is used to represent a complex geometric outline.
    The array shape is ``(2, N)`` where ``N`` is the number of points.
    """


@dataclass
class BoloChannel:
    """Represent a bolometer camera channel.

    Each channel contains one foil and associated several slits.
    """

    foil: Geometry
    """Geometry of the foil used in the bolometer channel."""
    slits: list[Geometry]
    """List of geometries representing the slits associated with the bolometer channel."""


@dataclass
class BoloCamera:
    """Represent a bolometer camera and its associated data.

    This class encapsulates all the properties and data channels associated with
    a bolometer camera used for plasma diagnostics.
    """

    name: str
    """Unique identifier or label for the bolometer camera."""
    description: str
    """Detailed description of the bolometer camera, including its purpose, location, or other relevant information."""
    type: CameraType
    """Type of Bolometer camera: Pinhole/Collimator"""
    channels: list[BoloChannel]
    """List of individual bolometer channels belonging to this camera."""


def load_cameras(ids: IDSToplevel) -> list[BoloCamera]:
    """Load bolometer cameras from the bolometer IDS.

    This function retrieves the camera information from the bolometer IDS and organizes it into a
    structured format.

    Parameters
    ----------
    ids
        The bolometer IDS.

    Returns
    -------
    `.BoloCamera`
        A list of bolometer camera data structures.

    Raises
    ------
    ValueError
        If the provided IDS is not a bolometer IDS.
    RuntimeError
        If no cameras are found in the IDS.
    """
    if not str(ids.metadata.name) == "bolometer":
        raise ValueError(f"Invalid bolometer IDS ({ids.metadata.name}).")

    cameras = getattr(ids, "camera", [])
    if not len(cameras):
        raise RuntimeError("No camera found in IDS.")

    bolo_data: list[BoloCamera] = []

    for camera in cameras:
        name = str(camera.name)
        description = str(camera.description)
        camera_type = CameraType.from_value(camera.type.index.value)

        channels: list[BoloChannel] = []
        for channel in camera.channel:
            channels.append(
                BoloChannel(
                    foil=load_geometry(channel.detector),
                    slits=[load_geometry(aperture) for aperture in channel.aperture],
                )
            )

        bolo_data.append(
            BoloCamera(
                name=name,
                description=description,
                type=camera_type,
                channels=channels,
            )
        )

    return bolo_data


def load_geometry(sensor: IDSStructure) -> Geometry:
    """Load the geometry of a sensor or aperture from the bolometer IDS.

    Parameters
    ----------
    sensor
        Detector or aperture structure object.

    Returns
    -------
    `.Geometry`
        Object with the following attributes:

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

    Raises
    ------
    ValueError
        If the geometry type is invalid.
    """
    geometry = Geometry()

    geometry.centre = from_cylindrical(
        sensor.centre.r, sensor.centre.z, np.rad2deg(sensor.centre.phi)
    )

    geometry_type = GeometryType.from_value(int(sensor.geometry_type.value))
    geometry.type = geometry_type

    match geometry_type:
        case GeometryType.RECTANGLE:
            # Unit vectors
            geometry.basis_x = Vector3D(
                sensor.x2_unit_vector.x, sensor.x2_unit_vector.y, sensor.x2_unit_vector.z
            )
            geometry.basis_y = Vector3D(
                sensor.x1_unit_vector.x, sensor.x1_unit_vector.y, sensor.x1_unit_vector.z
            )
            geometry.basis_z = Vector3D(
                sensor.x3_unit_vector.x, sensor.x3_unit_vector.y, sensor.x3_unit_vector.z
            )

            # Dimensions
            geometry.dx = float(sensor.x2_width.value)
            geometry.dy = float(sensor.x1_width.value)

        case GeometryType.CIRCULAR:
            # Radius
            radius = getattr(sensor, "radius", None)
            if radius is None or radius <= 0:
                raise ValueError(f"Invalid radius ({radius}).")

            geometry.radius = float(radius.value)

            # Unit vectors
            geometry.basis_z = Vector3D(
                sensor.x3_unit_vector.x, sensor.x3_unit_vector.y, sensor.x3_unit_vector.z
            )
            geometry.basis_x = geometry.basis_z.orthogonal().normalise()
            geometry.basis_y = geometry.basis_z.cross(geometry.basis_x).normalise()

        case GeometryType.OUTLINE:
            # Outline coordinates for basis_x and basis_y
            geometry.coords = np.vstack((sensor.outline.x2, sensor.outline.x1))

            # Unit vectors
            geometry.basis_x = Vector3D(
                sensor.x2_unit_vector.x, sensor.x2_unit_vector.y, sensor.x2_unit_vector.z
            )
            geometry.basis_y = Vector3D(
                sensor.x1_unit_vector.x, sensor.x1_unit_vector.y, sensor.x1_unit_vector.z
            )
            geometry.basis_z = Vector3D(
                sensor.x3_unit_vector.x, sensor.x3_unit_vector.y, sensor.x3_unit_vector.z
            )

    # Surface area
    surface = getattr(sensor, "surface", None)
    geometry.surface = float(surface.value) if surface is not None else None

    return geometry
