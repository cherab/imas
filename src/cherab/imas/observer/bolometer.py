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

from typing import Any

from raysect.core.scenegraph._nodebase import _NodeBase

from cherab.tools.observers.bolometry import BolometerCamera, BolometerFoil, BolometerSlit
from imas import DBEntry

from ..ids.bolometer import load_cameras
from ..ids.bolometer.utility import CameraType, GeometryType
from ..ids.common import get_ids_time_slice

__all__ = ["load_bolometers"]


def load_bolometers(*args, parent: _NodeBase | None = None, **kwargs) -> list[BolometerCamera]:
    """Load bolometer cameras from IMAS bolometer IDS.

    .. note::
        This function requires the Data dictionary v4.1.0 or later.

    Parameters
    ----------
    *args
        Arguments passed to `~imas.db_entry.DBEntry`.
    parent : _NodeBase | None
        The parent node of `~cherab.tools.observers.bolometry.BolometerCamera` in the Raysect
        scene-graph, by default None.
    **kwargs
        Keyword arguments passed to `~imas.db_entry.DBEntry` constructor.

    Returns
    -------
    list[BolometerCamera]
        List of `~cherab.tools.observers.bolometry.BolometerCamera` objects.

    Examples
    --------
    >>> from raysect.optical import World
    >>> world = World()

    If you have a local IMAS database and store the "bolometer.h5" file there:
    >>> bolometers = load_bolometers("imas:hdf5?path=path/to/db/", "r", parent=world)

    If you want to load netCDF files directly:
    >>> bolometers = load_bolometers("path/to/bolometer_file.nc", "r", parent=world)
    """
    # Load bolometer IDS
    with DBEntry(*args, **kwargs) as entry:
        # Get available time slices
        ids = get_ids_time_slice(entry, "bolometer")

    # Extract bolometer data
    bolo_data = load_cameras(ids)

    bolometers = []

    for camera_name, values in bolo_data.items():
        # Skip empty cameras
        if len(values["channels"]) == 0:
            continue

        # Instantiate BolometerCamera object
        camera = BolometerCamera(name=camera_name, parent=parent)

        # Check if the camera is pinhole type
        match values["type"]:
            case CameraType.PINHOLE:
                # Pick up only first aperture nad use it for all channels
                slit_data = values["channels"][0]["slit"][0]
                slit = BolometerSlit(
                    f"slit-{camera_name}",
                    slit_data["centre"],
                    slit_data["basis_x"],
                    slit_data["dx"],
                    slit_data["basis_y"],
                    slit_data["dy"],
                    curvature_radius=slit_data["radius"]
                    if slit_data["type"] == GeometryType.CIRCULAR
                    else 0.0,
                    parent=None,
                )
            case CameraType.COLLIMATOR:
                slit = None
            case _:
                raise NotImplementedError(f"Camera type {values['type']} not supported yet.")

        for i_channel, channel in enumerate(values["channels"]):
            foil_data: dict[str, Any] = channel["foil"]
            slits_data: list[dict[str, Any]] = channel["slit"]

            # Use only the first slit which is closest to plasma
            slit_data = slits_data[0]

            if values["type"] == CameraType.COLLIMATOR:
                # Create slit object
                match slit_data["type"]:
                    case GeometryType.CIRCULAR | GeometryType.RECTANGLE:
                        slit = BolometerSlit(
                            f"slit-{camera_name}-ch{i_channel}",
                            slit_data["centre"],
                            slit_data["basis_x"],
                            slit_data["dx"],
                            slit_data["basis_y"],
                            slit_data["dy"],
                            curvature_radius=slit_data["radius"]
                            if slit_data["type"] == GeometryType.CIRCULAR
                            else 0.0,
                            parent=None,
                        )
                    case _:
                        raise NotImplementedError("Outline geometry not supported yet.")

            # Create foil object
            match foil_data["type"]:
                case GeometryType.CIRCULAR | GeometryType.RECTANGLE:
                    foil = BolometerFoil(
                        f"foil-{camera_name}-ch{i_channel}",
                        foil_data["centre"],
                        foil_data["basis_x"],
                        foil_data["dx"],
                        foil_data["basis_y"],
                        foil_data["dy"],
                        slit,
                        curvature_radius=foil_data["radius"]
                        if foil_data["type"] == GeometryType.CIRCULAR
                        else 0.0,
                        parent=None,
                    )
                case _:
                    raise NotImplementedError("Outline geometry not supported yet.")

            # Link parent
            slit.parent = camera
            foil.parent = camera

            # Add foil to the camera
            camera.add_foil_detector(foil)

        bolometers.append(camera)

    return bolometers
