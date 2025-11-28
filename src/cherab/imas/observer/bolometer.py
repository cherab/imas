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

from raysect.core.scenegraph._nodebase import _NodeBase  # pyright: ignore[reportPrivateUsage]

from cherab.tools.observers.bolometry import BolometerCamera, BolometerFoil, BolometerSlit
from imas.db_entry import DBEntry

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
    parent
        The parent node of `~cherab.tools.observers.bolometry.BolometerCamera` in the Raysect
        scene-graph, by default None.
    **kwargs
        Keyword arguments passed to `~imas.db_entry.DBEntry` constructor.

    Returns
    -------
    `list[BolometerCamera]`
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

    bolometers: list[BolometerCamera] = []

    for data in bolo_data:
        # Skip empty cameras
        if len(data.channels) == 0:
            continue

        # Check if the camera is pinhole type
        match data.type:
            case CameraType.PINHOLE:
                # ------------------
                # === Camera Box ===
                # ------------------
                camera_box = None
                camera = BolometerCamera(camera_geometry=camera_box, name=data.name, parent=parent)

                # ----------------------
                # === Slit (Pinhole) ===
                # ----------------------
                # Pick up only first aperture nad use it for all channels
                slit_data = data.channels[0].slits[0]
                slit = BolometerSlit(
                    f"slit-{data.name}",
                    slit_data.centre,
                    slit_data.basis_x,
                    slit_data.dx,
                    slit_data.basis_y,
                    slit_data.dy,
                    curvature_radius=(slit_data.radius or 0.0)
                    if slit_data.type == GeometryType.CIRCULAR
                    else 0.0,
                    parent=camera,
                )
            case CameraType.COLLIMATOR:
                # ------------------
                # === Camera Box ===
                # ------------------
                camera_box = None
                slit = None
                camera = BolometerCamera(camera_geometry=camera_box, name=data.name, parent=parent)
            case _:
                raise NotImplementedError(f"Camera type {data.type} not supported yet.")

        for i_channel, channel in enumerate(data.channels):
            # Use only the first slit which is closest to plasma
            slit_data = channel.slits[0]

            if data.type == CameraType.COLLIMATOR:
                # Create slit object
                match slit_data.type:
                    case GeometryType.CIRCULAR | GeometryType.RECTANGLE:
                        slit = BolometerSlit(
                            f"slit-{data.name}-ch{i_channel}",
                            slit_data.centre,
                            slit_data.basis_x,
                            slit_data.dx,
                            slit_data.basis_y,
                            slit_data.dy,
                            curvature_radius=(slit_data.radius or 0.0)
                            if slit_data.type == GeometryType.CIRCULAR
                            else 0.0,
                            parent=camera,
                        )
                    case _:
                        raise NotImplementedError("Outline geometry not supported yet.")

            # Create foil object
            match channel.foil.type:
                case GeometryType.CIRCULAR | GeometryType.RECTANGLE:
                    foil = BolometerFoil(
                        f"foil-{data.name}-ch{i_channel}",
                        channel.foil.centre,
                        channel.foil.basis_x,
                        channel.foil.dx,
                        channel.foil.basis_y,
                        channel.foil.dy,
                        slit,
                        curvature_radius=(channel.foil.radius or 0.0)
                        if channel.foil.type == GeometryType.CIRCULAR
                        else 0.0,
                        parent=camera,
                    )
                case _:
                    raise NotImplementedError("Outline geometry not supported yet.")

            # Add foil to the camera
            camera.add_foil_detector(foil)

        bolometers.append(camera)

    return bolometers
