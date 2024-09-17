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

from typing import Literal

from imas import DBEntry  # type: ignore
from raysect.core import Node

from cherab.tools.observers.bolometry import BolometerCamera, BolometerFoil, BolometerSlit

from ..ids.bolometer import GEOMETRY_TYPE, load_channels
from ..ids.common import get_ids_time_slice

Backends = Literal["hdf5", "mdsplus", "uda", "memory"]


def load_bolometers(
    *args, backend: Backends = "hdf5", host: str | None = None, parent: Node | None = None
) -> list[BolometerCamera]:
    """Load bolometer cameras from IMAS bolometer IDS.

    :param *args: variable length argument list.
        If one argument is provided, it is assumed to be the path to the bolometer IDS.
        If four arguments are provided, they are assumed to be the *shot*, *run*, *user*, and
        *database* name. The version is assumed to be 3.
        Otherwise, a ValueError is raised.
    :param backend: IMAS database backend to use, by default "hdf5".
    :param host: IMAS database host address, by default None.
        When the database is located at another host, the host address must be provided.
    :param parent: The parent node in the Raysect scene-graph.

    :returns: A list of `BolometerCamera` objects.
    """
    if host is None:
        uri = f"imas:{backend}?"
    else:
        uri = f"imas://{host}/{backend}?"

    if len(args) == 1:
        uri += f"path={args[0]}"

    elif len(args) == 4:
        uri += f"shot={args[0]};run={args[1]};user={args[2]};database={args[3]};version=3"

    else:
        raise ValueError(f"Invalid number of arguments {len(args)}")

    entry = DBEntry(uri=uri, mode="r")  # type: ignore
    bolometer_ids = get_ids_time_slice(entry, "bolometer")

    bolo_data = load_channels(bolometer_ids)

    bolo_cameras = []

    for name, values in bolo_data.items():
        slit_data = values["slit"]
        foil_data = values["foil"]

        # Create BolometerCamera object
        camera = BolometerCamera(parent=parent, name=name)

        # Create slit object
        if slit_data["type"] != GEOMETRY_TYPE.OUTLINE:
            slit = BolometerSlit(
                f"slit-{name}",
                slit_data["centre"],
                slit_data["basis_x"],
                slit_data["dx"],
                slit_data["basis_y"],
                slit_data["dy"],
                curvature_radius=slit_data["radius"]
                if slit_data["type"] == GEOMETRY_TYPE.CIRCULAR
                else 0.0,
                parent=camera,
            )
        else:
            raise NotImplementedError("Outline geometry not supported yet.")

        # Create foil object
        if foil_data["type"] != GEOMETRY_TYPE.OUTLINE:
            foil = BolometerFoil(
                f"foil-{name}",
                foil_data["centre"],
                foil_data["basis_x"],
                foil_data["dx"],
                foil_data["basis_y"],
                foil_data["dy"],
                slit,
                curvature_radius=foil_data["radius"]
                if foil_data["type"] == GEOMETRY_TYPE.CIRCULAR
                else 0.0,
                parent=camera,
            )
        else:
            raise NotImplementedError("Outline geometry not supported yet.")

        camera.add_foil_detector(foil)
        bolo_cameras.append(camera)

    return bolo_cameras
