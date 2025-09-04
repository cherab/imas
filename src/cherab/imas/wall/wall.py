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

from numpy import inf

import imas

from raysect.primitive import Mesh

from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.wall import load_wall_3d, load_wall_2d


def load_wall_mesh(shot, run, user, database, backend=imas.imasdef.MDSPLUS_BACKEND, time=0, occurrence=0,
                   desc_ggd_index=0, subsets=None, materials=None, time_threshold=inf, parent=None):
    """
    Loads machine wall components from IMAS wall IDS and creates a dictionary
    of Raysect mesh primitives.

    :param shot: IMAS shot number.
    :param run: IMAS run number for a given shot.
    :param user: IMAS username.
    :param database: IMAS database name.
    :param backend: IMAS database backend. Default is imas.imasdef.MDSPLUS_BACKEND.
    :param time: Time moment. Default is 0.
    :param occurrence: Instance index of the 'wall' IDS. Default is 0.
    :param desc_ggd_index: Index of description_ggd. Default is 0.
    :param subsets: A list of names of specific ggd subsets to load.
                    Default is None (loads all subsets).
    :param materials: Optional dictionary with Raysect materials for each wall component.
        Default is None. Use component names as keys.
        The components are splitted by their grid subsets and for each grid subset by materials.
        All elements of the grid subset that share the same material are combined into
        a single component. The component names are assignes as follows:
        "{grid_name}.{subset_name}.{material_name}"
        E.g.: "TokamakWall.full_main_chamber_wall.Be".
    :param time_threshold: Sets the maximum allowable difference between the specified time and the nearest
        available time. Default is np.inf.
    :param parent: The parent node in the Raysect scene-graph.

    :returns: A dictinary with the Raysect Mesh instants.
    """

    entry = imas.DBEntry(backend, database, shot, run, user)
    wall_ids = get_ids_time_slice(entry, 'wall', time=time, occurrence=occurrence, time_threshold=time_threshold)

    wall_dict = load_wall_3d(wall_ids.description_ggd[desc_ggd_index], subsets)

    entry.close()

    components = {}

    for key, value in wall_dict.items():
        mesh = Mesh(value['vertices'], value['triangles'], closed=False)
        mesh.parent = parent
        mesh.name = key
        components[key] = mesh
        if materials:
            mesh.material = materials[key]

    return components


def load_wall_outline(shot, run, user, database, backend=imas.imasdef.MDSPLUS_BACKEND,
                      occurrence=0, desc_index=0):
    """
    Loads 2D wall outline (limiter contour only) from IMAS wall IDS and returns a dictionary.

    :param shot: IMAS shot number.
    :param run: IMAS run number for a given shot.
    :param user: IMAS username.
    :param database: IMAS database name.
    :param backend: IMAS database backend. Default is imas.imasdef.MDSPLUS_BACKEND.
    :param occurrence: Instance index of the 'wall' IDS. Default is 0.
    :param desc_index: Index of description_2d. Default is 0.

    :returns: A dictionary of wall unit outlines given in RZ coordinates.
    """

    entry = imas.DBEntry(backend, database, shot, run, user)
    entry.open()

    description2d = entry.partial_get('wall', 'description_2d({})'.format(desc_index), occurrence=occurrence)

    entry.close()

    wall_outline = load_wall_2d(description2d)

    return wall_outline
