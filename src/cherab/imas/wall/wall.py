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
"""Module for loading wall components from wall IDSs."""

import numpy as np
from raysect.core.scenegraph._nodebase import _NodeBase
from raysect.optical.material.material import Material
from raysect.primitive import Mesh

from imas import DBEntry

from ..ids.common import get_ids_time_slice
from ..ids.wall import load_wall_2d, load_wall_3d

__all__ = ["load_wall_mesh", "load_wall_outline"]


def load_wall_mesh(
    *args,
    time: float = 0,
    occurrence: int = 0,
    desc_ggd_index: int = 0,
    subsets: list[str] | None = None,
    materials: dict[str, Material] | None = None,
    time_threshold: float = np.inf,
    parent: _NodeBase | None = None,
    **kwargs,
) -> dict[str, Mesh]:
    """Load machine wall components from IMAS wall IDS and Create Raysect mesh primitives.

    Parameters
    ----------
    *args
        Arguments passed to the `~imas.db_entry.DBEntry` constructor.
    time
        Time for the wall, by default 0.
    occurrence
        Occurrence index of the ``wall`` IDS, by default 0.
    desc_ggd_index
        Index of ``description_ggd``, by default 0.
    subsets
        List of names of specific ggd subsets to load, by default None (loads all subsets).
    materials
        Optional dictionary with Raysect materials for each wall component, by default None.
        Use component names as keys. The components are split by their grid subsets and for
        each grid subset by materials. All elements of the grid subset that share the same material
        are combined into a single component. The component names are assigns as follows:
        ``"{grid_name}.{subset_name}.{material_name}"``
        E.g.: ``"TokamakWall.full_main_chamber_wall.Be"``.
    time_threshold
        Maximum allowed difference between the requested time and the nearest available
        time, by default `numpy.inf`.
    parent
        Parent node in the Raysect scene graph, by default None.
        Normally, `~raysect.optical.scenegraph.world.World` instance.
    **kwargs
        Keyword arguments passed to the `~imas.db_entry.DBEntry` constructor.

    Returns
    -------
    dict[str, Mesh]
        Dictionary with the Raysect Mesh instances.

    Examples
    --------
    >>> from raysect.optical import World
    >>> world = World()
    >>> meshes = load_wall_mesh(
    ...     "imas:hdf5?path=/work/imas/shared/imasdb/ITER_MD/3/116100/1001/", "r", parent=world
    ... )
    >>> meshes
    {'FullTokamak.none.none': <raysect.primitive.mesh.mesh.Mesh at 0x1766322a0>}
    """
    with DBEntry(*args, **kwargs) as entry:
        wall_ids = get_ids_time_slice(
            entry, "wall", time=time, occurrence=occurrence, time_threshold=time_threshold
        )

    wall_dict = load_wall_3d(wall_ids.description_ggd[desc_ggd_index], subsets)

    components = {}

    for key, value in wall_dict.items():
        mesh = Mesh(value["vertices"], value["triangles"], closed=False)
        mesh.parent = parent
        mesh.name = key
        components[key] = mesh
        if materials:
            mesh.material = materials[key]

    return components


def load_wall_outline(
    *args, occurrence: int = 0, desc_index: int = 0, **kwargs
) -> dict[str, np.ndarray]:
    """Load 2D wall outline (limiter contour only) from IMAS wall IDS.

    Parameters
    ----------
    *args
        Arguments passed to the `~imas.db_entry.DBEntry` constructor.
    occurrence
        Occurrence index of the ``wall`` IDS, by default 0.
    desc_index
        Index of ``description_2d``, by default 0.
    **kwargs
        Keyword arguments passed to the `~imas.db_entry.DBEntry` constructor.

    Returns
    -------
    dict[str, (N, 2) ndarray]
       Dictionary of wall unit outlines ``(N, 2)`` array given in RZ coordinates.

    Examples
    --------
    >>> load_wall_outline("imas:hdf5?path=/work/imas/shared/imasdb/ITER_MD/3/116000/5/", "r")
    {'First Wall': array([[ 4.11129713, -2.49559808],
                          [ 4.11129713, -1.48329401],
                          ...
                          [ 6.21105623, -3.06856108]]),
     'Divertor': array([[ 3.94210005, -2.53570008],
                        ...
                        [ 6.36320019, -3.24460006]])}
    """
    with DBEntry(*args, **kwargs) as entry:
        description2d = entry.get("wall", occurrence=occurrence, autoconvert=False).description_2d[
            desc_index
        ]

    wall_outline = load_wall_2d(description2d)

    return wall_outline
