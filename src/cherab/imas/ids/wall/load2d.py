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
"""Module for loading 2D wall description from IMAS wall IDS."""

import numpy as np

from imas.ids_structure import IDSStructure

__all__ = ["load_wall_2d"]


def load_wall_2d(description_2d: IDSStructure) -> dict[str, np.ndarray]:
    """Load 2D wall outline (limiter contour only) from IMAS wall IDS.

    Parameters
    ----------
    description_2d
        IDS structure with 2D description of the wall.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary of wall unit outlines given in RZ coordinates.
    """
    wall_outline = {}

    for unit in description_2d.limiter.unit:
        r = unit.outline.r
        z = unit.outline.z
        wall_outline[unit.name.value] = np.array([r, z]).T

    return wall_outline
