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

import numpy as np

from imas.imasdef import EMPTY_INT

from .load_unstruct_2d import load_unstruct_grid_2d


def load_grid(grid_ggd, with_subsets=False):
    """
    Loads grid from the grid_ggd structure.

    :param grid_ggd: The grid_ggd structure.
    :param with_subsets: Read grid subset data if True. Default is True.

    :returns:
    |   grid: A grid object that depends on the grid type (structured/unstructured, 2D/3D).
    |   subsets (optional): Dictionary with grid subsets for each subset name containing the indices
    |       of the cells from that subset. Note that 'Cells' subset is included only if cell indices
    |       are specified.
    |   subset_id (optional): Dictionary with grid subset indices.
    """

    spaces = get_standard_spaces(grid_ggd)

    if not len(spaces):
        raise RuntimeError("GGD grid contain no spaces.")
   
    if len(spaces) == 1:  # simple unstructured grids
        if len(spaces[0].objects_per_dimension) == 3: # 2D case
            return load_unstruct_grid_2d(grid_ggd, 0, with_subsets=with_subsets)
        if len(spaces[0].objects_per_dimension) == 4: # 3D case
            raise NotImplementedError("Loading unstructured 3D grids will be implemented in the future.")

        raise RuntimeError("Unsupported grid type.")

    if len(spaces) == 2:  # 2D structured grid or 2D unstructured grid extended in 3D
        if len(spaces[0].objects_per_dimension) == 3 and len(spaces[1].objects_per_dimension) < 3:
            raise NotImplementedError("Loading unstructured 2D grids extended in 3D will be implemented in the future.")
        if len(spaces[0].objects_per_dimension) < 3 and len(spaces[1].objects_per_dimension) < 3:
            raise NotImplementedError("Loading structured 2D grids will be implemented in the future.")

        raise RuntimeError("Unsupported grid type.")
    
    if len(spaces) == 3:  # 3D structured grid
        if len(spaces[0].objects_per_dimension) < 3 and len(spaces[1].objects_per_dimension) < 3 and len(spaces[2].objects_per_dimension) < 3:
            raise NotImplementedError("Loading structured 3D grids will be implemented in the future.")

        raise RuntimeError("Unsupported grid type.")

    raise RuntimeError("Unsupported grid type.")


def get_standard_spaces(grid_ggd):
    """
    Gets a list of standard non-empty spaces from the grid_ggd structure.

    :param grid_ggd: The grid_ggd structure.

    :returns: A list of standard spaces.
    """

    if not len(grid_ggd.space):
        error_massage = "Unable to read the grid. Grid space is not defined."
        if len(grid_ggd.path):
            error_massage += " The grid is defined in {}.".format(grid_ggd.path)
        raise ValueError(error_massage)

    # Get list of standard spaces:
    spaces = []
    for space in grid_ggd.space:
        # JINTRAC confuses geometry_type with identifier, uncomment this check when the bug is fixed
        # if space.geometry_type.index == 0 or space.geometry_type.index == EMPTY_INT:
        if len(space.objects_per_dimension) and len(space.objects_per_dimension[0].object) > 2:
            spaces.append(space)
    
    return spaces