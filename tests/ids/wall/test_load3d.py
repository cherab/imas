"""Regression tests for wall subset identifiers represented by IDS scalar wrappers."""

import numpy as np
import pytest
from imas import IDSFactory

from cherab.imas.ids.wall import load_wall_3d


@pytest.mark.parametrize("with_material", [False, True])
def test_wall_subset_identifier_is_hashable(with_material: bool) -> None:
    wall = IDSFactory().wall()
    wall.description_ggd.resize(1)
    description = wall.description_ggd[0]
    description.grid_ggd.resize(1)
    grid = description.grid_ggd[0]
    grid.identifier.name = "test"
    grid.space.resize(1)
    space = grid.space[0]
    space.objects_per_dimension.resize(3)
    space.objects_per_dimension[0].object.resize(3)
    for vertex, xyz in zip(
        space.objects_per_dimension[0].object, ((0, 0, 0), (1, 0, 0), (0, 1, 0)), strict=True
    ):
        vertex.geometry = xyz
    space.objects_per_dimension[2].object.resize(1)
    space.objects_per_dimension[2].object[0].nodes = [1, 2, 3]
    grid.grid_subset.resize(1)
    subset = grid.grid_subset[0]
    subset.identifier.name = "part"
    subset.identifier.index = -1
    subset.dimension = 3
    subset.element.resize(1)
    subset.element[0].object.resize(1)
    subset.element[0].object[0].index = 1
    if with_material:
        description.material.resize(1)
        description.material[0].grid_subset.resize(1)
        material = description.material[0].grid_subset[0]
        material.grid_subset_index = -1
        material.identifiers.names = ["W"]
    components = load_wall_3d(description, subsets=["part"])
    component = components[f"test.part.{'W' if with_material else 'none'}"]
    np.testing.assert_array_equal(component["triangles"], [[0, 1, 2]])
    np.testing.assert_array_equal(component["vertices"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
