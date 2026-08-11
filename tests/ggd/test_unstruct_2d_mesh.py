import numpy as np
import pytest

from cherab.imas.ggd.unstruct_2d_mesh import UnstructGrid2D
from cherab.imas.math.polygon import calculate_2d_cell_geometry


def test_cylindrical_cell_geometry_uses_area_centroid_and_full_torus_volume():
    # An asymmetric trapezoid whose area centroid differs from its vertex mean.
    vertices = np.array([[1.0, 0.0], [3.0, 0.0], [2.0, 2.0], [1.0, 2.0]])
    grid = UnstructGrid2D(vertices, [[0, 1, 2, 3]])

    np.testing.assert_allclose(grid.cell_area, [3.0])
    np.testing.assert_allclose(grid.cell_centre, [[16.0 / 9.0, 8.0 / 9.0]])
    np.testing.assert_allclose(grid.cell_volume, [32.0 * np.pi / 3.0])

    assert not grid.cell_area.flags["WRITEABLE"]
    assert not grid.cell_centre.flags["WRITEABLE"]
    assert not grid.cell_volume.flags["WRITEABLE"]


def test_cylindrical_subset_preserves_cell_volumes():
    vertices = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [3.0, 0.0],
            [3.0, 1.0],
        ]
    )
    grid = UnstructGrid2D(vertices, [[0, 1, 2, 3], [1, 4, 5, 2]])

    subset = grid.subset([1])

    np.testing.assert_allclose(subset.cell_volume, grid.cell_volume[[1]])
    assert not subset.cell_volume.flags["WRITEABLE"]


def test_cartesian_grid_does_not_calculate_cell_volumes():
    vertices = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])

    grid = UnstructGrid2D(vertices, [[0, 1, 2]], coordinate_system="cartesian")

    np.testing.assert_allclose(grid.cell_area, [1.0])
    np.testing.assert_allclose(grid.cell_centre, [[2.0 / 3.0, 1.0 / 3.0]])
    assert not hasattr(grid, "_cell_volume")


def test_zero_area_cell_is_rejected():
    vertices = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    with pytest.raises(ValueError, match="positive area"):
        UnstructGrid2D(vertices, [[0, 1, 2]])


def test_large_geometry_calculation_openmp_path():
    vertices = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    cell_to_triangle = np.empty((100_000, 2), dtype=np.int32)
    cell_to_triangle[:, 0] = 0
    cell_to_triangle[:, 1] = 1

    centres, areas = calculate_2d_cell_geometry(vertices, triangles, cell_to_triangle)

    np.testing.assert_allclose(centres[:, 0], 2.0 / 3.0)
    np.testing.assert_allclose(centres[:, 1], 1.0 / 3.0)
    np.testing.assert_allclose(areas, 1.0)
