from typing import cast

import numpy as np
import pytest
from imas import DBEntry

from cherab.imas.ggd import CellData
from cherab.imas.ggd.base_mesh import as_cell_data
from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.common.ggd import load_grid


def test_cell_data_validation():
    """Cell data accepts one-dimensional float sequences and arrays only."""
    data: CellData = (1.0, 2.0)

    np.testing.assert_array_equal(as_cell_data(data, 2), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        as_cell_data(cast(CellData, [[1.0, 2.0]]), 2)
    with pytest.raises(ValueError, match="contain 2 values"):
        as_cell_data(np.array([1.0]), 2)


def test_iter_solps_cells_subset(path_iter_solps: str):
    """The SOLPS ``Cells`` subset maps directly onto the loaded grid cells."""
    with DBEntry(path_iter_solps, "r") as entry:
        ids = get_ids_time_slice(entry, "edge_profiles", time=0)

    grid, subsets, _ = load_grid(ids.grid_ggd[0], with_subsets=True)

    cells, valid_mask = subsets["Cells"]
    subset = grid.subset(cells, name="Cells", valid_data_mask=valid_mask)

    assert len(cells) == grid.num_cell
    assert any(len(cell) == 3 for cell in grid.cells)
    assert subset.num_cell == len(cells)
