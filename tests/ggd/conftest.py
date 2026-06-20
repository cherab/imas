import shutil
from pathlib import Path

import numpy as np
import pytest
from imas import DBEntry

from cherab.imas.datasets import iter_jintrac, iter_jorek
from cherab.imas.ggd.unstruct_2d_extend_mesh import UnstructGrid2DExtended
from cherab.imas.ggd.unstruct_2d_mesh import UnstructGrid2D
from cherab.imas.ggd.unstruct_3d_mesh import UnstructGrid3D
from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.common.ggd import load_grid


def _copy_dataset_to_tmp(path: Path, tmp_path_factory: pytest.TempPathFactory) -> str:
    """Copy a dataset into a temporary location, handling files and directories."""
    tmp_path = tmp_path_factory.mktemp("cherab-imas-data")
    target = tmp_path / path.name
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)
    return str(target)


@pytest.fixture(scope="session")
def path_iter_jintrac(tmp_path_factory) -> str:
    """Fixture to provide the path to a sample JINTRAC IMAS dataset."""
    path = Path(iter_jintrac())
    return _copy_dataset_to_tmp(path, tmp_path_factory)


@pytest.fixture(scope="session")
def path_iter_jorek(tmp_path_factory) -> str:
    """Fixture to provide the path to a sample JOREK IMAS dataset."""
    path = Path(iter_jorek())
    return _copy_dataset_to_tmp(path, tmp_path_factory)


@pytest.fixture(scope="module")
def jintrac_unstruct_2d_grid(path_iter_jintrac: str) -> UnstructGrid2D:
    """Fixture to provide a compact UnstructGrid2D loaded from JINTRAC."""
    with DBEntry(path_iter_jintrac, "r") as entry:
        ids = get_ids_time_slice(entry, "edge_profiles", time=0)

    grid = load_grid(ids.grid_ggd[0], with_subsets=False)
    assert isinstance(grid, UnstructGrid2D)

    num_cells = min(64, grid.num_cell)
    return grid.subset(np.arange(num_cells), name="jintrac-cache-subset")


@pytest.fixture(scope="module")
def jorek_unstruct_2d_extended_grid(path_iter_jorek: str) -> UnstructGrid2DExtended:
    """Fixture to provide a compact UnstructGrid2DExtended loaded from JOREK."""
    with DBEntry(path_iter_jorek, "r") as entry:
        ids = get_ids_time_slice(entry, "radiation", time=0)

    grid = load_grid(ids.grid_ggd[0], with_subsets=False)
    assert isinstance(grid, UnstructGrid2DExtended)

    num_faces = min(8, grid.num_faces)
    return grid.subset_faces(np.arange(num_faces), name="jorek-cache-subset")


@pytest.fixture(scope="module")
def jorek_unstruct_3d_grid(
    jorek_unstruct_2d_extended_grid: UnstructGrid2DExtended,
) -> UnstructGrid3D:
    """Fixture to provide a compact UnstructGrid3D generated via grid.subset()."""
    num_cells = min(64, jorek_unstruct_2d_extended_grid.num_cell)
    return jorek_unstruct_2d_extended_grid.subset(
        np.arange(num_cells),
        name="jorek-3d-cache-subset",
    )
