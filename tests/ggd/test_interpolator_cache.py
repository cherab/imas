from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from cherab.imas.ggd.base_mesh import GGDGrid, InterpolatorCacheMode
from cherab.imas.ggd.unstruct_2d_extend_mesh import UnstructGrid2DExtended
from cherab.imas.ggd.unstruct_2d_mesh import UnstructGrid2D
from cherab.imas.ggd.unstruct_3d_mesh import UnstructGrid3D


def _assert_disk_cache_file_exists(
    cache_dir: Path,
    grid: UnstructGrid2D | UnstructGrid2DExtended | UnstructGrid3D,
    namespace: str,
    interpolator_kind: str,
) -> Path:
    cache_key = grid._interpolator_cache_key(
        namespace=namespace, interpolator_kind=interpolator_kind
    )
    assert cache_key is not None
    cache_file = cache_dir / f"{grid._interpolator_cache_filename(cache_key)}.pkl"
    assert cache_file.exists()
    return cache_file


def _load_grid_by_kind(
    grid_kind: str,
    jintrac_unstruct_2d_grid: UnstructGrid2D,
    jorek_unstruct_2d_extended_grid: UnstructGrid2DExtended,
    jorek_unstruct_3d_grid: UnstructGrid3D,
) -> UnstructGrid2D | UnstructGrid2DExtended | UnstructGrid3D:
    if grid_kind == "unstruct_2d":
        return jintrac_unstruct_2d_grid.subset(
            np.arange(jintrac_unstruct_2d_grid.num_cell),
            name="jintrac-cache-instance",
        )
    if grid_kind == "unstruct_2d_extended":
        return jorek_unstruct_2d_extended_grid.subset_faces(
            np.arange(jorek_unstruct_2d_extended_grid.num_faces),
            name="jorek-cache-instance",
        )
    if grid_kind == "unstruct_3d":
        return jorek_unstruct_3d_grid.subset(
            np.arange(jorek_unstruct_3d_grid.num_cell),
            name="jorek-3d-cache-instance",
        )
    raise ValueError(f"Unsupported grid_kind: {grid_kind}.")


@pytest.mark.parametrize("interpolator_cache", ["none", "memory", "disk"])
@pytest.mark.parametrize("grid_kind", ["unstruct_2d", "unstruct_2d_extended", "unstruct_3d"])
def test_interpolator_cache_modes(
    grid_kind: str,
    interpolator_cache: InterpolatorCacheMode,
    jintrac_unstruct_2d_grid: UnstructGrid2D,
    jorek_unstruct_2d_extended_grid: UnstructGrid2DExtended,
    jorek_unstruct_3d_grid: UnstructGrid3D,
    tmp_path: Path,
):
    grid = _load_grid_by_kind(
        grid_kind,
        jintrac_unstruct_2d_grid,
        jorek_unstruct_2d_extended_grid,
        jorek_unstruct_3d_grid,
    )
    namespace = f"pytest-{grid_kind}-scalar-{interpolator_cache}-{uuid4().hex}"
    cache_dir = tmp_path / "interpolator-cache"
    data = np.linspace(0.0, 1.0, grid.num_cell)

    GGDGrid._interpolator_cache_memory.clear()

    grid.interpolator(
        data,
        interpolator_cache=interpolator_cache,
        interpolator_cache_dir=cache_dir,
        interpolator_cache_namespace=namespace,
    )

    cache_key = grid._interpolator_cache_key(namespace=namespace, interpolator_kind="scalar")
    assert cache_key is not None

    if interpolator_cache == "none":
        assert cache_key not in GGDGrid._interpolator_cache_memory
        assert not (cache_dir / f"{grid._interpolator_cache_filename(cache_key)}.pkl").exists()
        return

    assert cache_key in GGDGrid._interpolator_cache_memory
    if interpolator_cache == "disk":
        _assert_disk_cache_file_exists(cache_dir, grid, namespace, "scalar")

    grid2 = _load_grid_by_kind(
        grid_kind,
        jintrac_unstruct_2d_grid,
        jorek_unstruct_2d_extended_grid,
        jorek_unstruct_3d_grid,
    )
    assert grid2._scalar_interpolator is None

    GGDGrid._interpolator_cache_memory.clear()

    grid2.interpolator(
        np.zeros(grid2.num_cell),
        interpolator_cache=interpolator_cache,
        interpolator_cache_dir=cache_dir,
        interpolator_cache_namespace=namespace,
    )

    assert grid2._scalar_interpolator is not None
    assert (
        grid2._interpolator_cache_key(namespace=namespace, interpolator_kind="scalar") == cache_key
    )
    assert cache_key in GGDGrid._interpolator_cache_memory


@pytest.mark.parametrize("interpolator_cache", ["none", "memory", "disk"])
@pytest.mark.parametrize("grid_kind", ["unstruct_2d", "unstruct_2d_extended", "unstruct_3d"])
def test_vector_interpolator_cache_modes(
    grid_kind: str,
    interpolator_cache: InterpolatorCacheMode,
    jintrac_unstruct_2d_grid: UnstructGrid2D,
    jorek_unstruct_2d_extended_grid: UnstructGrid2DExtended,
    jorek_unstruct_3d_grid: UnstructGrid3D,
    tmp_path: Path,
):
    grid = _load_grid_by_kind(
        grid_kind,
        jintrac_unstruct_2d_grid,
        jorek_unstruct_2d_extended_grid,
        jorek_unstruct_3d_grid,
    )
    namespace = f"pytest-{grid_kind}-vector-{interpolator_cache}-{uuid4().hex}"
    cache_dir = tmp_path / "interpolator-cache"

    GGDGrid._interpolator_cache_memory.clear()

    grid.vector_interpolator(
        np.zeros((3, grid.num_cell), dtype=np.float64),
        interpolator_cache=interpolator_cache,
        interpolator_cache_dir=cache_dir,
        interpolator_cache_namespace=namespace,
    )

    cache_key = grid._interpolator_cache_key(namespace=namespace, interpolator_kind="vector")
    assert cache_key is not None

    if interpolator_cache == "none":
        assert cache_key not in GGDGrid._interpolator_cache_memory
        assert not (cache_dir / f"{grid._interpolator_cache_filename(cache_key)}.pkl").exists()
        return

    assert cache_key in GGDGrid._interpolator_cache_memory
    if interpolator_cache == "disk":
        _assert_disk_cache_file_exists(cache_dir, grid, namespace, "vector")

    GGDGrid._interpolator_cache_memory.clear()

    grid2 = _load_grid_by_kind(
        grid_kind,
        jintrac_unstruct_2d_grid,
        jorek_unstruct_2d_extended_grid,
        jorek_unstruct_3d_grid,
    )
    assert grid2._vector_interpolator is None

    grid2.vector_interpolator(
        np.ones((3, grid2.num_cell), dtype=np.float64),
        interpolator_cache=interpolator_cache,
        interpolator_cache_dir=cache_dir,
        interpolator_cache_namespace=namespace,
    )

    assert (
        grid2._interpolator_cache_key(namespace=namespace, interpolator_kind="vector") == cache_key
    )
    assert cache_key in GGDGrid._interpolator_cache_memory
    assert grid2._vector_interpolator is not None
