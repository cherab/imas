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
"""Module defining the base class for general grids (GGD)."""

from __future__ import annotations

import hashlib
import pickle
from abc import abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, SupportsIndex, TypeAlias, TypeVar, cast

import matplotlib.axes
import numpy as np
import pooch
from numpy.typing import NDArray
from raysect.core.math.function.float import Function2D, Function3D
from raysect.core.math.function.vector3d.function2d import Function2D as VectorFunction2D
from raysect.core.math.function.vector3d.function3d import Function3D as VectorFunction3D
from raysect.core.math.vector import Vector3D

__all__ = ["GGDGrid", "CellSelection", "InterpolatorCacheMode", "as_index_array"]

ZEROVECTOR = Vector3D(0, 0, 0)
CellSelection: TypeAlias = Sequence[SupportsIndex] | NDArray[np.integer[Any]]
InterpolatorCacheMode: TypeAlias = Literal["none", "memory", "disk"]
"""Cache mode for interpolator templates.
"""
InterpolatorT = TypeVar("InterpolatorT")


def as_index_array(indices: CellSelection) -> NDArray[np.intp]:
    """Return cell-selection indices as a NumPy integer array.

    Parameters
    ----------
    indices
        Cell-selection indices as a sequence of integers or a NumPy array of integers.

    Returns
    -------
    NDArray[np.intp]
        Cell-selection indices as a NumPy integer array.
    """
    return np.asarray(indices, dtype=np.intp)


def _default_interpolator_cache_root() -> Path:
    return pooch.os_cache("cherab/imas") / "interpolators"


class GGDGrid:
    """Base class for general grids (GGD).

    Parameters
    ----------
    name
        Name of the grid.
    dimension
        Grid dimensions, by default 1.
    coordinate_system
        Coordinate system, by default ``"cartesian"``.
    """

    def __init__(
        self,
        name: str = "",
        dimension: int = 1,
        coordinate_system: Literal["cylindrical", "cartesian"] = "cartesian",
    ) -> None:
        if dimension < 1:
            raise ValueError("Attribute dimension must be >= 1.")

        self._dimension: int = dimension
        self._name: str = name
        self._coordinate_system: str = coordinate_system

        self._scalar_interpolator: object | None = None
        self._vector_interpolator: object | None = None
        self._cell_centre: NDArray[np.float64]
        self._cell_area: NDArray[np.float64]
        self._cell_volume: NDArray[np.float64]
        self._mesh_extent: dict[str, float]
        self._num_cell: int

        self._initial_setup()

    _interpolator_cache_memory: dict[str, object] = {}

    @abstractmethod
    def _initial_setup(self) -> None:
        raise NotImplementedError("To be defined in subclass.")

    @property
    def name(self) -> str:
        """Grid name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def dimension(self) -> int:
        """Grid dimension."""
        return self._dimension

    @property
    def num_cell(self) -> int:
        """Number of grid cells."""
        return self._num_cell

    @property
    def coordinate_system(self) -> str:
        """Coordinate system."""
        return self._coordinate_system

    @property
    def cell_centre(self) -> NDArray[np.float64]:
        """Coordinate of cell centres as ``(num_cell, dimension)`` array."""
        return self._cell_centre

    @property
    def cell_area(self) -> NDArray[np.float64]:
        """Cell areas as ``(num_cell,)`` array."""
        return self._cell_area

    @property
    def cell_volume(self) -> NDArray[np.float64]:
        """Cell volume as ``(num_cell,)`` array."""
        return self._cell_volume

    @property
    def mesh_extent(self) -> dict[str, float]:
        """Extent of the mesh.

        A dictionary with xmin, xmax, ymin and ymax, ... keys.
        """
        return self._mesh_extent

    def _interpolator_geometry_hash(self) -> str | None:
        """Return a stable geometry hash based on the grid `vertices` and `cells`.

        Returns
        -------
        str | None
            Hash digest for cache key generation. Returns None if the grid does
            not expose both `vertices` and `cells` attributes.
        """
        vertices = getattr(self, "vertices", None)
        cells = getattr(self, "cells", None)
        if vertices is None or cells is None:
            return None

        vertices_array = np.ascontiguousarray(vertices)
        cells_array = np.ascontiguousarray(cells)

        digest = hashlib.blake2b(digest_size=20)
        digest.update(str(vertices_array.dtype).encode("ascii"))
        digest.update(np.asarray(vertices_array.shape, dtype=np.int64).tobytes())
        digest.update(vertices_array.tobytes())
        digest.update(str(cells_array.dtype).encode("ascii"))
        digest.update(np.asarray(cells_array.shape, dtype=np.int64).tobytes())
        digest.update(cells_array.tobytes())
        return digest.hexdigest()

    def _interpolator_cache_key(self, namespace: str = "ggd") -> str | None:
        """Return a cache key derived from the grid geometry hash.

        Parameters
        ----------
        namespace
            Prefix used to prevent cache collisions across different interpolator types.

        Returns
        -------
        str | None
            Cache key string if the grid geometry can be hashed, otherwise None.
        """
        geometry_hash = self._interpolator_geometry_hash()
        if geometry_hash is None:
            return None
        return f"{namespace}:{geometry_hash}"

    def _load_cached_interpolator(
        self,
        *,
        mode: InterpolatorCacheMode = "memory",
        cache_dir: str | Path | None = None,
        namespace: str = "ggd",
    ) -> object | None:
        """Load an interpolator cache entry for this grid geometry.

        Parameters
        ----------
        mode
            Cache mode. See `InterpolatorCacheMode`.
        cache_dir
            Directory used for disk cache mode.
        namespace
            Namespace prefix to avoid cache-key collisions.

        Returns
        -------
        object | None
            Cached interpolator object if found, otherwise ``None``.
        """
        cache_key = self._interpolator_cache_key(namespace=namespace)
        if cache_key is None or mode == "none":
            return None

        if mode == "memory":
            return self._interpolator_cache_memory.get(cache_key)

        # For disk mode, use in-process memory as a fast-path to avoid repeated unpickling.
        memory_cached = self._interpolator_cache_memory.get(cache_key)
        if memory_cached is not None:
            return memory_cached

        root = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else _default_interpolator_cache_root()
        )
        path = root / f"{cache_key}.pkl"
        if not path.exists():
            return None

        with path.open("rb") as handle:
            loaded = pickle.load(handle)

        # Keep loaded template in memory for the current process lifetime.
        self._interpolator_cache_memory[cache_key] = loaded
        return loaded

    def _store_cached_interpolator(
        self,
        interpolator: object,
        *,
        mode: InterpolatorCacheMode = "memory",
        cache_dir: str | Path | None = None,
        namespace: str = "ggd",
    ) -> None:
        """Store an interpolator cache entry for this grid geometry.

        Parameters
        ----------
        interpolator
            Interpolator object to cache.
        mode
            Cache mode. See `InterpolatorCacheMode`.
        cache_dir
            Directory used for disk cache mode.
        namespace
            Namespace prefix to avoid cache-key collisions.
        """
        geometry_hash = self._interpolator_geometry_hash()
        if geometry_hash is None or mode == "none":
            return

        cache_key = f"{namespace}:{geometry_hash}"
        if mode == "memory":
            self._interpolator_cache_memory[cache_key] = interpolator
            return

        # Keep disk-cached templates also in memory for faster reuse in-process.
        self._interpolator_cache_memory[cache_key] = interpolator

        root = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else _default_interpolator_cache_root()
        )
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{cache_key}.pkl"
        with path.open("wb") as handle:
            pickle.dump(interpolator, handle)

    def _build_cached_interpolator(
        self,
        *,
        interpolator_cls: type[InterpolatorT],
        template_builder: Callable[[], InterpolatorT],
        data: NDArray[np.float64],
        fill: float | Vector3D,
        template_data: NDArray[np.float64],
        template_fill: float | Vector3D,
        cached_slot: str,
        mode: InterpolatorCacheMode = "memory",
        cache_dir: str | Path | None = None,
        namespace: str = "ggd",
    ) -> InterpolatorT:
        """Build and return a per-call interpolator backed by a cached geometry template.

        The cached object stores only mesh acceleration data (KDTree). `data` and
        `fill` are injected per call via ``interpolator_cls.instance(template, data, fill)``.

        Parameters
        ----------
        interpolator_cls
            Interpolator class; used for ``isinstance`` checks and ``.instance()`` calls.
        template_builder
            Callable that creates a new geometry-only template on cache miss.
        data
            Per-call data array (``grid_data`` or ``grid_vectors``).
        fill
            Per-call fill value (``fill_value: float`` or ``fill_vector: Vector3D``).
        template_data
            Zero-valued data array used to normalise a loaded cache entry.
        template_fill
            Zero fill used to normalise a loaded cache entry.
        cached_slot
            Name of the instance attribute used to store the cached template
            (``"_scalar_interpolator"`` or ``"_vector_interpolator"``).
        mode
            Cache mode. See `InterpolatorCacheMode`.
        cache_dir
            Directory used for disk cache mode.
        namespace
            Namespace prefix to avoid cache-key collisions.

        Returns
        -------
        InterpolatorT
            A per-call interpolator instance created from the cached template.

        Raises
        ------
        TypeError
            If the cached template is not compatible with `interpolator_cls`.
        """
        itype = cast(Any, interpolator_cls)
        cached: InterpolatorT | None = getattr(self, cached_slot, None)

        if mode != "none" and cached is None:
            loaded = self._load_cached_interpolator(
                mode=mode,
                cache_dir=cache_dir,
                namespace=namespace,
            )
            if isinstance(loaded, interpolator_cls):
                cached = itype.instance(loaded, template_data, template_fill)
                setattr(self, cached_slot, cached)

        if cached is None:
            cached = template_builder()
            setattr(self, cached_slot, cached)
            self._store_cached_interpolator(
                cached,
                mode=mode,
                cache_dir=cache_dir,
                namespace=namespace,
            )

        if not isinstance(cached, interpolator_cls):
            raise TypeError(
                f"The existing interpolator is not an instance of {interpolator_cls.__name__}. "
                "Cannot create a new interpolator instance sharing the same KDtree structure."
            )

        return itype.instance(cached, data, fill)

    @abstractmethod
    def subset(self, indices: CellSelection, name: str | None = None) -> GGDGrid:
        """Create a subset grid from this instance.

        Parameters
        ----------
        indices
            Indices of the cells of the original grid in the subset.
        name
            Name of the grid subset. Default is ``instance.name + " subset"``.

        Returns
        -------
        GGDGrid
            Subset grid instance.
        """
        raise NotImplementedError("To be defined in subclass.")

    @abstractmethod
    def interpolator(
        self,
        grid_data: NDArray[np.float64],
        fill_value: float = 0.0,
        *,
        interpolator_cache: InterpolatorCacheMode = "memory",
        interpolator_cache_dir: str | Path | None = None,
        interpolator_cache_namespace: str = "ggd",
    ) -> Function2D | Function3D:
        """Return an Function interpolator instance for the data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance of the
        previously created interpolator.

        Parameters
        ----------
        grid_data
            Array containing data in the grid cells.
        fill_value
            A value returned outside the grid, by default is 0.0.
        interpolator_cache
            Cache mode for the interpolator. See `InterpolatorCacheMode`.
        interpolator_cache_dir
            Directory used for disk cache mode.
        interpolator_cache_namespace
            Namespace prefix to avoid cache-key collisions.

        Returns
        -------
        `Function2D` | `Function3D`
            Interpolator instance.
        """
        raise NotImplementedError("To be defined in subclass.")

    @abstractmethod
    def vector_interpolator(
        self,
        grid_vectors: NDArray[np.float64],
        fill_vector: Vector3D = ZEROVECTOR,
        *,
        interpolator_cache: InterpolatorCacheMode = "memory",
        interpolator_cache_dir: str | Path | None = None,
        interpolator_cache_namespace: str = "ggd",
    ) -> VectorFunction2D | VectorFunction3D:
        """Return a VectorFunction interpolator instance for the vector data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance of the
        previously created interpolator.

        Parameters
        ----------
        grid_vectors
            ``(3, num_cell)`` Array containing 3D vectors in the grid cells.
        fill_vector
            3D vector returned outside the grid, by default ``Vector3D(0, 0, 0)``.
        interpolator_cache
            Cache mode for the interpolator. See `InterpolatorCacheMode`.
        interpolator_cache_dir
            Directory used for disk cache mode.
        interpolator_cache_namespace
            Namespace prefix to avoid cache-key collisions.

        Returns
        -------
        `VectorFunction2D` | `VectorFunction3D`
            Interpolator instance.
        """
        raise NotImplementedError("To be defined in subclass.")

    def plot_mesh(
        self,
        data: NDArray[np.float64] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        **grid_styles,
    ) -> matplotlib.axes.Axes:
        """Plot the grid geometry to a matplotlib figure.

        Parameters
        ----------
        data
            Data array defined on the grid.
        ax
            Matplotlib axes to plot on. If None, a new figure and axes are created.
        **grid_styles
            Styles for the grid lines and faces.
        """
        raise NotImplementedError("To be defined in subclass.")
