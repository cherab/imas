"""Module defining simple interpolators for data defined on a 3D structured grid."""

from numpy import float64
from numpy.typing import ArrayLike, NDArray
from raysect.core.math import Vector3D
from raysect.core.math.function.float import Function3D
from raysect.core.math.function.vector3d import Function3D as VectorFunction3D

ZERO_VECTOR = Vector3D(0, 0, 0)

class StructGridFunction3D(Function3D):
    """Simple interpolator for the data defined on the 3D structured grid.

    Find the cell containing the point (x, y, z).
    Return the data value for this cell or the `fill_value` if the points lies outside the grid.

    Parameters
    ----------
    x : (L,) array_like
        The corners of the quadrilateral cells along x axis.
    y : (M,) array_like
        The corners of the quadrilateral cells along y axis.
    z : (N,) array_like
        The corners of the quadrilateral cells along z axis.
    grid_data : (L-1, M-1, N-1) ndarray
        Array containing data in the grid cells.
    fill_value : float, optional
        A value returned outside the grid, by default 0.
    """

    _x: NDArray[float64]
    _y: NDArray[float64]
    _z: NDArray[float64]
    _grid_data: NDArray[float64]
    _fill_value: float
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        grid_data: NDArray[float64],
        fill_value: float = 0.0,
    ) -> None: ...

class StructGridVectorFunction3D(VectorFunction3D):
    """Simple vector interpolator for the data defined on the 3D structured grid.

    Find the cell containing the point (x, y, z).
    Return the 3D vector value this cell or the `fill_vector` if the points lies outside the grid.

    Parameters
    ----------
    x : (L,) array_like
        The corners of the quadrilateral cells along x axis.
    y : (M,) array_like
        The corners of the quadrilateral cells along y axis.
    z : (N,) array_like
        The corners of the quadrilateral cells along z axis.
    grid_vectors : (3, L-1, M-1, N-1) ndarray
        Array containing 3D vectors in the grid cells.
    fill_vector : Vector3D
        3D vector returned outside the gird, by default `Vector3D(0, 0, 0)`.
    """

    _x: NDArray[float64]
    _y: NDArray[float64]
    _z: NDArray[float64]
    _grid_vectors: NDArray[float64]
    _fill_vector: Vector3D
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        grid_vectors: NDArray[float64],
        fill_vector: Vector3D = ZERO_VECTOR,
    ) -> None: ...
