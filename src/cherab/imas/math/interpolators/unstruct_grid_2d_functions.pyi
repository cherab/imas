"""Module defining simple interpolators for data defined on a 2D unstructured grid."""

from numpy import float64, int32
from numpy.typing import ArrayLike, NDArray
from raysect.core.math import Vector3D
from raysect.core.math.function.float import Function2D
from raysect.core.math.function.float.function2d.interpolate.common import MeshKDTree2D
from raysect.core.math.function.vector3d import Function3D as VectorFunction3D

ZERO_VECTOR = Vector3D(0, 0, 0)

class UnstructGridFunction2D(Function2D):
    """Simple interpolator for the data defined on the 2D unstructured grid.

    Find the cell containing the point (x, y) using the KDtree algorithm.
    Return the data value for this cell or the `fill_value` if the grid does not contain the point.

    Parameters
    ----------
    vertex_coords : (N,3) array_like
        2D array-like with the vertex coordinates of triangles.
    triangles : (M,3) array_like
        2D integer array-like with the vertex indices forming the triangles.
    triangle_to_cell_map : (M,) array_like
        1D integer array-like with the indices of the grid cells (polygons) containing the
        triangles.
    grid_data : (L,) ndarray
        Array containing data in the grid cells.
    fill_value : float, optional
        Value returned outside the gird, by default 0.
    """

    _kdtree: MeshKDTree2D
    _triangle_to_cell_map: NDArray[int32]
    _grid_data: NDArray[float64]
    _fill_value: float

    def __init__(
        self,
        vertex_coords: ArrayLike,
        triangles: ArrayLike,
        triangle_to_cell_map: ArrayLike,
        grid_data: NDArray[float64],
        fill_value: float = 0.0,
    ) -> None: ...
    @classmethod
    def instance(
        cls,
        instance: UnstructGridFunction2D | UnstructGridVectorFunction2D,
        grid_data: NDArray[float64] | None = None,
        fill_value: float | None = None,
    ) -> UnstructGridFunction2D:
        """Create a new interpolator instance from an existing `UnstructGridFunction2D` or `UnstructGridVectorFunction2D` instance.

        The new interpolator instance will share the same internal acceleration data as the original
        interpolator. The grid_data of the new instance can be redefined.
        This method should be used if the user has multiple datasets that lie on the same mesh
        geometry. Using this methods avoids the repeated rebuilding of the mesh acceleration
        structures by sharing the geometry data between multiple interpolator objects.

        If created from the UnstructGridVectorFunction2D instance, the grid_data and the fill_value
        must not be None.

        Parameters
        ----------
        instance : UnstructGridFunction2D | UnstructGridVectorFunction2D
            The instance from which to create the new interpolator.
        grid_data : (L,) ndarray, optional
            Array containing data in the grid cells.
        fill_value : float, optional
            Value returned outside the grid, by default None.
            If None, inherited from the original instance.

        Returns
        -------
        UnstructGridFunction2D
            New interpolator instance.
        """

class UnstructGridVectorFunction2D(VectorFunction3D):
    """Simple vector interpolator for the data defined on the 2D unstructured grid.

    Find the cell containing the point (x, y) using the KDtree algorithm.
    Return the 3D vector value for this cell or the `fill_vector` if the grid does not contain the
    point.

    Parameters
    ----------
    vertex_coords : (N,3) array_like
        2D array-like with the vertex coordinates of triangles.
    triangles : (M,3) array_like
        2D integer array-like with the vertex indices forming the triangles.
    triangle_to_cell_map : (M,1) array_like
        1D integer array-like with the indices of the grid cells (polygons) containing the
        triangles.
    grid_vectors : (3,K) ndarray
        Array containing 3D vectors in the grid cells.
    fill_vector : Vector3D
        3D vector returned outside the gird, by default `Vector3D(0, 0, 0)`.
    """

    _kdtree: MeshKDTree2D
    _grid_vectors: NDArray[float64]
    _triangle_to_cell_map: NDArray[int32]
    _fill_vector: Vector3D

    def __init__(
        self,
        vertex_coords: ArrayLike,
        triangles: ArrayLike,
        triangle_to_cell_map: ArrayLike,
        grid_vectors: NDArray[float64],
        fill_vector: Vector3D = ZERO_VECTOR,
    ) -> None: ...
    @classmethod
    def instance(
        cls,
        instance: UnstructGridVectorFunction2D | UnstructGridFunction2D,
        grid_vectors: NDArray[float64] | None = None,
        fill_vector: Vector3D | None = None,
    ) -> UnstructGridVectorFunction2D:
        """Create a new interpolator instance from an existing `UnstructGridVectorFunction2D` or `UnstructGridFunction2D` instance.

        The new interpolator instance will share the same internal acceleration
        data as the original interpolator. The grid_vectors of the new instance can
        be redefined.
        This method should be used if the user has multiple datasets
        that lie on the same mesh geometry. Using this methods avoids the
        repeated rebuilding of the mesh acceleration structures by sharing the
        geometry data between multiple interpolator objects.

        If created from the UnstructGridFunction2D instance,
        the grid_vectors and the fill_vector must not be None.

        Parameters
        ----------
        instance : UnstructGridVectorFunction2D | UnstructGridFunction2D
            The instance from which to create the new interpolator.
        grid_vectors : (3,L) ndarray, optional
            Array containing vector grid data.
        fill_vector : Vector3D, optional
            3D vector returned outside the grid, by default None.
            If None, inherited from the original instance.

        Returns
        -------
        UnstructGridVectorFunction2D
            New interpolator instance.
        """
