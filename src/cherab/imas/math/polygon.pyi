from numpy import float64, int32
from numpy.typing import NDArray

def calculate_2d_cell_geometry(
    vertices: NDArray[float64],
    triangles: NDArray[int32],
    cell_to_triangle: NDArray[int32],
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Calculate areas and area centroids of triangulated 2-D cells.

    Each polygonal cell is represented by one or more triangles. The centroid is
    the area-weighted centroid of those triangles.

    Parameters
    ----------
    vertices : (N, 2) ndarray [numpy.float64]
        Coordinates of all polygon vertices.
    triangles : (M, 3) ndarray [numpy.int32]
        Vertex indices of the triangles forming the cells.
    cell_to_triangle : (K, 2) ndarray [numpy.int32]
        For each cell, the first triangle index and number of triangles.

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        Cell centroids with shape ``(K, 2)`` and areas with shape ``(K,)``.
    """
