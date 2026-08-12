"""Fast geometry calculations for polygonal meshes."""

import numpy as np

cimport cython
from cython.parallel import prange
from numpy cimport import_array, ndarray, int32_t

__all__ = ["calculate_2d_cell_geometry"]

DEF OPENMP_MIN_CELLS = 100000


import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _calculate_cell_geometry(
    const double[:, ::1] vertices,
    const int32_t[:, ::1] triangles,
    const int32_t[:, ::1] cell_to_triangle,
    Py_ssize_t cell_index,
    double[:, ::1] cell_centres,
    double[::1] cell_areas,
) noexcept nogil:
    cdef:
        Py_ssize_t j, triangle_index
        int32_t vertex0, vertex1, vertex2
        double x0, y0, x1, y1, x2, y2
        double triangle_area, area_sum = 0.0
        double first_moment_x = 0.0, first_moment_y = 0.0

    for j in range(cell_to_triangle[cell_index, 1]):
        triangle_index = cell_to_triangle[cell_index, 0] + j
        vertex0 = triangles[triangle_index, 0]
        vertex1 = triangles[triangle_index, 1]
        vertex2 = triangles[triangle_index, 2]

        x0, y0 = vertices[vertex0, 0], vertices[vertex0, 1]
        x1, y1 = vertices[vertex1, 0], vertices[vertex1, 1]
        x2, y2 = vertices[vertex2, 0], vertices[vertex2, 1]

        triangle_area = 0.5 * abs(
            (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
        )
        area_sum += triangle_area
        first_moment_x += triangle_area * (x0 + x1 + x2) / 3.0
        first_moment_y += triangle_area * (y0 + y1 + y2) / 3.0

    cell_areas[cell_index] = area_sum
    if area_sum > 0.0:
        cell_centres[cell_index, 0] = first_moment_x / area_sum
        cell_centres[cell_index, 1] = first_moment_y / area_sum
    else:
        cell_centres[cell_index, 0] = 0.0
        cell_centres[cell_index, 1] = 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef tuple calculate_2d_cell_geometry(
    const double[:, ::1] vertices,
    const int32_t[:, ::1] triangles,
    const int32_t[:, ::1] cell_to_triangle,
):
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
    tuple of ndarray
        Cell centroids with shape ``(K, 2)`` and areas with shape ``(K,)``.
    """
    cdef:
        Py_ssize_t i
        ndarray[double, ndim=2] cell_centres
        ndarray[double, ndim=1] cell_areas
        double[:, ::1] centres_mv
        double[::1] areas_mv

    if vertices.shape[1] != 2:
        raise ValueError("vertices must have a shape of (N, 2).")
    if triangles.shape[1] != 3:
        raise ValueError("triangles must have a shape of (M, 3).")
    if cell_to_triangle.shape[1] != 2:
        raise ValueError("cell_to_triangle must have a shape of (K, 2).")

    cell_centres = np.empty((cell_to_triangle.shape[0], 2), dtype=np.float64)
    cell_areas = np.empty(cell_to_triangle.shape[0], dtype=np.float64)

    centres_mv = cell_centres
    areas_mv = cell_areas

    for i in prange(
        cell_to_triangle.shape[0],
        nogil=True,
        schedule="static",
        use_threads_if=cell_to_triangle.shape[0] >= OPENMP_MIN_CELLS,
    ):
        _calculate_cell_geometry(
            vertices,
            triangles,
            cell_to_triangle,
            i,
            centres_mv,
            areas_mv,
        )

    if np.any(cell_areas == 0.0):
        raise ValueError("All cells must have a positive area.")

    return cell_centres, cell_areas
