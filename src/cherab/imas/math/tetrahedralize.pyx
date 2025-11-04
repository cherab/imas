"""Module for tetrahedralization of hexahedral cells."""
import numpy as np

cimport cython
from numpy cimport import_array, ndarray, int32_t

from cython.parallel import prange

__all__ = ["cell_to_5tetra", "cell_to_6tetra", "calculate_tetra_volume"]


import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef ndarray[int32_t, ndim=2] cell_to_5tetra(const int32_t[:, ::1] cells):
    """Generate tetrahedral indices by dividing one cell into 5 tetrahedra.

    One cubic-like cell having 8 vertices can be divided into a minimum of five tetrahedra.
    Reference: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/48509/versions/3/previews/COMP_GEOM_TLBX/html/Divide_hypercube_5_simplices_3D.html  # noqa: E501

    .. note::
        If one side face of the cell is twisted, Adjacent cells do not share a plane with each
        other.
        In this case, it is required to alternate combinations of tetrahedra, or simply use
        :func:`~cherab.imas.math.tetrahedralize.cell_to_6tetra`.

    Parameters
    ----------
    cells : (N,8) ndarray [numpy.int32]
        cell indices 2D array, the shape of which is :math:`(N, 8)`, where :math:`N` is the number
        of cells.

    Returns
    -------
    (5N,4) ndarray
        tetrahedra indices array, the shape of which is :math:`(5N, 4)`.

    Examples
    --------
    >>> import numpy as np
    >>> from cherab.imas.math.tetrahedralize import cell_to_5tetra
    >>>
    >>> array = np.arrange(16, dtype=np.int32).reshape((2, -1))
    >>> array
    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15]], dtype=int32)
    >>> cell_to_5tetra(array)
        array([[ 0,  1,  3,  4],
               [ 1,  3,  4,  6],
               [ 3,  6,  7,  4],
               [ 1,  2,  3,  6],
               [ 1,  6,  4,  5],
               [ 8,  9, 11, 12],
               [ 9, 11, 12, 14],
               [11, 14, 15, 12],
               [ 9, 10, 11, 14],
               [ 9, 14, 12, 13]], dtype=int32)
    """
    cdef:
        Py_ssize_t i, j, k
        int[5][4] tetra_indices
        ndarray[int32_t, ndim=2] tetrahedra
        int32_t[:, ::1] tetrahedra_mv

    if cells.ndim != 2:
        raise ValueError("cells must be a 2 dimensional array.")

    if cells.shape[1] != 8:
        raise ValueError("cells must have a shape of (N, 8).")

    # tetrahedra indices array
    tetrahedra = np.empty((cells.shape[0] * 5, 4), dtype=np.int32)

    # five tetrahedra indices at one cell
    tetra_indices[0][:] = [0, 1, 3, 4]
    tetra_indices[1][:] = [1, 3, 4, 6]
    tetra_indices[2][:] = [3, 6, 7, 4]
    tetra_indices[3][:] = [1, 2, 3, 6]
    tetra_indices[4][:] = [1, 6, 4, 5]

    # memory view
    tetrahedra_mv = tetrahedra

    for i in prange(cells.shape[0], nogil=True):
        for j in range(5):
            for k in range(4):
                tetrahedra_mv[5 * i + j, k] = cells[i, tetra_indices[j][k]]

    return tetrahedra


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef ndarray[int32_t, ndim=2] cell_to_6tetra(const int32_t[:, ::1] cells):
    """Generate tetrahedral indices by dividing one cell into 6 tetrahedra.

    One cubic-like cell having 8 vertices can be divided into six tetrahedra.
    This manner is useful when the cell is twisted.

    Parameters
    ----------
    cells : (N,8) ndarray [numpy.int32]
        Cell indices 2D array, the shape of which is :math:`(N, 8)`, where :math:`N` is the number
        of cells.

    Returns
    -------
    (6N,4) ndarray
        Tetrahedra indices array, the shape of which is :math:`(6N, 4)`.

    Examples
    --------
    >>> import numpy as np
    >>> from cherab.imas.math.tetrahedralize import cell_to_6tetra
    >>>
    >>> array = np.arrange(16, dtype=np.int32).reshape((2, -1))
    >>> array
    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15]], dtype=int32)
    >>> cell_to_6tetra(array)
    array([[ 6,  2,  1,  0],
           [ 7,  3,  2,  0],
           [ 0,  7,  6,  2],
           [ 1,  5,  6,  4],
           [ 0,  4,  6,  7],
           [ 6,  4,  0,  1],
           [14, 10,  9,  8],
           [15, 11, 10,  8],
           [ 8, 15, 14, 10],
           [ 9, 13, 14, 12],
           [ 8, 12, 14, 15],
           [14, 12,  8,  9]], dtype=int32)
    """
    cdef:
        int i, j, k
        int[6][4] tetra_indices
        ndarray[int32_t, ndim=2] tetrahedra
        int32_t[:, ::1] tetrahedra_mv

    if cells.ndim != 2:
        raise ValueError("cells must be a 2 dimensional array.")

    if cells.shape[1] != 8:
        raise ValueError("cells must have a shape of (N, 8).")

    # tetrahedra indices array
    tetrahedra = np.empty((cells.shape[0] * 6, 4), dtype=np.int32)

    # six tetrahedra indices at one cell
    tetra_indices[0][:] = [6, 2, 1, 0]
    tetra_indices[1][:] = [7, 3, 2, 0]
    tetra_indices[2][:] = [0, 7, 6, 2]
    tetra_indices[3][:] = [1, 5, 6, 4]
    tetra_indices[4][:] = [0, 4, 6, 7]
    tetra_indices[5][:] = [6, 4, 0, 1]

    # memory view
    tetrahedra_mv = tetrahedra

    for i in prange(cells.shape[0], nogil=True):
        for j in range(6):
            for k in range(4):
                tetrahedra_mv[6 * i + j, k] = cells[i, tetra_indices[j][k]]

    return tetrahedra


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double calculate_tetra_volume(
    const double[:, ::1] vertices,
    const int32_t[:, ::1] tets,
) nogil:
    """Calculate the volume of tetrahedra.

    Parameters
    ----------
    vertices : ndarray [numpy.float64]
        Vertices of tetrahedra.
    tets : ndarray [numpy.int32]
        Tetrahedra indices.

    Returns
    -------
    float
        Volume of tetrahedra.

    Examples
    --------
    >>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> tetras = np.array([[0, 1, 2, 3]], dtype=np.int32)
    >>> calculate_tetra_volume(vertices, tetras)
    0.16666666666666666
    """
    cdef:
        double volume = 0.0
        int i
        double v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z
        double cx, cy, cz, dot_product

    for i in range(tets.shape[0]):
        v0x, v0y, v0z = vertices[tets[i, 0], 0], vertices[tets[i, 0], 1], vertices[tets[i, 0], 2]
        v1x, v1y, v1z = vertices[tets[i, 1], 0], vertices[tets[i, 1], 1], vertices[tets[i, 1], 2]
        v2x, v2y, v2z = vertices[tets[i, 2], 0], vertices[tets[i, 2], 1], vertices[tets[i, 2], 2]
        v3x, v3y, v3z = vertices[tets[i, 3], 0], vertices[tets[i, 3], 1], vertices[tets[i, 3], 2]

        cx = (v1y - v0y) * (v2z - v0z) - (v1z - v0z) * (v2y - v0y)
        cy = (v1z - v0z) * (v2x - v0x) - (v1x - v0x) * (v2z - v0z)
        cz = (v1x - v0x) * (v2y - v0y) - (v1y - v0y) * (v2x - v0x)

        dot_product = cx * (v3x - v0x) + cy * (v3y - v0y) + cz * (v3z - v0z)
        volume += (1.0 / 6.0) * abs(dot_product)

    return volume
