from numpy import float64, int32
from numpy.typing import NDArray

def cell_to_5tetra(cells: NDArray[int32]) -> NDArray[int32]:
    """Generate tetrahedral indices by dividing one cell into 5 tetrahedra.

    One cubic-like cell having 8 vertices can be divided into a minimum of five tetrahedra.
    Reference: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/48509/versions/3/previews/COMP_GEOM_TLBX/html/Divide_hypercube_5_simplices_3D.html

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
    `(5N,4) ndarray`
        tetrahedra indices array, the shape of which is :math:`(5N, 4)`.

    Examples
    --------
    >>> import numpy as np
    >>> from cherab.imas.math.tetrahedralize import cell_to_5tetra
    >>>
    >>> array = np.arange(16, dtype=np.int32).reshape((2, -1))
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

def cell_to_6tetra(cells: NDArray[int32]) -> NDArray[int32]:
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
    `(6N,4) ndarray`
        Tetrahedra indices array, the shape of which is :math:`(6N, 4)`.

    Examples
    --------
    >>> import numpy as np
    >>> from cherab.imas.math.tetrahedralize import cell_to_6tetra
    >>>
    >>> array = np.arange(16, dtype=np.int32).reshape((2, -1))
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

def calculate_tetra_volume(vertices: NDArray[float64], tetrahedra: NDArray[int32]) -> float:
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
