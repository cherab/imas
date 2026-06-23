"""Fourier-Bezier reconstruction for JOREK discretized fields.

This module implements reconstruction of fields discretized using
Bezier basis functions in the poloidal direction and Fourier modes
in the toroidal direction (Fourier-Bezier scheme as used in JOREK).
"""

cimport cython
cimport numpy as np

import numpy as np

__all__ = [
    "FourierBezierConstructor",
    "py_bezier_basis",
    "py_bezier_basis_derivative_wrt_s",
    "py_bezier_basis_derivative_wrt_t",
    "py_fourier_mode",
]

# ============================================================================
# === Fourier-Bezier Mathematics Kernels ===
# ============================================================================

DEF TO_RAD = 3.14159265358979323846 / 180.0


cdef void bezier_basis(double *s, double *t, double[4][4] a) noexcept nogil:
    """Compute cubic Bezier basis functions on [0,1]^2."""
    a[0][0] = (1 - s[0]) ** 2 * (1 + 2 * s[0]) * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[0][1] = 3 * (1 - s[0]) ** 2 * s[0] * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[0][2] = 3 * (1 - s[0]) ** 2 * (1 + 2 * s[0]) * (-1 + t[0]) ** 2 * t[0]
    a[0][3] = 9 * (1 - s[0]) ** 2 * s[0] * (1 - t[0]) ** 2 * t[0]

    a[1][0] = s[0] ** 2 * (3 - 2 * s[0]) * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[1][1] = 3 * (1 - s[0]) * s[0] ** 2 * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[1][2] = 3 * s[0] ** 2 * (3 - 2 * s[0]) * (1 - t[0]) ** 2 * t[0]
    a[1][3] = 9 * (1 - s[0]) * s[0] ** 2 * (1 - t[0]) ** 2 * t[0]

    a[2][0] = s[0] ** 2 * (3 - 2 * s[0]) * t[0] ** 2 * (3 - 2 * t[0])
    a[2][1] = 3 * (1 - s[0]) * s[0] ** 2 * t[0] ** 2 * (3 - 2 * t[0])
    a[2][2] = 3 * s[0] ** 2 * (3 - 2 * s[0]) * (1 - t[0]) * t[0] ** 2
    a[2][3] = 9 * (1 - s[0]) * s[0] ** 2 * (1 - t[0]) * t[0] ** 2

    a[3][0] = (1 - s[0]) ** 2 * (1 + 2 * s[0]) * t[0] ** 2 * (3 - 2 * t[0])
    a[3][1] = 3 * (1 - s[0]) ** 2 * s[0] * t[0] ** 2 * (3 - 2 * t[0])
    a[3][2] = 3 * (1 - s[0]) ** 2 * (1 + 2 * s[0]) * (1 - t[0]) * t[0] ** 2
    a[3][3] = 9 * (1 - s[0]) ** 2 * s[0] * (1 - t[0]) * t[0] ** 2


cdef void bezier_basis_derivative_wrt_s(double *s, double *t, double[4][4] a) noexcept nogil:
    """Compute derivatives of cubic Bezier basis functions w.r.t. s."""
    a[0][0] = -6 * (1 - s[0]) * s[0] * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[0][1] = 3 * (1 - s[0]) * (1 - 3 * s[0]) * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[0][2] = -18 * (1 - s[0]) * s[0] * (1 - t[0]) ** 2 * t[0]
    a[0][3] = 9 * (1 - s[0]) * (1 - 3 * s[0]) * (1 - t[0]) ** 2 * t[0]

    a[1][0] = 6 * (1 - s[0]) * s[0] * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[1][1] = 3 * s[0] * (2 - 3 * s[0]) * (1 - t[0]) ** 2 * (1 + 2 * t[0])
    a[1][2] = 18 * (1 - s[0]) * s[0] * (1 - t[0]) ** 2 * t[0]
    a[1][3] = 9 * s[0] * (2 - 3 * s[0]) * (1 - t[0]) ** 2 * t[0]

    a[2][0] = 6 * (1 - s[0]) * s[0] * t[0] ** 2 * (3 - 2 * t[0])
    a[2][1] = 3 * s[0] * (2 - 3 * s[0]) * t[0] ** 2 * (3 - 2 * t[0])
    a[2][2] = 18 * (1 - s[0]) * s[0] * (1 - t[0]) * t[0] ** 2
    a[2][3] = 9 * s[0] * (2 - 3 * s[0]) * (1 - t[0]) * t[0] ** 2

    a[3][0] = -6 * (1 - s[0]) * s[0] * t[0] ** 2 * (3 - 2 * t[0])
    a[3][1] = 3 * (1 - 3 * s[0]) * (1 - s[0]) * t[0] ** 2 * (3 - 2 * t[0])
    a[3][2] = -18 * (1 - s[0]) * s[0] * (1 - t[0]) * t[0] ** 2
    a[3][3] = 9 * (1 - 3 * s[0]) * (1 - s[0]) * (1 - t[0]) * t[0] ** 2


cdef void bezier_basis_derivative_wrt_t(double *s, double *t, double[4][4] a) noexcept nogil:
    """Compute derivatives of cubic Bezier basis functions w.r.t. t."""
    a[0][0] = -6 * (1 - s[0]) ** 2 * (1 + 2 * s[0]) * (1 - t[0]) * t[0]
    a[0][1] = -18 * (1 - s[0]) ** 2 * s[0] * (1 - t[0]) * t[0]
    a[0][2] = 3 * (1 - s[0]) ** 2 * (1 + 2 * s[0]) * (1 - t[0]) * (1 - 3 * t[0])
    a[0][3] = 9 * (1 - s[0]) ** 2 * s[0] * (1 - t[0]) * (1 - 3 * t[0])

    a[1][0] = -6 * s[0] ** 2 * (3 - 2 * s[0]) * (1 - t[0]) * t[0]
    a[1][1] = -18 * (1 - s[0]) * s[0] ** 2 * (1 - t[0]) * t[0]
    a[1][2] = 3 * s[0] ** 2 * (3 - 2 * s[0]) * (1 - 3 * t[0]) * (1 - t[0])
    a[1][3] = 9 * (1 - s[0]) * s[0] ** 2 * (1 - 3 * t[0]) * (1 - t[0])

    a[2][0] = 6 * s[0] ** 2 * (3 - 2 * s[0]) * (1 - t[0]) * t[0]
    a[2][1] = 18 * (1 - s[0]) * s[0] ** 2 * (1 - t[0]) * t[0]
    a[2][2] = 3 * s[0] ** 2 * (3 - 2 * s[0]) * t[0] * (2 - 3 * t[0])
    a[2][3] = 9 * (1 - s[0]) * s[0] ** 2 * t[0] * (2 - 3 * t[0])

    a[3][0] = 6 * (1 - s[0]) ** 2 * (1 + 2 * s[0]) * (1 - t[0]) * t[0]
    a[3][1] = 18 * (1 - s[0]) ** 2 * s[0] * (1 - t[0]) * t[0]
    a[3][2] = 3 * (1 - s[0]) ** 2 * (1 + 2 * s[0]) * t[0] * (2 - 3 * t[0])
    a[3][3] = 9 * (1 - s[0]) ** 2 * s[0] * t[0] * (2 - 3 * t[0])


cdef enum SpaceType:
    rz = 0
    fourier = 1


cdef enum DomainType:
    vertex = 0
    edge = 1
    face = 2


cdef class FourierBezierConstructor:
    """Reconstruct field values on poloidal faces from Fourier-Bezier coefficients.

    This class reconstructs physical quantities (e.g., emissivity, temperature)
    discretized using Bezier polynomials in the poloidal plane and Fourier modes
    in the toroidal direction, following the JOREK MHD code discretization scheme.

    Parameters
    ----------
    grid_ggd : IDSStructure
        The grid GGD `~imas.ids_structure.IDSStructure` containing the `space`
        `~imas.ids_struct_array.IDSStructArray`.
    coefficients : array_like, optional
        The coefficients of the physical quantity.
    """

    cdef:
        int _num_faces
        int _num_vertices
        int _num_toroidal_modes
        int _fourier_periodicity
        np.int32_t[:, ::1] _vertex_indices
        double[:, :, ::1] _vertex_coefficients
        double[:, :, ::1] _scale_factors
        double[:, :, :, ::1] _coefficients

    def __init__(self, object grid_ggd, object coefficients=None):
        cdef int i_face, i_vert, i_dof, i_node

        sp_rz = grid_ggd.space[SpaceType.rz]
        sp_fourier = grid_ggd.space[SpaceType.fourier]

        self._num_faces = len(sp_rz.objects_per_dimension[DomainType.face].object)
        self._num_vertices = len(sp_rz.objects_per_dimension[DomainType.vertex].object)
        self._num_toroidal_modes = len(sp_fourier.objects_per_dimension[DomainType.vertex].object)
        self._fourier_periodicity = sp_fourier.geometry_type.index

        self._vertex_indices = np.empty((self._num_faces, 4), dtype=np.int32)
        self._scale_factors = np.empty((self._num_faces, 4, 4), dtype=np.double)

        for i_vert in range(4):
            for i_face, obj in enumerate(sp_rz.objects_per_dimension[DomainType.face].object):
                self._vertex_indices[i_face, i_vert] = obj.nodes[i_vert] - 1
                self._scale_factors[i_face, i_vert, 0] = obj.geometry_2d[0, i_vert]
                self._scale_factors[i_face, i_vert, 1] = obj.geometry_2d[1, i_vert]
                self._scale_factors[i_face, i_vert, 2] = obj.geometry_2d[2, i_vert]
                self._scale_factors[i_face, i_vert, 3] = obj.geometry_2d[3, i_vert]

        self._vertex_coefficients = np.empty((self._num_vertices, 2, 4), dtype=np.double)
        for i_dof in range(4):
            for i_node, obj in enumerate(sp_rz.objects_per_dimension[DomainType.vertex].object):
                self._vertex_coefficients[i_node, 0, i_dof] = obj.geometry_2d[0, i_dof]
                self._vertex_coefficients[i_node, 1, i_dof] = obj.geometry_2d[1, i_dof]

        if coefficients is not None:
            self.coefficients = coefficients
        else:
            self._coefficients = None

    @property
    def coefficients(self) -> np.ndarray:
        """Fourier-Bezier coefficient tensor as a numpy array.

        The coefficients are the coefficients that define the physical quantity on the JOREK grid.

        Returns
        -------
        (I, K, J, L) ndarray
            The coefficients of the physical quantity.

            :I: the number of faces in the RZ grid,
            :K: the number of vertices in one face, by default 4,
            :J: the number of degrees of freedom in one vertex, by default 4,
            :L: the number of toroidal fourier modes.
        """
        return np.asarray(self._coefficients)

    @coefficients.setter
    def coefficients(self, object coefficients):
        coefficients = np.asarray(coefficients)
        if coefficients.ndim != 2:
            raise ValueError("Coefficients must be a 2-dimensional array.")
        if coefficients.shape[0] != self._num_vertices * self._num_toroidal_modes:
            raise ValueError(
                "Coefficients array must have "
                f"{self._num_vertices * self._num_toroidal_modes} rows."
            )
        if coefficients.shape[1] != 4:
            raise ValueError("Coefficients array must have 4 columns.")

        if not coefficients.flags["C_CONTIGUOUS"]:
            coefficients = np.ascontiguousarray(coefficients)

        self._coefficients = self._set_coefficients(coefficients)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double[:, :, :, ::1] _set_coefficients(self, double[:, ::1] coefficients):
        cdef:
            int i_face, i_vert, i_dof, i_mode, index
            np.ndarray[double, ndim=4] coeff
            double[:, :, :, ::1] coeff_view

        coeff = np.empty((self._num_faces, 4, 4, self._num_toroidal_modes), dtype=float)
        coeff_view = coeff

        for i_face in range(self._num_faces):
            for i_vert in range(self._vertex_indices.shape[1]):
                for i_dof in range(self._scale_factors.shape[2]):
                    for i_mode in range(self._num_toroidal_modes):
                        index = self._vertex_indices[i_face, i_vert] + i_mode * self._num_vertices
                        coeff_view[i_face, i_vert, i_dof, i_mode] = coefficients[index, i_dof]

        return coeff_view

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef double _get_value(self, double *s, double *t, double *phi, int face) noexcept nogil:
        cdef:
            double value = 0.0
            int i_vert, i_dof, i_mode
            double[4][4] bezier_bases

        bezier_basis(s, t, bezier_bases)

        for i_vert in range(4):
            for i_dof in range(4):
                for i_mode in range(self._num_toroidal_modes):
                    value += (
                        self._coefficients[face, i_vert, i_dof, i_mode]
                        * bezier_bases[i_vert][i_dof]
                        * self._scale_factors[face, i_vert, i_dof]
                        * fourier_mode(phi, &i_mode, &self._fourier_periodicity)
                    )

        return value

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cdef void _interp_rz(
        self,
        double *s,
        double *t,
        double *phi,
        double *r,
        double *z,
        double *drds,
        double *dzds,
        double *drdt,
        double *dzdt,
        int face,
    ) noexcept nogil:
        cdef:
            int i_vert, i_dof
            double[4][4] bezier_bases, bezier_bases_derivative_s, bezier_bases_derivative_t

        bezier_basis(s, t, bezier_bases)
        bezier_basis_derivative_wrt_s(s, t, bezier_bases_derivative_s)
        bezier_basis_derivative_wrt_t(s, t, bezier_bases_derivative_t)

        r[0] = 0.0
        z[0] = 0.0
        drds[0] = 0.0
        dzds[0] = 0.0
        drdt[0] = 0.0
        dzdt[0] = 0.0

        for i_vert in range(4):
            for i_dof in range(4):
                r[0] += (
                    self._vertex_coefficients[self._vertex_indices[face, i_vert], 0, i_dof]
                    * bezier_bases[i_vert][i_dof]
                    * self._scale_factors[face, i_vert, i_dof]
                )
                z[0] += (
                    self._vertex_coefficients[self._vertex_indices[face, i_vert], 1, i_dof]
                    * bezier_bases[i_vert][i_dof]
                    * self._scale_factors[face, i_vert, i_dof]
                )
                drds[0] += (
                    self._vertex_coefficients[self._vertex_indices[face, i_vert], 0, i_dof]
                    * bezier_bases_derivative_s[i_vert][i_dof]
                    * self._scale_factors[face, i_vert, i_dof]
                )
                dzds[0] += (
                    self._vertex_coefficients[self._vertex_indices[face, i_vert], 1, i_dof]
                    * bezier_bases_derivative_s[i_vert][i_dof]
                    * self._scale_factors[face, i_vert, i_dof]
                )
                drdt[0] += (
                    self._vertex_coefficients[self._vertex_indices[face, i_vert], 0, i_dof]
                    * bezier_bases_derivative_t[i_vert][i_dof]
                    * self._scale_factors[face, i_vert, i_dof]
                )
                dzdt[0] += (
                    self._vertex_coefficients[self._vertex_indices[face, i_vert], 1, i_dof]
                    * bezier_bases_derivative_t[i_vert][i_dof]
                    * self._scale_factors[face, i_vert, i_dof]
                )

    @cython.cdivision(True)
    cdef double _average_gaussian(self, double *phi, int face) noexcept nogil:
        cdef:
            int i_s, i_t
            double value = 0.0
            double total_volume = 0.0
            double r, z, drds, dzds, drdt, dzdt, volume
            double[4] st_points = [
                0.0694318442029735,
                0.3300094782075720,
                0.6699905217924280,
                0.9305681557970265,
            ]
            double[4] st_weight = [
                0.173927422568727,
                0.326072577431273,
                0.326072577431273,
                0.173927422568727,
            ]

        for i_s in range(4):
            for i_t in range(4):
                self._interp_rz(
                    &st_points[i_s], &st_points[i_t], phi, &r, &z, &drds, &dzds, &drdt, &dzdt, face
                )

                volume = (drds * dzdt - dzds * drdt) * r * st_weight[i_s] * st_weight[i_t]
                total_volume += volume
                value += self._get_value(&st_points[i_s], &st_points[i_t], phi, face) * volume

        return value / total_volume

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray average_gaussian_faces(self, double phi):
        """Calculate face-averaged field value at a single toroidal angle [deg].

        Parameters
        ----------
        phi : float
            Toroidal angle [deg] at which to evaluate the face-averaged field values.

        Returns
        -------
        (F,) ndarray
            Face-averaged field values for each face at the given toroidal angle.
        """
        cdef:
            int i_face
            np.ndarray[np.float64_t, ndim=1] values
            double[::1] values_view

        if self._coefficients is None:
            raise ValueError("Coefficients are not provided.")

        values = np.empty(self._num_faces, dtype=np.float64)
        values_view = values

        for i_face in range(self._num_faces):
            values_view[i_face] = self._average_gaussian(&phi, i_face)

        return values

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray average_gaussian_faces_per_toroidal(self, object phis):
        """Calculate face-averaged field values for each toroidal angle [deg].

        Parameters
        ----------
        phis : array_like
            Array of toroidal angles [deg] at which to evaluate the face-averaged field values.

        Returns
        -------
        (N, F) ndarray
            Face-averaged field values for each toroidal angle.
            Axis 0: Toroidal angle index. Axis 1: Face index.
        """
        cdef:
            int i_face, i_phi
            np.ndarray[np.float64_t, ndim=1] phi_array
            np.ndarray[np.float64_t, ndim=2] values
            double[::1] phi_view
            double[:, ::1] values_view

        if self._coefficients is None:
            raise ValueError("Coefficients are not provided.")

        phi_array = np.asarray(phis)
        if phi_array.ndim != 1:
            raise ValueError("The phis array must be a 1-dimensional array.")
        if phi_array.shape[0] == 0:
            raise ValueError("The phis array must have at least one element.")

        values = np.empty((phi_array.shape[0], self._num_faces), dtype=np.float64)

        phi_view = phi_array
        values_view = values

        for i_phi in range(phi_view.shape[0]):
            for i_face in range(self._num_faces):
                values_view[i_phi, i_face] = self._average_gaussian(&phi_view[i_phi], i_face)

        return values


# --------------------------------------------------------------------------------------------
# Python wrappers for Cython functions (for use in Python code, not optimized for performance)
# --------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def py_bezier_basis(s: float, t: float) -> np.ndarray:
    r"""Calculate the bezier bases for each node and degree of freedom.

    .. note::
        This is the python wrapper for the Cython function :func:`bezier_basis`.
        If you are using this function in a loop, consider using the Cython function directly.

    JOREK uses two-thirds order Bernstein polynomial :math:`B_{i, j}^{(3)}(s, t)` defined as:

    .. math::
        B_{i, j}^{(3)}(s, t) \equiv B_i^{(3)}(s) B_j^{(3)}(t)

        B_i^{(3)}(x) \equiv \begin{pmatrix} 3\\i \end{pmatrix} x^i (1 - x)^{3 - i}

    where :math:`1 \leq i, j \leq 4` and :math:`0 \leq s, t \leq 1`.

    The bezier basis (or cubic Hermite finite element) :math:`H_{i, j}(s, t)` is constructed as
    linear combinations of the above Bernstein polynomials, written as a product of 1D basis
    functions:

    .. math::
        H_{i, j}(s, t) \equiv H_i(s) H_j(t),

    which satisfy the following boundary conditions:

    .. math::
        H_1(0) = 1, H_1'(0) = 0, H_1(1) = 0, H_1'(1) = 0

        H_2(0) = 0, H_2'(0) = 1, H_2(1) = 0, H_2'(1) = 1

        H_3(0) = 0, H_3'(0) = 0, H_3(1) = 1, H_3'(1) = 0

        H_4(0) = 0, H_4'(0) = 0, H_4(1) = 0, H_4'(1) = 1

    Parameters
    ----------
    s : float
        First parameter in the range [0, 1].
    t : float
        Second parameter in the range [0, 1].

    Returns
    -------
    (4, 4) ndarray
        Bezier bases :math:`H_{i, j}(s, t)` for each node :math:`i` and degree of freedom :math:`j`.
        Axis 0: Node index. Axis 1: Degree of freedom index.

    Examples
    --------
    >>> py_bezier_basis(0.5, 0.5)
    array([[0.25    , 0.1875  , 0.1875  , 0.140625],
           [0.25    , 0.1875  , 0.1875  , 0.421875],
           [0.25    , 0.1875  , 0.1875  , 0.140625],
           [0.25    , 0.1875  , 0.1875  , 0.140625]])
    """
    cdef:
        double[4][4] a
        int i, j
        np.ndarray a_arr = np.empty((4, 4), dtype=np.float64)
        double[:, ::1] a_view

    bezier_basis(&s, &t, a)

    a_view = a_arr
    for i in range(4):
        for j in range(4):
            a_view[i, j] = a[i][j]

    return a_arr


@cython.boundscheck(False)
@cython.wraparound(False)
def py_bezier_basis_derivative_wrt_s(s: float, t: float) -> np.ndarray:
    """Calculate the derivative of the bezier basis w.r.t. :math:`s` for each node and degree of
    freedom.

    .. note::
        This is the python wrapper for the Cython function :func:`bezier_basis_derivative_wrt_s`.
        If you are using this function in a loop, consider using the Cython function directly.

    Parameters
    ----------
    s : float
        First parameter in the range [0, 1].
    t : float
        Second parameter in the range [0, 1].

    Returns
    -------
    (4, 4) ndarray
        Derivative of the bezier basis w.r.t. :math:`s` for each node and degree of
        freedom.
        Axis 0: Node index. Axis 1: Degree of freedom index.
    """
    cdef:
        double[4][4] a
        int i, j
        np.ndarray a_arr = np.empty((4, 4), dtype=np.float64)
        double[:, ::1] a_view

    bezier_basis_derivative_wrt_s(&s, &t, a)

    a_view = a_arr
    for i in range(4):
        for j in range(4):
            a_view[i, j] = a[i][j]
    return a_arr


@cython.boundscheck(False)
@cython.wraparound(False)
def py_bezier_basis_derivative_wrt_t(s: float, t: float) -> np.ndarray:
    """Calculate the derivative of the bezier basis w.r.t. :math:`t` for each node and degree of
    freedom.

    .. note::
        This is the python wrapper for the Cython function :func:`bezier_basis_derivative_wrt_t`.
        If you are using this function in a loop, consider using the Cython function directly.

    Parameters
    ----------
    s : float
        First parameter in the range [0, 1].
    t : float
        Second parameter in the range [0, 1].

    Returns
    -------
    (4, 4) ndarray
        Derivative of the bezier basis w.r.t. :math:`t` for each node and degree of
        freedom.
        Axis 0: Node index. Axis 1: Degree of freedom index.
    """
    cdef:
        double[4][4] a
        int i, j
        np.ndarray a_arr = np.empty((4, 4), dtype=np.float64)
        double[:, ::1] a_view

    bezier_basis_derivative_wrt_t(&s, &t, a)

    a_view = a_arr
    for i in range(4):
        for j in range(4):
            a_view[i, j] = a[i][j]
    return a_arr


def py_fourier_mode(double phi, int index, int periodicity = 1) -> float:
    r"""Calculate the value of a Fourier mode at a given angle :math:`\varphi`.

    .. note::
        This is the python wrapper for the Cython function :func:`fourier_mode`.
        If you are using this function in a loop, consider using the Cython function directly.

    Fourier modes :math:`Z_l(\varphi)` corresponding to the different mode indices :math:`l` with
    :math:`n_\mathrm{p}` as periodicity of the simulation are defined as:

    .. math::
        Z_l(\varphi) = \begin{cases}
            1
                & \text{if } l = 0, \\
            \sin\left(\displaystyle\frac{l}{2}n_\mathrm{p} \varphi\right)
                & \text{if } l \text{ is even}, \\
            \cos\left(\displaystyle\frac{l + 1}{2}n_\mathrm{p} \varphi\right)
                & \text{if } l \text{ is odd}.
        \end{cases}

    Parameters
    ----------
    phi : float
        Angle :math:`\varphi` at which to evaluate the Fourier mode in degree.
    index : int
        Index :math:`l` of the Fourier mode.
    periodicity : int, optional
        Periodicity :math:`n_\mathrm{p}` of the simulation, by default 1.

    Returns
    -------
    float
        Value of the Fourier mode at the given angle :math:`\varphi`.
    """
    return fourier_mode(&phi, &index, &periodicity)
