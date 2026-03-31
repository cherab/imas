"""Module for solving some plasma models, such as the coronal equilibrium model."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cherab.core.atomic import AtomicData, Element
from cherab.openadas import OpenADAS

__all__ = [
    "solve_coronal_equilibrium",
]


def solve_coronal_equilibrium(
    element: Element,
    density: ArrayLike,
    n_e: ArrayLike,
    t_e: ArrayLike,
    atomic_data: AtomicData | None = None,
    z_min: int | None = None,
    z_max: int | None = None,
) -> NDArray[np.float64]:
    r"""Solve the charge state distribution of an element in coronal equilibrium.

    The coronal equilibrium assumption is valid when the plasma is optically thin and the timescales
    for ionisation and recombination are much shorter than the timescales for transport processes.
    Under these conditions, the charge state distribution can be determined solely by the balance
    between ionisation and recombination processes.
    The charge state density of species :math:`Z`, denoted :math:`n_Z`, can be calculated from the
    following relation:

    .. math::

        n_{Z+1} = \frac{S_Z(n_\mathrm{e}, T_\mathrm{e})}{\alpha_{Z+1}(n_\mathrm{e}, T_\mathrm{e})}\cdot n_{Z},

    where :math:`S_Z(n_\mathrm{e}, T_\mathrm{e})` and :math:`\alpha_{Z+1}(n_\mathrm{e}, T_\mathrm{e})`
    are the ionisation and recombination rates coefficients for the charge state :math:`Z` as
    functions of the electron density :math:`n_\mathrm{e}` and temperature :math:`T_\mathrm{e}`,
    respectively.

    Using the total density of the element in the range of charge states considered i.e.,

    .. math::

        n_\mathrm{total}
            &= \sum_{Z=Z_\mathrm{min}}^{Z_\mathrm{max}} n_Z\\
            &= n_{Z_\mathrm{min}} + n_{Z_\mathrm{min}+1} + n_{Z_\mathrm{min}+2} + \dots\\
            &= n_{Z_\mathrm{min}}
               +
               \frac{S_{Z_\mathrm{min}}}
                    {\alpha_{Z_\mathrm{min}+1}}
               \cdot n_{Z_\mathrm{min}}
               +
               \frac{S_{Z_\mathrm{min}+1}}
                    {\alpha_{Z_\mathrm{min}+2}}
               \cdot
               \frac{S_{Z_\mathrm{min}}}
                    {\alpha_{Z_\mathrm{min}+1}}
               \cdot n_{Z_\mathrm{min}}
               +
               \dots\\
            &= n_{Z_\mathrm{min}}
               \cdot
               \left[
                 1
                 +
                 \sum_{i=0}^{Z_\mathrm{max} - Z_\mathrm{min} - 1}
                 \left(
                   \prod_{j=0}^{i}
                      \frac{S_{Z_\mathrm{min}+j}}
                           {\alpha_{Z_\mathrm{min}+j+1}}
                 \right)
               \right]

    allows us to express the first charge state density :math:`n_{Z_\mathrm{min}}` as

    .. math::

        n_{Z_\mathrm{min}}
            =
            n_\mathrm{total}
            \cdot
            \left[
                1 + \sum_{i=0}^{Z_\mathrm{max} - Z_\mathrm{min} - 1}
                    \left(
                        \prod_{j=0}^{i}\frac{S_{Z_\mathrm{min}+j}}{\alpha_{Z_\mathrm{min}+j+1}}
                    \right)
            \right]^{-1}.

    Substituting this back into the first equation :math:`n_{Z+1} = \frac{S_Z}{\alpha_{Z+1}} n_Z`
    allows us to compute the density of each charge state as follows:

    .. math::

        \begin{bmatrix} n_{Z_\mathrm{min}}\\ n_{Z_\mathrm{min}+1}\\ n_{Z_\mathrm{min}+2}\\ \vdots\\ n_{Z_\mathrm{max}} \end{bmatrix}
        =
        \frac{
          n_\mathrm{total}
        }{
          1 + \displaystyle\sum_{i=0}^{Z_\mathrm{max} - Z_\mathrm{min} - 1}
            \left(
                \prod_{j=0}^{i}\frac{S_{Z_\mathrm{min}+j}}{\alpha_{Z_\mathrm{min}+j+1}}
            \right)
        }
        \begin{bmatrix}
            1\\
            \frac{S_{Z_\mathrm{min}}}{\alpha_{Z_\mathrm{min}+1}}\\
            \frac{S_{Z_\mathrm{min}+1}}
                 {\alpha_{Z_\mathrm{min}+2}}
              \cdot
              \frac{S_{Z_\mathrm{min}}}{\alpha_{Z_\mathrm{min}+1}}\\
            \vdots\\
            \displaystyle\prod_{j=0}^{Z_\mathrm{max} - Z_\mathrm{min} - 1}
              \frac{S_{Z_\mathrm{min}+j}}{\alpha_{Z_\mathrm{min}+j+1}}
        \end{bmatrix}.

    Parameters
    ----------
    element
        Element for which to solve the charge state distribution.
    density
        Total density of the element, :math:`n_\mathrm{total}` (in m^-3).
        Should be the 1-D array of the same shape as `n_e` and `t_e`, or a scalar.
    n_e
        Electron density, :math:`n_\mathrm{e}` (in m^-3).
        Should be the 1-D array of the same shape as `density` and `t_e`, or a scalar.
    t_e
        Electron temperature, :math:`T_\mathrm{e}` (in eV).
        Should be the 1-D array of the same shape as `density` and `n_e`, or a scalar.
    atomic_data
        Atomic data provider to use for ionisation and recombination rates.
        If None, `~cherab.openadas.openadas.OpenADAS` will be used by default.
    z_min
        Minimum charge state to consider (inclusive), :math:`Z_\mathrm{min}`. If None, defaults to 0.
    z_max
        Maximum charge state to consider (inclusive), :math:`Z_\mathrm{max}`. If None, defaults to the atomic number of the element.

    Returns
    -------
    `numpy.ndarray`
        Density of the element in each charge state (in m^-3), with shape ``(z_max - z_min + 1, *density.shape)``.

    Raises
    ------
    ValueError
        If `z_min` or `z_max` are out of valid range, or if `density`, `n_e`, and `t_e` do not have the same shape.
    TypeError
        If `atomic_data` is not an instance of `~cherab.core.atomic.interface.AtomicData`.

    Examples
    --------
    >>> from cherab.core.atomic.elements import neon as element
    >>> density = 1.0  # if the density is 1, the output will be the relative abundance of each charge state
    >>> n_e = 1e20
    >>> t_e = 10
    >>> result = solve_coronal_equilibrium(element, density, n_e, t_e)
    >>> print([f"{x:.1%}" for x in result.ravel()])
    ['0.0%', '0.0%', '13.1%', '80.4%', '6.4%', '0.0%', '0.0%', '0.0%', '0.0%', '0.0%', '0.0%']
    """
    # Initialize variables if not provided
    z_min = z_min if z_min is not None else 0
    z_max = z_max if z_max is not None else element.atomic_number
    atomic_data = atomic_data if atomic_data is not None else OpenADAS(permit_extrapolation=True)

    # Validate z_min/z_max
    if z_min < 0:
        raise ValueError("z_min must be non-negative.")
    if z_max > element.atomic_number:
        raise ValueError("z_max cannot exceed the atomic number of the element.")
    if z_min > z_max:
        raise ValueError("z_min cannot be greater than z_max.")

    # Validate atomic_data
    if not isinstance(atomic_data, AtomicData):
        raise TypeError("atomic_data must be an instance of AtomicData.")

    # Validate array-like inputs
    density = np.atleast_1d(density)
    n_e = np.atleast_1d(n_e)
    t_e = np.atleast_1d(t_e)

    if density.shape != n_e.shape or density.shape != t_e.shape:
        raise ValueError("density, n_e, and t_e must have the same shape.")

    # Define charge states and compute ionisation/recombination ratios
    charges = np.arange(z_min, z_max + 1, dtype=int)
    ratios = np.zeros((len(charges), len(density)))

    for i_charge in range(len(charges) - 1):
        rate_ion = atomic_data.ionisation_rate(element, charges[i_charge])
        rate_rec = atomic_data.recombination_rate(element, charges[i_charge] + 1)

        for i_density in range(density.size):
            ratios[i_charge, i_density] = rate_ion(n_e[i_density], t_e[i_density]) / rate_rec(
                n_e[i_density], t_e[i_density]
            )

    # Compute fractional abundances of each charge state
    fractions = np.vstack(
        [
            np.ones((1, ratios.shape[1]), dtype=float),
            np.cumprod(ratios[:-1, :], axis=0, dtype=float),
        ]
    )

    # Normalize
    fractions /= np.sum(fractions, axis=0)

    # Return the charge state densities by multiplying the fractions with the total density
    return fractions * density
