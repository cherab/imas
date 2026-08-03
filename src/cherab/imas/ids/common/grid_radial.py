"""Module for loading core grid properties and calculating normalized poloidal flux."""

from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from imas.ids_structure import IDSStructure

from ._ids_numeric import get_ids_numeric_field

__all__ = ["GridData", "load_core_grid", "get_psi_norm"]


@dataclass
class GridData:
    """Dataclass for storing grid properties of the core profiles."""

    rho_tor_norm: NDArray[np.float64] | None = None
    """Normalized toroidal flux coordinate."""
    psi: NDArray[np.float64] | None = None
    """Toroidal flux [Wb]."""
    volume: NDArray[np.float64] | None = None
    """Volume enclosed by the flux surface [m^3]."""
    area: NDArray[np.float64] | None = None
    """Area of the flux surface [m^2]."""
    surface: NDArray[np.float64] | None = None
    """Surface-averaged value of the profile on the flux surface."""


def load_core_grid(grid_struct: IDSStructure) -> GridData:
    """Load grid properties of the core profiles.

    The returned dictionary values for missing data are None.

    Parameters
    ----------
    grid_struct
        The IDS structure containing the grid data for 1D profiles.

    Returns
    -------
    `.GridData`
        Instance of the `.GridData` dataclass containing the grid properties for the core profiles.
    """
    grid = GridData()
    for field in fields(grid):
        setattr(grid, field.name, get_ids_numeric_field(grid_struct, field.name))

    return grid


def get_psi_norm(
    psi: NDArray[np.float64] | None,
    psi_axis: float,
    psi_lcfs: float,
    rho_tor_norm: NDArray[np.float64] | None,
    psi_interpolator: Callable[[float], float] | None,
) -> NDArray[np.float64]:
    """Calculate normalized poloidal flux.

    Parameters
    ----------
    psi
        Poloidal flux values from the core grid.
    psi_axis
        Poloidal flux at the magnetic axis.
    psi_lcfs
        Poloidal flux at the last closed flux surface.
    rho_tor_norm
        Normalized toroidal flux values.
    psi_interpolator
        Interpolator function to map `rho_tor_norm` to `psi_norm`.
        Used only if ``psi`` is None.

    Returns
    -------
    `NDArray[np.float64]`
        Normalized poloidal flux values.

    Raises
    ------
    RuntimeError
        If both ``psi`` and ``rho_tor_norm`` are None, or if ``psi_interpolator`` is None when
        ``psi`` is None.
    """
    if psi is None:
        if psi_interpolator is None:
            raise RuntimeError(
                "Unable to map rho_tor_norm to psi_norm grid: psi_interpolator is not provided."
            )

        if rho_tor_norm is None:
            raise RuntimeError(
                "No rho_tor_norm values are available in the core grid: unable to interpolate to psi_norm."
            )

        return np.array([psi_interpolator(rho) for rho in rho_tor_norm])

    return (-psi / (2 * np.pi) - psi_axis) / (psi_lcfs - psi_axis)
