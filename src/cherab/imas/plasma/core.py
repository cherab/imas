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
"""Module for loading core plasma profiles from the core_profiles IDS."""

from collections.abc import Callable

import numpy as np
from raysect.core.math import Vector3D, translate
from raysect.core.math.function.float import Function2D, Function3D, Interpolator1DArray
from raysect.core.math.function.vector3d import Constant3D as ConstantVector3D
from raysect.core.math.function.vector3d import Function2D as VectorFunction2D
from raysect.core.scenegraph._nodebase import _NodeBase
from raysect.primitive import Cylinder, Subtract
from scipy.constants import atomic_mass, electron_mass

from cherab.core import Maxwellian, Plasma, Species
from cherab.core.math import VectorAxisymmetricMapper
from cherab.core.utility import RecursiveDict
from cherab.tools.equilibrium import EFITEquilibrium
from imas import DBEntry

from ..ids.common import get_ids_time_slice
from ..ids.core_profiles import load_core_grid, load_core_species
from .equilibrium import load_equilibrium, load_magnetic_field
from .utility import warn_unsupported_species

__all__ = ["load_core_plasma"]


def load_core_plasma(
    *args,
    time: float = 0,
    occurrence: int = 0,
    equilibrium: EFITEquilibrium | None = None,
    b_field: VectorFunction2D | None = None,
    psi_interpolator: Callable[[float], float] | None = None,
    time_threshold: float = np.inf,
    parent: _NodeBase | None = None,
    **kwargs,
) -> Plasma:
    """Load core profiles and Create a `~cherab.core.plasma.node.Plasma` object.

    Prefer ``density_thermal`` over ``density`` profile.

    Parameters
    ----------
    *args
        Arguments passed to the `~imas.db_entry.DBEntry` constructor.
    time
        Time for the core plasma, by default 0.
    occurrence
        Occurrence index of the ``core_profiles`` IDS, by default 0.
    equilibrium
        Alternative `~cherab.tools.equilibrium.efit.EFITEquilibrium` used to map core
        profiles. By default None: the equilibrium is read from the same IMAS query as the
        core profiles. Ignored if the core plasma is not available.
    b_field
        Alternative 2D interpolator of the magnetic field vector (Br, Bphi, Bz).
        By default None: the magnetic field is loaded from the ``equilibrium`` IDS.
    psi_interpolator
        Alternative ``psi_norm(rho_tor_norm)`` interpolator.
        Used only if ``psi`` is missing in the core grid. By default None.
        Obtained from the ``equilibrium`` IDS.
    mask
        Mask function used for blending: ``(1 - mask) * f_edge + mask * f_core``.
        By default, uses `~cherab.tools.equilibrium.efit.EFITEquilibrium`'s `inside_lcfs`.
    time_threshold
        Maximum allowed difference between the requested time and the nearest
        available time, by default `numpy.inf`.
    parent
        Parent node in the Raysect scene graph, by default None.
        Typically a `~raysect.optical.scenegraph.world.World` instance.
    **kwargs
        Keyword arguments passed to the `~imas.db_entry.DBEntry` constructor.

    Returns
    -------
    `~cherab.core.plasma.node.Plasma`
        Plasma object with core profiles.

    Raises
    ------
    RuntimeError
        If the ``profiles_1d`` AOS in core_profiles IDS is empty.
    ValueError
        If the ``equilibrium`` argument is not an `~cherab.tools.equilibrium.efit.EFITEquilibrium`
        instance when provided.
    """
    with DBEntry(*args, **kwargs) as entry:
        core_profiles_ids = get_ids_time_slice(
            entry, "core_profiles", time=time, occurrence=occurrence, time_threshold=time_threshold
        )

    if not len(core_profiles_ids.profiles_1d):
        raise RuntimeError("The profiles_1d AOS in core_profiles IDS is empty.")

    if equilibrium is None:
        equilibrium, psi_interp = load_equilibrium(
            *args, time=time, occurrence=occurrence, with_psi_interpolator=True, **kwargs
        )
        psi_interpolator = psi_interpolator or psi_interp

    if not isinstance(equilibrium, EFITEquilibrium):
        raise ValueError("Argiment equilibrium must be a EFITEquilibrium instance.")

    if b_field is None:
        try:
            b_field = load_magnetic_field(*args, time=time, occurrence=occurrence, **kwargs)
        except RuntimeError:
            b_field = equilibrium.b_field

    core_grid = load_core_grid(core_profiles_ids.profiles_1d[0].grid)

    composition = load_core_species(core_profiles_ids.profiles_1d[0])

    psi_norm = get_psi_norm(
        core_grid["psi"],
        equilibrium.psi_axis,
        equilibrium.psi_lcfs,
        core_grid["rho_tor_norm"],
        psi_interpolator,
    )

    name = f"IMAS core plasma: time {core_profiles_ids.time[0]}, uri {entry.uri}."
    plasma = Plasma(parent=parent, name=name)

    # Create plasma geometry
    radius_inner, radius_outer = equilibrium.r_range
    zmin, zmax = equilibrium.z_range
    height = zmax - zmin
    plasma.geometry = Subtract(Cylinder(radius_outer, height), Cylinder(radius_inner, height))
    plasma.geometry_transform = translate(0, 0, zmin)

    plasma.b_field = VectorAxisymmetricMapper(b_field)

    # Add electron species
    electrons = get_core_interpolators(
        psi_norm, composition["electron"], equilibrium, return3d=True
    )
    if electrons["density_thermal"] is not None:
        electrons["density"] = electrons["density_thermal"]

    if electrons["density"] is None:
        print("Unable to create Core Plasma: electron density is not available.")
    if electrons["temperature"] is None:
        print("Unable to create Core Plasma: electron temperature is not available.")

    zero_velocity = ConstantVector3D(Vector3D(0, 0, 0))

    plasma.electron_distribution = Maxwellian(
        electrons["density"], electrons["temperature"], zero_velocity, electron_mass
    )

    warn_unsupported_species(composition, "molecule")
    warn_unsupported_species(composition, "molecular_bundle")
    warn_unsupported_species(composition, "ion_bundle")

    # Add ion and neutral species
    for species_id, profiles in composition["ion"].items():
        d = {first: second for first, second in species_id}
        species_type = d["element"]
        charge = int(round(d["z"]))

        sp_key = (species_type, charge)
        if sp_key in plasma.composition:
            print(
                f"Warning! Skipping {d['name']} species. "
                + f"Species with the same (element, charge): {sp_key} is already added."
            )
            continue

        interp = get_core_interpolators(psi_norm, profiles, equilibrium, return3d=True)
        if interp["density_thermal"] is not None:
            interp["density"] = interp["density_thermal"]

        if interp["density"] is None:
            print(f"Warning! Skipping {d['name']} species: density is not available.")
        if interp["temperature"] is None:
            print(f"Warning! Skipping {d['name']} species: temperature is not available.")

        distribution = Maxwellian(
            interp["density"],
            interp["temperature"],
            zero_velocity,
            species_type.atomic_weight * atomic_mass,
        )

        plasma.composition.add(Species(species_type, charge, distribution))

    return plasma


def get_core_interpolators(
    psi_norm: np.ndarray,
    profiles: dict,
    equilibrium: EFITEquilibrium,
    return3d: bool = False,
) -> dict[str, Function3D | Function2D | None]:
    """Create interpolators for the core profiles.

    Parameters
    ----------
    psi_norm
        Normalized poloidal flux values.
    profiles
       Dictionary with core plasma profiles.
    equilibrium
        `EFITEquilibrium` object used to map core profiles.
    return3d
        If True, return the 3D interpolators assuming rotational symmetry, by default False.

    Returns
    -------
    dict[str, Function3D | Function2D | None]
       Dictionary with core interpolators.

    Raises
    ------
    ValueError
        If the ``equilibrium`` argument is not an `~cherab.tools.equilibrium.efit.EFITEquilibrium`
        instance.
    """
    if not isinstance(equilibrium, EFITEquilibrium):
        raise ValueError("Argument equilibrium must be a EFITEquilibrium instance.")

    psi_norm, index = np.unique(psi_norm, return_index=True)

    interpolators = RecursiveDict()

    for prof_key, profile in profiles.items():
        if profile is not None:
            extrapolation_range = max(0, psi_norm[0], 1.0 - psi_norm[-1])
            func = Interpolator1DArray(
                psi_norm, profile[index], "cubic", "nearest", extrapolation_range
            )
            interpolators[prof_key] = (
                equilibrium.map3d(func) if return3d else equilibrium.map2d(func)
            )
        else:
            interpolators[prof_key] = None

    return interpolators.freeze()


def get_psi_norm(
    psi: np.ndarray | None,
    psi_axis: float,
    psi_lcfs: float,
    rho_tor_norm: np.ndarray | None,
    psi_interpolator: Callable[[float], float] | None,
) -> np.ndarray:
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
    ndarray
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
