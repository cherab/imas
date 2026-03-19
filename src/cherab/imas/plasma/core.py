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
from numpy.typing import NDArray
from raysect.core.math import translate
from raysect.core.math.function.float import Interpolator1DArray
from raysect.core.math.function.vector3d import Function2D as VectorFunction2D
from raysect.core.scenegraph._nodebase import _NodeBase
from raysect.primitive import Cylinder, Subtract
from scipy.constants import atomic_mass, electron_mass

from cherab.core import AtomicData, Maxwellian, Plasma, Species
from cherab.core.math import VectorAxisymmetricMapper
from cherab.imas.ids.core_profiles.load_profiles import ProfileData
from cherab.tools.equilibrium import EFITEquilibrium
from imas import DBEntry

from ..ids.common import get_ids_time_slice
from ..ids.core_profiles import load_core_grid, load_core_species
from .equilibrium import load_equilibrium, load_magnetic_field
from .utility import ZERO_VELOCITY, ProfileInterporater, warn_unsupported_species

__all__ = ["load_core_plasma"]


def load_core_plasma(
    *args,
    time: float = 0,
    occurrence: int = 0,
    equilibrium: EFITEquilibrium | None = None,
    b_field: VectorFunction2D | None = None,
    psi_interpolator: Callable[[float], float] | None = None,
    time_threshold: float = np.inf,
    split_ion_bundles: bool = True,
    atomic_data: AtomicData | None = None,
    parent: _NodeBase | None = None,
    **kwargs,
) -> Plasma:
    """Load core profiles and Create a `~cherab.core.plasma.node.Plasma` object.

    Prefer ``density_thermal`` over ``density`` profile.

    The distribution of each species is defined with `~cherab.core.distribution.Maxwellian` using
    its density and temperature profiles, which are mapped to 3D using the provided `equilibrium`.

    The plasma geometry is defined as a cylindrical annulus between the inner and outer radii of the
    equilibrium, and between the minimum and maximum z values of the equilibrium.

    The ion bundle species are split into their constituent charge states using `.solve_coronal_equilibrium`.

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
    split_ion_bundles
        Whether to split ion bundles into their constituent charge states using
        `.solve_coronal_equilibrium`, by default True.
    atomic_data
        Atomic data provider class for this plasma, by default None.
        If None, some species (e.g. ion_bundle) may not be properly loaded.
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
    # ---------------------------------------------------------
    # === Load core profiles and equilibrium data from IMAS ===
    # ---------------------------------------------------------
    # Load required data from the core_profiles IDS and form the core grid and species composition
    # data structures.
    with DBEntry(*args, **kwargs) as entry:
        core_profiles_ids = get_ids_time_slice(
            entry, "core_profiles", time=time, occurrence=occurrence, time_threshold=time_threshold
        )

    if not len(core_profiles_ids.profiles_1d):
        raise RuntimeError("The profiles_1d AOS in core_profiles IDS is empty.")

    # Load equilibrium and magnetic field data
    if equilibrium is None:
        equilibrium, psi_interp = load_equilibrium(
            *args, time=time, occurrence=occurrence, with_psi_interpolator=True, **kwargs
        )
        psi_interpolator = psi_interpolator or psi_interp

    if not isinstance(equilibrium, EFITEquilibrium):
        raise ValueError("Argument equilibrium must be a EFITEquilibrium instance.")

    if b_field is None:
        try:
            b_field = load_magnetic_field(*args, time=time, occurrence=occurrence, **kwargs)
        except RuntimeError:
            b_field = equilibrium.b_field

    # Create core grid
    core_grid = load_core_grid(core_profiles_ids.profiles_1d[0].grid)

    psi_norm = get_psi_norm(
        core_grid.psi,
        equilibrium.psi_axis,
        equilibrium.psi_lcfs,
        core_grid.rho_tor_norm,
        psi_interpolator,
    )

    # Load species composition
    composition = load_core_species(
        core_profiles_ids.profiles_1d[0],
        split_ion_bundles=split_ion_bundles,
        atomic_data=atomic_data,
    )

    # ----------------------------
    # === Create Plasma object ===
    # ----------------------------
    name = f"IMAS core plasma: time {core_profiles_ids.time[0]}, uri {entry.uri}."
    plasma = Plasma(parent=parent, name=name)
    radius_inner, radius_outer = equilibrium.r_range
    zmin, zmax = equilibrium.z_range
    height = zmax - zmin
    plasma.geometry = Subtract(Cylinder(radius_outer, height), Cylinder(radius_inner, height))
    plasma.geometry_transform = translate(0, 0, zmin)

    # Add magnetic field
    plasma.b_field = VectorAxisymmetricMapper(b_field)

    # Add atomic data
    if atomic_data is not None:
        plasma.atomic_data = atomic_data

    # ------------------------------------
    # === Define Electron Distribution ===
    # ------------------------------------
    if composition.electron.density_thermal is not None:
        composition.electron.density = composition.electron.density_thermal

    if composition.electron is None:
        raise RuntimeError("Electron profile is missing in the core_profiles IDS.")
    if composition.electron.density is None:
        raise RuntimeError("Electron density profile is missing in the core_profiles IDS.")
    if composition.electron.temperature is None:
        raise RuntimeError("Electron temperature profile is missing in the core_profiles IDS.")

    interp = get_core_interpolators(psi_norm, composition.electron, equilibrium, return3d=True)

    plasma.electron_distribution = Maxwellian(
        interp.density,
        interp.temperature,
        interp.velocity or ZERO_VELOCITY,
        electron_mass,
    )

    # -----------------------------------------------
    # === Define Species Distribution/Composition ===
    # -----------------------------------------------

    # === Ion/Neutral Species ===
    for profile in composition.ion + composition.neutral:
        if profile.species is None:
            print(f"Warning! Skipping species with missing element or charge: {profile}")
            continue
        if profile.species.element is None:
            print(f"Warning! Skipping species with missing element: {profile}")
            continue
        if profile.density_thermal is not None:
            profile.density = profile.density_thermal
        if profile.density is None:
            print(f"Warning! Skipping {profile.species}: density profile is missing.")
            continue
        if profile.temperature is None:
            print(f"Warning! Skipping {profile.species}: temperature profile is missing.")
            continue

        element = profile.species.element
        charge = profile.species.z_min

        try:
            species = plasma.composition.get(element, int(charge))
            print(f"Warning! Skipping {species}: already defined")
            continue
        except ValueError:
            pass

        interp = get_core_interpolators(psi_norm, profile, equilibrium, return3d=True)

        distribution = Maxwellian(
            interp.density,
            interp.temperature,
            interp.velocity or ZERO_VELOCITY,
            element.atomic_weight * atomic_mass,
        )

        plasma.composition.add(Species(element, int(charge), distribution))

    # === Ion Bundles ===
    # Ion bundles are split into their constituent charge states at the composition level.
    warn_unsupported_species(composition, "ion_bundle")

    # === Molecular Species ===
    # TODO: properly support molecular species.
    # For now, just issue a warning if any molecule or molecular_bundle species are present in the composition.
    warn_unsupported_species(composition, "molecule")
    warn_unsupported_species(composition, "molecular_bundle")

    return plasma


def get_core_interpolators(
    psi_norm: NDArray[np.float64],
    profile: ProfileData,
    equilibrium: EFITEquilibrium,
    return3d: bool = False,
) -> ProfileInterporater:
    """Create interpolators for the core profiles.

    Parameters
    ----------
    psi_norm
        Normalized poloidal flux values.
    profile
        Instance of the `ProfileData` dataclass containing core plasma profiles.
    equilibrium
        `EFITEquilibrium` object used to map core profiles.
    return3d
        If True, return the 3D interpolators assuming rotational symmetry, by default False.

    Returns
    -------
    `.ProfileInterporater`
        Instance of the `ProfileInterporater` dataclass containing the interpolators for density and temperature.

    Raises
    ------
    ValueError
        If the ``equilibrium`` argument is not an `~cherab.tools.equilibrium.efit.EFITEquilibrium`
        instance.
    """
    if not isinstance(equilibrium, EFITEquilibrium):
        raise ValueError("Argument equilibrium must be a EFITEquilibrium instance.")

    psi_norm, index = np.unique(psi_norm, return_index=True)

    interpolators = ProfileInterporater()

    for prof_key in profile.__dataclass_fields__:
        if prof_key in {"species", "velocity"}:
            continue
        data_1d = getattr(profile, prof_key, None)
        if isinstance(data_1d, np.ndarray) and data_1d.size > 0:
            extrapolation_range = max(0, psi_norm[0], 1.0 - psi_norm[-1])
            func = Interpolator1DArray(
                psi_norm, data_1d[index], "cubic", "nearest", extrapolation_range
            )
            setattr(
                interpolators,
                prof_key,
                equilibrium.map3d(func) if return3d else equilibrium.map2d(func),
            )

    if profile.velocity is not None:
        pass  # TODO: handle velocity profile

    return interpolators


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
