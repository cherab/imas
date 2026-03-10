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
"""Module for loading edge plasma profiles from the edge_profiles IDS."""

import numpy as np
from numpy.typing import NDArray
from raysect.core.math import Vector3D, translate
from raysect.core.math.function.float import Constant2D, Constant3D, Function2D
from raysect.core.math.function.vector3d import Constant2D as ConstantVector2D
from raysect.core.math.function.vector3d import Constant3D as ConstantVector3D
from raysect.core.math.function.vector3d import Function2D as VectorFunction2D
from raysect.core.scenegraph._nodebase import _NodeBase
from raysect.primitive import Cylinder, Subtract
from scipy.constants import atomic_mass, electron_mass

from cherab.core import AtomicData, Maxwellian, Plasma, Species
from cherab.core.math import AxisymmetricMapper, VectorAxisymmetricMapper
from cherab.tools.equilibrium.efit import FluxSurfaceNormal, PoloidalFieldVector
from imas import DBEntry
from imas.ids_structure import IDSStructure

from ..ggd.base_mesh import GGDGrid
from ..ids.common import get_ids_time_slice
from ..ids.common.ggd import load_grid
from ..ids.common.species import ProfileData, VelocityData
from ..ids.edge_profiles import load_edge_species
from ..math import UnitVector2D
from ._model import solve_coronal_equilibrium
from .equilibrium import load_equilibrium, load_magnetic_field
from .utility import (
    ZERO_VELOCITY,
    ProfileInterporater,
    get_subset_name_index,
    warn_unsupported_species,
)

__all__ = ["load_edge_plasma"]


def load_edge_plasma(
    *args,
    time: float = 0,
    occurrence: int = 0,
    grid_ggd: IDSStructure | None = None,
    grid_subset_id: int | str = 5,
    b_field: VectorFunction2D | None = None,
    time_threshold: float = np.inf,
    atomic_data: AtomicData | None = None,
    parent: _NodeBase | None = None,
    **kwargs,
) -> Plasma:
    """Load edge profiles and Create a `~cherab.core.plasma.node.Plasma` object.

    Prefer ``density_thermal`` over ``density`` profile.

    The distribution of each species is defined with `~cherab.core.distribution.Maxwellian` using
    its density, temperature, and bulk velocity profiles, which are mapped to GGD grid coordinates.

    The plasma geometry is defined as a cylindrical annulus between the inner and outer limit of
    the grid, and its height is defined as the difference between the maximum and minimum
    z-coordinate of the grid.

    The ion bundle species are split into their constituent charge states using `.solve_coronal_equilibrium`.

    Parameters
    ----------
    *args
        Arguments passed to the `~imas.db_entry.DBEntry` constructor.
    time
        Time for the edge plasma, by default 0.
    occurrence
        Occurrence index of the ``edge_profiles`` IDS, by default 0.
    grid_ggd
        Alternative ``grid_ggd`` structure with the grid description, by default None.
    grid_subset_id
        Identifier of the grid subset. Either index or name, by default 5 (``"Cells"``).
    b_field
        Alternative 2D interpolator of the magnetic field vector (Br, Bphi, Bz).
        Default is None. The magnetic field will be loaded from the ``equilibrium`` IDS.
    time_threshold
        Maximum allowed difference between the requested time and the nearest
        available time, by default `numpy.inf`.
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
        Plasma object with edge profiles.

    Raises
    ------
    RuntimeError
        If the ``grid_ggd`` or ``ggd`` AOS of the edge_profiles IDS is empty.
    """
    # -----------------------------------------
    # === Load edge profiles data from IMAS ===
    # -----------------------------------------
    # Load required data from the edge_profiles IDS and form the edge grid and species composition
    # data structures.
    with DBEntry(*args, **kwargs) as entry:
        edge_profiles_ids = get_ids_time_slice(
            entry, "edge_profiles", time=time, occurrence=occurrence, time_threshold=time_threshold
        )

    if not len(edge_profiles_ids.grid_ggd) and grid_ggd is None:
        raise RuntimeError(
            "The 'grid_ggd' AOS of the edge_profiles IDS is empty "
            + "and an alternative grid_ggd structure is not provided."
        )

    if not len(edge_profiles_ids.ggd):
        raise RuntimeError("The 'ggd' AOS of the edge_profiles IDS is empty.")

    # Load magnetic field data. If not provided, try to load from the equilibrium IDS.
    if b_field is None:
        try:
            b_field = load_magnetic_field(*args, time=time, occurrence=occurrence, **kwargs)
        except RuntimeError:
            try:
                b_field = load_equilibrium(
                    *args, time=time, occurrence=occurrence, **kwargs
                ).b_field
            except RuntimeError:
                print("Warning! No magnetic field data available in the equilibrium IDS.")

    # Create edge grid
    grid_ggd = grid_ggd or edge_profiles_ids.grid_ggd[0]
    grid, subsets, subset_id = load_grid(grid_ggd, with_subsets=True)

    grid_subset_name, grid_subset_index = get_subset_name_index(subset_id, grid_subset_id)

    if np.all(subsets[grid_subset_name] != np.arange(grid.num_cell, dtype=int)):
        # To reduce memory usage, create the sub-grid only if needed.
        grid = grid.subset(subsets[grid_subset_name], name=grid_subset_name)

    # Load species composition
    composition = load_edge_species(edge_profiles_ids.ggd[0], grid_subset_index=grid_subset_index)

    # ------------------------------
    # === Create plasma geometry ===
    # ------------------------------
    name = f"IMAS edge plasma: time {edge_profiles_ids.time[0]}, uri {entry.uri}."
    plasma = Plasma(parent=parent, name=name)

    # Create plasma geometry
    radius_outer = grid.mesh_extent["rmax"]
    radius_inner = grid.mesh_extent["rmin"]
    height = grid.mesh_extent["zmax"] - grid.mesh_extent["zmin"]
    zmin = grid.mesh_extent["zmin"]
    plasma.geometry = Subtract(Cylinder(radius_outer, height), Cylinder(radius_inner, height))
    plasma.geometry_transform = translate(0, 0, zmin)

    # Add magnetic field
    if b_field is not None:
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

    interp = get_edge_interpolators(grid, composition.electron, b_field, return3d=True)

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

        interp = get_edge_interpolators(grid, profile, b_field, return3d=True)

        distribution = Maxwellian(
            interp.density,
            interp.temperature,
            interp.velocity or ZERO_VELOCITY,
            element.atomic_weight * atomic_mass,
        )

        plasma.composition.add(Species(element, int(charge), distribution))

    # === Ion Bundles ===
    for profile in composition.ion_bundle:
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

        element = profile.species.element

        # Split the ion bundle into its constituent charge states using the coronal equilibrium solver
        z_min, z_max = profile.species.z_min, profile.species.z_max
        densities_per_charge = solve_coronal_equilibrium(
            element,
            profile.density,
            composition.electron.density,
            composition.electron.temperature,
            atomic_data=atomic_data,
            z_min=z_min,
            z_max=z_max,
        )
        charge_states = np.arange(int(z_min), int(z_max) + 1, dtype=int)
        for i_charge, charge in enumerate(charge_states):
            # Check if any of the constituent charge states are already defined
            try:
                species = plasma.composition.get(element, charge)
                print(f"Warning! Skipping {species}: already defined")
                continue
            except ValueError:
                pass

            profile.density = densities_per_charge[i_charge, :]
            interp = get_edge_interpolators(grid, profile, b_field, return3d=True)

            distribution = Maxwellian(
                interp.density,
                interp.temperature,
                interp.velocity or ZERO_VELOCITY,
                element.atomic_weight * atomic_mass,
            )

            plasma.composition.add(Species(element, int(charge), distribution))

    # === Molecular Species ===
    # TODO: properly support molecular species.
    # For now, just issue a warning if any molecule or molecular_bundle species are present in the composition.
    warn_unsupported_species(composition, "molecule")
    warn_unsupported_species(composition, "molecular_bundle")

    return plasma


def get_edge_interpolators(
    grid: GGDGrid,
    profile: ProfileData,
    b_field: VectorFunction2D | None = None,
    return3d: bool = False,
) -> ProfileInterporater:
    """Create interpolators for the profiles defined on a grid.

    Parameters
    ----------
    grid
        GGD-compatible grid object.
    profile
        Instance of the `ProfileData` dataclass containing the profiles to be interpolated.
    b_field
        2D interpolator of the magnetic field vector (Br, Btor, Bz), by default None.
    return3d
        If True, return the 3D functions for 2D interpolators assuming
        rotational symmetry, by default False.

    Returns
    -------
    ProfileInterporater
        Instance of the `ProfileInterporater` dataclass containing the interpolators for density and temperature.
    """
    interpolators = ProfileInterporater()

    for prof_key in profile.__dataclass_fields__:
        if prof_key in {"species", "velocity"}:
            continue
        data = getattr(profile, prof_key, None)
        if data is not None:
            func = grid.interpolator(data)
            if isinstance(func, Function2D) and return3d:
                func = AxisymmetricMapper(func)
            setattr(interpolators, prof_key, func)

    # Create velocity interpolator
    if profile.velocity is not None:
        vector_func = _get_velocity_interpolators(grid, profile.velocity, b_field)
        if isinstance(vector_func, VectorFunction2D) and return3d:
            vector_func = VectorAxisymmetricMapper(vector_func)
        interpolators.velocity = vector_func

    return interpolators


def _get_velocity_interpolators(
    grid: GGDGrid, v: VelocityData, b_field: VectorFunction2D | None = None
):
    if b_field is None:
        return _get_cylindrical_velocity_interpolators(grid, v.r, v.z, v.phi)

    if v.radial is None and v.r is not None and v.z is not None:
        if v.phi is None and v.parallel is not None:
            _, v.phi = _get_components_from_vpar(grid, v.parallel, b_field)
        return _get_cylindrical_velocity_interpolators(grid, v.r, v.z, v.phi)

    if v.parallel is None:
        return _get_poloidal_velocity_interpolators(grid, v.poloidal, v.radial, v.phi, b_field)

    return _get_parallel_velocity_interpolators(grid, v.parallel, v.radial, b_field)


def _get_cylindrical_velocity_interpolators(
    grid: GGDGrid,
    vr: NDArray[np.float64] | None,
    vz: NDArray[np.float64] | None,
    vtor: NDArray[np.float64] | None,
):
    if vr is None and vz is None and vtor is None:
        if grid.dimension == 2:
            return ConstantVector2D(
                Vector3D(0, 1.0e-16, 0)
            )  # avoid zero-length vectors for blending

        return ConstantVector3D(Vector3D(0, 1.0e-16, 0))  # avoid zero-length vectors for blending

    if vr is None:
        vr = np.zeros(grid.num_cell, dtype=np.float64)
    if vz is None:
        vz = np.zeros(grid.num_cell, dtype=np.float64)
    if vtor is None:
        vtor = np.zeros(grid.num_cell, dtype=np.float64)

    return grid.vector_interpolator(np.array([vr, vtor, vz]))


def _get_parallel_velocity_interpolators(
    grid: GGDGrid,
    vpar: NDArray[np.float64] | None,
    vrad: NDArray[np.float64] | None,
    b_field: VectorFunction2D,
):
    if vpar is None and vrad is None:
        if grid.dimension == 2:  # 2D case
            return ConstantVector2D(
                Vector3D(0, 1.0e-16, 0)
            )  # avoid zero-length vectors for blending

        return ConstantVector3D(Vector3D(0, 1.0e-16, 0))  # avoid zero-length vectors for blending

    const_func = Constant2D if grid.dimension == 2 else Constant3D

    vpar_i = const_func(0) if vpar is None else grid.interpolator(vpar)
    vrad_i = const_func(0) if vrad is None else grid.interpolator(vrad)

    parallel_vector = UnitVector2D(b_field)
    surface_normal = FluxSurfaceNormal(b_field)

    if grid.dimension == 3:  # 3D case
        parallel_vector = VectorAxisymmetricMapper(parallel_vector)
        surface_normal = VectorAxisymmetricMapper(surface_normal)

    return vpar_i * parallel_vector + vrad_i * surface_normal


def _get_poloidal_velocity_interpolators(
    grid: GGDGrid,
    vpol: NDArray[np.float64] | None,
    vrad: NDArray[np.float64] | None,
    vtor: NDArray[np.float64] | None,
    b_field: VectorFunction2D,
):
    if vpol is None and vrad is None and vtor is None:
        if grid.dimension == 2:  # 2D case
            return ConstantVector2D(
                Vector3D(0, 1.0e-16, 0)
            )  # avoid zero-length vectors for blending

        return ConstantVector3D(Vector3D(0, 1.0e-16, 0))  # avoid zero-length vectors for blending

    const_func = Constant2D if grid.dimension == 2 else Constant3D

    vpol_i = const_func(0) if vpol is None else grid.interpolator(vpol)
    vrad_i = const_func(0) if vrad is None else grid.interpolator(vrad)
    vtor_i = const_func(0) if vtor is None else grid.interpolator(vtor)

    poloidal_vector = PoloidalFieldVector(b_field)
    surface_normal = FluxSurfaceNormal(b_field)
    toroidal_vector = ConstantVector2D(Vector3D(0, 1, 0))

    if grid.dimension == 3:  # 3D case
        poloidal_vector = VectorAxisymmetricMapper(poloidal_vector)
        surface_normal = VectorAxisymmetricMapper(surface_normal)
        toroidal_vector = VectorAxisymmetricMapper(toroidal_vector)

    return vpol_i * poloidal_vector + vrad_i * surface_normal + vtor_i * toroidal_vector


def _get_components_from_vpar(grid: GGDGrid, vpar: NDArray[np.float64], b_field: VectorFunction2D):
    vpol = np.zeros(grid.num_cell, dtype=np.float64)
    vtor = np.zeros(grid.num_cell, dtype=np.float64)

    for i, cell_centre in enumerate(grid.cell_centre):
        if grid.dimension == 2:  # 2D case
            r, z = cell_centre
        else:  # 3D case
            if grid.coordinate_system == "cartesian":
                r = np.sqrt(cell_centre[0] ** 2 + cell_centre[1] ** 2)
                z = cell_centre[2]
            else:
                r, _, z = cell_centre
        try:
            b_field = b_field(r, z)
            vpol[i] = np.sqrt(b_field.x**2 + b_field.z**2) * (vpar[i] / b_field.length)
            vtor[i] = vpar[i] * b_field.y / b_field.length
        except ValueError:  # Outside equilibrium grid
            continue

    return vpol, vtor
