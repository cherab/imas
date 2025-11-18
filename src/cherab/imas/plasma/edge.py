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
from raysect.core.math import Vector3D, translate
from raysect.core.math.function.float import Constant2D, Constant3D, Function2D, Function3D
from raysect.core.math.function.vector3d import Constant2D as ConstantVector2D
from raysect.core.math.function.vector3d import Constant3D as ConstantVector3D
from raysect.core.math.function.vector3d import Function2D as VectorFunction2D
from raysect.core.scenegraph._nodebase import _NodeBase
from raysect.primitive import Cylinder, Subtract
from scipy.constants import atomic_mass, electron_mass

from cherab.core import Maxwellian, Plasma, Species
from cherab.core.math import AxisymmetricMapper, VectorAxisymmetricMapper
from cherab.core.utility import RecursiveDict
from cherab.tools.equilibrium.efit import FluxSurfaceNormal, PoloidalFieldVector
from imas import DBEntry
from imas.ids_structure import IDSStructure

from ..ggd.base_mesh import GGDGrid
from ..ids.common import get_ids_time_slice
from ..ids.common.ggd import load_grid
from ..ids.edge_profiles import load_edge_species
from ..math import UnitVector2D
from .equilibrium import load_equilibrium, load_magnetic_field
from .utility import get_subset_name_index, warn_unsupported_species

__all__ = ["load_edge_plasma"]


def load_edge_plasma(
    *args,
    time: float = 0,
    occurrence: int = 0,
    grid_ggd: IDSStructure | None = None,
    grid_subset_id: int | str = 5,
    b_field: VectorFunction2D | None = None,
    time_threshold: float = np.inf,
    parent: _NodeBase | None = None,
    **kwargs,
) -> Plasma:
    """Load edge profiles and Create a `~cherab.core.plasma.node.Plasma` object.

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

    grid_ggd = grid_ggd or edge_profiles_ids.grid_ggd[0]
    grid, subsets, subset_id = load_grid(grid_ggd, with_subsets=True)

    grid_subset_name, grid_subset_index = get_subset_name_index(subset_id, grid_subset_id)

    if np.all(subsets[grid_subset_name] != np.arange(grid.num_cell, dtype=int)):
        # To reduce memory usage, create the sub-grid only if needed.
        grid = grid.subset(subsets[grid_subset_name], name=grid_subset_name)

    composition = load_edge_species(edge_profiles_ids.ggd[0], grid_subset_index=grid_subset_index)

    name = f"IMAS edge plasma: time {edge_profiles_ids.time[0]}, uri {entry.uri}."
    plasma = Plasma(parent=parent, name=name)

    # Create plasma geometry
    radius_outer = grid.mesh_extent["rmax"]
    radius_inner = grid.mesh_extent["rmin"]
    height = grid.mesh_extent["zmax"] - grid.mesh_extent["zmin"]
    zmin = grid.mesh_extent["zmin"]
    plasma.geometry = Subtract(Cylinder(radius_outer, height), Cylinder(radius_inner, height))
    plasma.geometry_transform = translate(0, 0, zmin)

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

    if b_field is not None:
        plasma.b_field = VectorAxisymmetricMapper(b_field)

    # Add electron species
    electrons = get_edge_interpolators(grid, composition["electron"], b_field, return3d=True)

    if electrons["density"] is None:
        print("Unable to create Edge Plasma: electron density is not available.")
    if electrons["temperature"] is None:
        print("Unable to create Edge Plasma: electron temperature is not available.")

    plasma.electron_distribution = Maxwellian(
        electrons["density"], electrons["temperature"], electrons["velocity"], electron_mass
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

        interp = get_edge_interpolators(grid, profiles, b_field, return3d=True)

        if interp["density"] is None:
            print(f"Warning! Skipping {d['name']} species: density is not available.")
        if interp["temperature"] is None:
            print(f"Warning! Skipping {d['name']} species: temperature is not available.")

        distribution = Maxwellian(
            interp["density"],
            interp["temperature"],
            interp["velocity"],
            species_type.atomic_weight * atomic_mass,
        )

        plasma.composition.add(Species(species_type, charge, distribution))

    return plasma


def get_edge_interpolators(
    grid: GGDGrid,
    profiles: dict[str, np.ndarray | None],
    b_field: VectorFunction2D | None = None,
    return3d: bool = False,
) -> dict[str, Function3D | Function2D | VectorFunction2D | VectorFunction2D | None]:
    """Create interpolators for the profiles defined on a grid.

    Parameters
    ----------
    grid
        GGD-compatible grid object.
    profiles
        Dictionary with edge plasma profiles.
    b_field
        2D interpolator of the magnetic field vector (Br, Btor, Bz), by default None.
    return3d
        If True, return the 3D functions for 2D interpolators assuming
        rotational symmetry, by default False.

    Returns
    -------
    dict[str, Function3D | Function2D | None]
        Dictionary with edge interpolators.
    """
    interpolators = RecursiveDict()

    for prof_key, profile in profiles.items():
        if "velocity" in prof_key:
            continue
        if profile is not None:
            func = grid.interpolator(profile)
            if isinstance(func, Function2D) and return3d:
                func = AxisymmetricMapper(func)
            interpolators[prof_key] = func
        else:
            interpolators[prof_key] = None

    vector_func = _get_velocity_interpolators(grid, profiles, b_field)
    if isinstance(vector_func, VectorFunction2D) and return3d:
        vector_func = VectorAxisymmetricMapper(vector_func)
    interpolators["velocity"] = vector_func

    return interpolators.freeze()


def _get_velocity_interpolators(grid, profiles, b_field=None):
    # Note: np.all(None == 0) returns False
    vrad = None if np.all(profiles["velocity_radial"] == 0) else profiles["velocity_radial"]
    vpol = None if np.all(profiles["velocity_poloidal"] == 0) else profiles["velocity_poloidal"]
    vpar = None if np.all(profiles["velocity_parallel"] == 0) else profiles["velocity_parallel"]
    vtor = None if np.all(profiles["velocity_phi"] == 0) else profiles["velocity_phi"]
    vr = None if np.all(profiles["velocity_r"] == 0) else profiles["velocity_r"]
    vz = None if np.all(profiles["velocity_z"] == 0) else profiles["velocity_z"]

    if not b_field:
        return _get_cylindrical_velocity_interpolators(grid, vr, vz, vtor)

    if vrad is None and vr is not None and vz is not None:
        if vtor is None and vpar is not None:
            _, vtor = _get_components_from_vpar(grid, vpar, b_field)
        return _get_cylindrical_velocity_interpolators(grid, vr, vz, vtor)

    if vpar is None:
        return _get_poloidal_velocity_interpolators(grid, vpol, vrad, vtor, b_field)

    return _get_parallel_velocity_interpolators(grid, vpar, vrad, b_field)


def _get_cylindrical_velocity_interpolators(grid, vr, vz, vtor):
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


def _get_parallel_velocity_interpolators(grid, vpar, vrad, b_field):
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


def _get_poloidal_velocity_interpolators(grid, vpol, vrad, vtor, b_field):
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


def _get_components_from_vpar(grid, vpar, b_field):
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
