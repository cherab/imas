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

import numpy as np
from scipy.constants import atomic_mass, electron_mass

import imas

from raysect.core.math.function.float import Function2D, Constant2D, Constant3D
from raysect.core.math.function.vector3d import Function2D as VectorFunction2D
from raysect.core.math.function.vector3d import Constant2D as ConstantVector2D
from raysect.core.math.function.vector3d import Constant3D as ConstantVector3D
from raysect.core.math import translate, Vector3D
from raysect.primitive import Cylinder, Subtract

from cherab.core import Plasma, Species, Maxwellian
from cherab.core.math import AxisymmetricMapper, VectorAxisymmetricMapper
from cherab.core.utility import RecursiveDict
from cherab.tools.equilibrium.efit import PoloidalFieldVector, FluxSurfaceNormal

from cherab.imas.math import UnitVector2D
from cherab.imas.ids.edge_profiles import load_edge_species
from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.common.ggd import load_grid
from cherab.imas.plasma.equilibrium import load_equilibrium, load_magnetic_field
from cherab.imas.plasma.utility import warn_unsupported_species, get_subset_name_index


def load_edge_plasma(shot, run, user, database, backend=imas.imasdef.MDSPLUS_BACKEND, time=0, occurrence=0,
                     grid_ggd=None, grid_subset_id=5, b_field=None, time_threshold=np.inf, parent=None):
    """
    Loads edge profiles and creates a Plasma object.

    :param shot: IMAS shot number.
    :param run: IMAS run number for a given shot.
    :param user: IMAS username.
    :param database: IMAS database name.
    :param backend: IMAS database backend. Default is imas.imasdef.MDSPLUS_BACKEND.
    :param time: Time moment. Default is 0.
    :param occurrence: Instance index of the 'edge_profiles' IDS. Default is 0.
    :param grid_ggd: An alternative grid_ggd structure with the grid description. Default is None.
    :param grid_subset_id: Identifier of the grid subset. Either index or name. Default is 5 ("Cells").
    :param b_field: An alternative 2D interpolator of the magnetic field vector (Br, Btor, Bz).
        Default is None: the magnetic field will be loaded from the 'equilibrium' IDS.
    :param time_threshold: Sets the maximum allowable difference between the specified time and the nearest
        available time. Default is np.inf.
    :param parent: The parent node in the Raysect scene-graph. Default is None.
    """

    entry = imas.DBEntry(backend, database, shot, run, user)
    edge_profiles_ids = get_ids_time_slice(entry, 'edge_profiles', time=time, occurrence=occurrence, time_threshold=time_threshold)

    if not len(edge_profiles_ids.grid_ggd) and grid_ggd is None:
        raise RuntimeError("The 'grid_ggd' AOS of the edge_profiles IDS is empty and an alternative grid_ggd structure is not provided.")

    if not len(edge_profiles_ids.ggd):
        raise RuntimeError("The 'ggd' AOS of the edge_profiles IDS is empty.")

    grid_ggd = grid_ggd or edge_profiles_ids.grid_ggd[0]
    grid, subsets, subset_id = load_grid(grid_ggd, with_subsets=True)

    grid_subset_name, grid_subset_index = get_subset_name_index(subset_id, grid_subset_id)

    if np.all(subsets[grid_subset_name] != np.arange(grid.num_cell, dtype=int)):
        # To reduce memory usage, create the sub-grid only if needed.
        grid = grid.subset(subsets[grid_subset_name], name=grid_subset_name)

    composition = load_edge_species(edge_profiles_ids.ggd[0], grid_subset_index=grid_subset_index)

    name = 'IMAS edge plasma: shot {}, run {}, time {}.'.format(shot, run, edge_profiles_ids.time[0])
    plasma = Plasma(parent=parent, name=name)

    # Create plasma geometry
    radius_outer = grid.mesh_extent['rmax']
    radius_inner = grid.mesh_extent['rmin']
    height = grid.mesh_extent['zmax'] - grid.mesh_extent['zmin']
    zmin = grid.mesh_extent['zmin']
    plasma.geometry = Subtract(Cylinder(radius_outer, height), Cylinder(radius_inner, height))
    plasma.geometry_transform = translate(0, 0, zmin)

    if b_field is None:
        try:
            b_field = load_magnetic_field(shot, run, user, database, backend=backend, time=time)
        except RuntimeError:
            try:
                b_field = load_equilibrium(shot, run, user, database, backend=backend, time=time).b_field
            except RuntimeError:
                print('Warning! No magnetic field data available in the equilibrium IDS.')

    if b_field is not None:
        plasma.b_field = VectorAxisymmetricMapper(b_field)

    # Add electron species
    electrons = get_edge_interpolators(grid, composition['electron'], b_field, return3d=True)

    if electrons['density'] is None:
        print("Unable to create Edge Plasma: electron density is not available.")
    if electrons['temperature'] is None:
        print("Unable to create Edge Plasma: electron temperature is not available.")

    plasma.electron_distribution = Maxwellian(electrons['density'], electrons['temperature'],
                                              electrons['velocity'], electron_mass)

    warn_unsupported_species(composition, 'molecule')
    warn_unsupported_species(composition, 'molecular_bundle')
    warn_unsupported_species(composition, 'ion_bundle')

    # Add ion and neutral species
    for species_id, profiles in composition['ion'].items():
        d = {first: second for first, second in species_id}
        species_type = d['element']
        charge = int(round(d['z']))

        sp_key = (species_type, charge)
        if sp_key in plasma.composition:
            print("Warning! Skipping {} species. Species with the same (element, charge): {} is already added.".format(d['label'], sp_key))
            continue

        interp = get_edge_interpolators(grid, profiles, b_field, return3d=True)

        if interp['density'] is None:
            print("Warning! Skipping {} species: density is not available.".format(d['label']))
        if interp['temperature'] is None:
            print("Warning! Skipping {} species: temperature is not available.".format(d['label']))

        distribution = Maxwellian(interp['density'], interp['temperature'],
                                  interp['velocity'], species_type.atomic_weight * atomic_mass)

        plasma.composition.add(Species(species_type, charge, distribution))

    return plasma


def get_edge_interpolators(grid, profiles, b_field=None, return3d=False):
    """
    Create interpolators for the profiles defined on a grid.

    :param grid: GGD-compatible grid object.
    :param profiles: A dictionary with edge plasma profiles.
    :param b_field: 2D interpolator of the magnetic field vector (Br, Btor, Bz). Default is None.
    :param return3d: If True, convert 2D interpolators to 3D assuming rotational symmetry.
        Default is False.

    :returns: A dictionary with edge interpolators.
    """

    interpolators = RecursiveDict()

    for prof_key, profile in profiles.items():
        if 'velocity' in prof_key:
            continue
        if profile is not None:
            func = grid.interpolator(profile)
            if isinstance(func, Function2D) and return3d:
                func = AxisymmetricMapper(func)
            interpolators[prof_key] = func
        else:
            interpolators[prof_key] = None

    vector_func = get_velocity_interpolators(grid, profiles, b_field)
    if isinstance(vector_func, VectorFunction2D) and return3d:
        vector_func = VectorAxisymmetricMapper(vector_func)
    interpolators['velocity'] = vector_func

    return interpolators.freeze()


def get_velocity_interpolators(grid, profiles, b_field=None):

    # Note: np.all(None == 0) returns False
    vrad = None if np.all(profiles['velocity_radial'] == 0) else profiles['velocity_radial']
    vpol = None if np.all(profiles['velocity_poloidal'] == 0) else profiles['velocity_poloidal']
    vpar = None if np.all(profiles['velocity_parallel'] == 0) else profiles['velocity_parallel']
    vtor = None if np.all(profiles['velocity_toroidal'] == 0) else profiles['velocity_toroidal']
    vr = None if np.all(profiles['velocity_r'] == 0) else profiles['velocity_r']
    vz = None if np.all(profiles['velocity_z'] == 0) else profiles['velocity_z']

    if not b_field:
        return get_cylindrical_velocity_interpolators(grid, vr, vz, vtor)

    if vrad is None and vr is not None and vz is not None:
        if vtor is None and vpar is not None:
            _, vtor = _get_components_from_vpar(grid, vpar, b_field)
        return get_cylindrical_velocity_interpolators(grid, vr, vz, vtor)

    if vpar is None:
        return get_poloidal_velocity_interpolators(grid, vpol, vrad, vtor, b_field)

    return get_parallel_velocity_interpolators(grid, vpar, vrad, b_field)


def get_cylindrical_velocity_interpolators(grid, vr, vz, vtor):

    if vr is None and vz is None and vtor is None:
        if grid.dimension == 2:
            return ConstantVector2D(Vector3D(0, 1.e-16, 0))  # avoid zero-length vectors for blending

        return ConstantVector3D(Vector3D(0, 1.e-16, 0))  # avoid zero-length vectors for blending

    if vr is None:
        vr = np.zeros(grid.num_cell, dtype=np.float64)
    if vz is None:
        vz = np.zeros(grid.num_cell, dtype=np.float64)
    if vtor is None:
        vtor = np.zeros(grid.num_cell, dtype=np.float64)

    return grid.vector_interpolator(np.array([vr, vtor, vz]))


def get_parallel_velocity_interpolators(grid, vpar, vrad, b_field):

    if vpar is None and vrad is None:
        if grid.dimension == 2:  # 2D case
            return ConstantVector2D(Vector3D(0, 1.e-16, 0))  # avoid zero-length vectors for blending

        return ConstantVector3D(Vector3D(0, 1.e-16, 0))  # avoid zero-length vectors for blending

    const_func = Constant2D if grid.dimension == 2 else Constant3D

    vpar_i = const_func(0) if vpar is None else grid.interpolator(vpar)
    vrad_i = const_func(0) if vrad is None else grid.interpolator(vrad)

    parallel_vector = UnitVector2D(b_field)
    surface_normal = FluxSurfaceNormal(b_field)

    if grid.dimension == 3:  # 3D case
        parallel_vector = VectorAxisymmetricMapper(parallel_vector)
        surface_normal = VectorAxisymmetricMapper(surface_normal)

    return vpar_i * parallel_vector + vrad_i * surface_normal


def get_poloidal_velocity_interpolators(grid, vpol, vrad, vtor, b_field):

    if vpol is None and vrad is None and vtor is None:
        if grid.dimension == 2:  # 2D case
            return ConstantVector2D(Vector3D(0, 1.e-16, 0))  # avoid zero-length vectors for blending

        return ConstantVector3D(Vector3D(0, 1.e-16, 0))  # avoid zero-length vectors for blending

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
            if grid.coordinate_system == 'cartesian':
                r = np.sqrt(cell_centre[0]**2 + cell_centre[1]**2)
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
