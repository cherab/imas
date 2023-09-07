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

from raysect.core.math.function.float import Function2D, Function3D, Blend2D, Blend3D
from raysect.core.math.function.vector3d import Function2D as VectorFunction2D
from raysect.core.math.function.vector3d import Function3D as VectorFunction3D
from raysect.core.math.function.vector3d import Blend2D as VectorBlend2D
from raysect.core.math.function.vector3d import Blend3D as VectorBlend3D
from raysect.core.math.function.vector3d import Constant3D as ConstantVector3D
from raysect.core.math import translate, Vector3D
from raysect.primitive import Cylinder, Subtract

from cherab.core import Plasma, Species, Maxwellian
from cherab.core.math import AxisymmetricMapper, VectorAxisymmetricMapper

from cherab.imas.ids.core_profiles import load_core_species, load_core_grid
from cherab.imas.ids.edge_profiles import load_edge_species
from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.common.ggd import load_grid
from cherab.imas.plasma.equilibrium import load_equilibrium, load_magnetic_field
from cherab.imas.plasma.core import load_core_plasma, get_core_interpolators, get_psi_norm
from cherab.imas.plasma.edge import load_edge_plasma, get_edge_interpolators
from cherab.imas.plasma.utility import warn_unsupported_species, get_subset_name_index


def load_plasma(shot, run, user, database, backend=imas.imasdef.MDSPLUS_BACKEND, time=0, occurrence_core=0,
                shot_edge=None, run_edge=None, user_edge=None, database_edge=None, backend_edge=None,
                time_edge=None, occurrence_edge=0, grid_ggd=None, grid_subset_id=5,
                equilibrium=None, b_field=None, psi_interpolator=None, mask=None,
                time_threshold=np.inf, parent=None):
    """
    Loads core and edge profiles and creates a Plasma object.
    If edge_profiles IDS is empty returns core plasma only.
    If core_profiles IDS is empty returns edge plasma only.

    :param shot: IMAS shot number.
    :param run: IMAS run number for a given shot.
    :param user: IMAS username.
    :param database: IMAS database name.
    :param backend: IMAS database backend. Default is imas.imasdef.MDSPLUS_BACKEND.
    :param time: Time moment. Default is 0.
    :param occurrence: Instance index of the 'core_profiles' IDS. Default is 0.
    :param shot_edge: IMAS shot number for the edge plasma if different from the 'shot'. Default is None.
    :param run_edge: IMAS run number for a given shot for the edge plasma if different from the 'run'. Default is None.
    :param user_edge: IMAS username for the edge plasma if different from the 'user'. Default is None.
    :param database_edge: IMAS database name for the edge plasma if different from the 'database'. Default is None,
    :param backend_edge: IMAS database backend for the edge plasma if different from the 'backend'.
        Default is None.
    :param time_edge: Time moment for the edge plasma if different from the 'time'. Default is None.
    :param occurrence_edge: Instance index of the 'edge_profiles' IDS. Default is 0.
    :param grid_ggd: An alternative grid_ggd structure with the grid description. Default is None.
    :param grid_subset_id: Identifier of the grid subset. Either index or name. Default is 5 ("Cells").
    :param equilibrium: Alternative EFITEquilibrium object used to map core profiles.
        Default is None: equilibrium is read from the same shot, run, time, etc. as the core profiles.
        This parameter is ignored if core plasma is not available.
    :param b_field: An alternative 2D interpolator of the magnetic field vector (Br, Btor, Bz).
        Default is None: the magnetic field will be loaded from the 'equilibrium' IDS.
    :param psi_interpolator: Alternative psi_norm(rho_tor_norm) interpolator.
        Used only if 'psi' is missing in the core grid. Default is None: obtained from the equilibrium IDS.
    :param mask: The 2D or 3D mask function used for blending: (1 - mask) * f_edge + mask * f_core.
        Default is None: use equilibrium.inside_lcfs as a mask function.
    :param time_threshold: Sets the maximum allowable difference between the specified time and the nearest
        available time. Default is np.inf.
    :param parent: The parent node in the Raysect scene-graph. Default is None.
    """

    shot_edge = shot_edge or shot
    run_edge = run_edge or run
    user_edge = user_edge or user
    database_edge = database_edge or database
    backend_edge = backend_edge or backend
    if time_edge is None:
        time_edge = time

    entry = imas.DBEntry(backend, database, shot, run, user)
    try:
        core_profiles_ids = get_ids_time_slice(entry, 'core_profiles', time=time,
                                               occurrence=occurrence_core, time_threshold=time_threshold)
    except RuntimeError:
        return load_edge_plasma(shot_edge, run_edge, user_edge, database_edge,
                                backend=backend_edge, time=time_edge, occurrence=occurrence_edge,
                                grid_ggd=grid_ggd, grid_subset_id=grid_subset_id, b_field=b_field,
                                time_threshold=time_threshold, parent=parent)

    entry = imas.DBEntry(backend_edge, database_edge, shot_edge, run_edge, user_edge)
    try:
        edge_profiles_ids = get_ids_time_slice(entry, 'edge_profiles', time=time_edge,
                                               occurrence=occurrence_edge, time_threshold=time_threshold)
    except RuntimeError:
        return load_core_plasma(shot, run, user, database, backend=backend, time=0, occurrence=0,
                                equilibrium=equilibrium, b_field=b_field, psi_interpolator=psi_interpolator,
                                time_threshold=time_threshold, parent=parent)

    if not len(core_profiles_ids.profiles_1d):
        raise RuntimeError('The profiles_1d AOS in core_profiles IDS is empty.')

    if not len(edge_profiles_ids.grid_ggd) and grid_ggd is None:
        raise RuntimeError("The 'grid_ggd' AOS of the edge_profiles IDS is empty and an alternative grid_ggd structure is not provided.")

    if not len(edge_profiles_ids.ggd):
        raise RuntimeError("The 'ggd' AOS of the edge_profiles IDS is empty.")

    # Getting equilibrium and magnetic field
    if equilibrium is None:
        equilibrium, psi_interp = load_equilibrium(shot, run, user, database, backend=backend, time=time, with_psi_interpolator=True)
        psi_interpolator = psi_interpolator or psi_interp

    if b_field is None:
        try:
            b_field = load_magnetic_field(shot, run, user, database, backend=backend, time=time)
        except RuntimeError:
            b_field = equilibrium.b_field

    mask = mask or equilibrium.inside_lcfs

    # Getting core data
    core_grid = load_core_grid(core_profiles_ids.profiles_1d[0].grid)

    composition_core = load_core_species(core_profiles_ids.profiles_1d[0])

    psi_norm = get_psi_norm(core_grid['psi'], equilibrium.psi_axis, equilibrium.psi_lcfs, core_grid['rho_tor_norm'], psi_interpolator)

    # Getting edge data
    grid_ggd = grid_ggd or edge_profiles_ids.grid_ggd[0]
    grid, subsets, subset_id = load_grid(grid_ggd, with_subsets=True)

    grid_subset_name, grid_subset_index = get_subset_name_index(subset_id, grid_subset_id)

    if np.all(subsets[grid_subset_name] != np.arange(len(grid.cells), dtype=int)):
        # To reduce memory usage, create the sub-grid only if needed.
        grid = grid.subset(subsets[grid_subset_name], name=grid_subset_name)

    composition_edge = load_edge_species(edge_profiles_ids.ggd[0], grid_subset_index=grid_subset_index)

    # Creating plasma
    tcore = core_profiles_ids.time[0]
    tedge = edge_profiles_ids.time[0]
    name = 'IMAS core + edge plasma: core/edge shot {}/{}, core/edge run {}/{}, core/edge time {}/{}.'.format(shot, shot_edge,
                                                                                                              run, run_edge,
                                                                                                              tcore, tedge)
    plasma = Plasma(parent=parent, name=name)

    # Create plasma geometry
    radius_outer = grid.mesh_extent['rmax']
    radius_inner = grid.mesh_extent['rmin']
    height = grid.mesh_extent['zmax'] - grid.mesh_extent['zmin']
    zmin = grid.mesh_extent['zmin']
    plasma.geometry = Subtract(Cylinder(radius_outer, height), Cylinder(radius_inner, height))
    plasma.geometry_transform = translate(0, 0, zmin)

    plasma.b_field = VectorAxisymmetricMapper(b_field)

    # Add electron species
    electrons_core = get_core_interpolators(psi_norm, composition_core['electron'], equilibrium, return3d=False)
    if electrons_core['density_thermal'] is not None:
        electrons_core['density'] = electrons_core['density_thermal']

    electrons_edge = get_edge_interpolators(grid, composition_edge['electron'], b_field, return3d=False)

    electrons = blend_core_edge_interpolators(electrons_core, electrons_edge, mask, return3d=True)

    if electrons['density'] is None:
        print("Unable to create Core Plasma: electron density is not available.")
    if electrons['temperature'] is None:
        print("Unable to create Core Plasma: electron temperature is not available.")

    plasma.electron_distribution = Maxwellian(electrons['density'], electrons['temperature'],
                                              electrons['velocity'], electron_mass)

    warn_unsupported_species(composition_core, 'molecule')
    warn_unsupported_species(composition_edge, 'molecule')
    warn_unsupported_species(composition_core, 'molecular_bundle')
    warn_unsupported_species(composition_edge, 'molecular_bundle')
    warn_unsupported_species(composition_core, 'ion_bundle')
    warn_unsupported_species(composition_edge, 'ion_bundle')

    # Add ion and neutral species

    # List core species
    core_species = {}
    for species_id, profiles in composition_core['ion'].items():
        d = {first: second for first, second in species_id}
        sp_key = (d['element'], int(round(d['z'])))
        if sp_key in core_species:
            print("Warning! Skipping {} core species. Species with the same (element, charge): {} is already added.".format(d['label'], sp_key))
            continue
        if profiles['density_thermal'] is not None:
            profiles['density'] = profiles['density_thermal']
        core_species[sp_key] = profiles

    # List edge species
    edge_species = {}
    for species_id, profiles in composition_edge['ion'].items():
        d = {first: second for first, second in species_id}
        sp_key = (d['element'], int(round(d['z'])))
        if sp_key in edge_species:
            print("Warning! Skipping {} edge species. Species with the same (element, charge): {} is already added.".format(d['label'], sp_key))
        edge_species[sp_key] = profiles

    species = {}
    for core_key, core_profiles in core_species.items():
        if core_key in edge_species:
            edge_profiles = edge_species[core_key]
            core_interp = get_core_interpolators(psi_norm, core_profiles, equilibrium, return3d=False)
            edge_interp = get_edge_interpolators(grid, edge_profiles, b_field, return3d=False)
            species[core_key] = blend_core_edge_interpolators(core_interp, edge_interp, mask, return3d=True)
        else:
            species[core_key] = get_core_interpolators(psi_norm, core_profiles, equilibrium, return3d=True)
    for edge_key, edge_profiles in edge_species.items():
        if edge_key not in core_species:
            species[edge_key] = get_edge_interpolators(grid, edge_profiles, b_field, return3d=True)

    for (species_type, charge), interp in species.items():
        if 'velocity' not in interp or interp['velocity'] is None:
            interp['velocity'] = ConstantVector3D(Vector3D(0, 0, 0))

        if interp['density'] is None:
            print("Warning! Skipping {} species: density is not available.".format(d['label']))
        if interp['temperature'] is None:
            print("Warning! Skipping {} species: temperature is not available.".format(d['label']))

        distribution = Maxwellian(interp['density'], interp['temperature'],
                                  interp['velocity'], species_type.atomic_weight * atomic_mass)

        plasma.composition.add(Species(species_type, charge, distribution))

    return plasma


def blend_core_edge_interpolators(core_interpolators, edge_interpolators, mask, return3d=False):
    """
    Blends together interpolators for the core and edge using the modulating mask function.

    :param core_interpolators: A dictionary with 2D or 3D core profiles interpolators.
    :param edge_interpolators: A dictionary with 2D or 3D edge profiles interpolators.
    :param mask: The 2D or 3D mask function used for blending: (1 - mask) * f_edge + mask * f_core.
    :param return3d: If True, return the 3D functions for 2D interpolators assuming
        rotational symmetry. Default is False.

    :returns: A dictionary with blended interpolators.
    """

    interpolators = {}

    for core_key, core_func in core_interpolators.items():
        edge_func = edge_interpolators[core_key] if core_key in edge_interpolators else None
        interpolators[core_key] = blend_core_edge_functions(core_func, edge_func, mask, return3d)

    for edge_key, edge_func in edge_interpolators.items():
        if edge_key not in core_interpolators:
            interpolators[edge_key] = blend_core_edge_functions(None, edge_func, mask, return3d)

    return interpolators


def blend_core_edge_functions(core_func, edge_func, mask, return3d):
    """
    Blends together the core and edge interpolating functions using the modulating mask function.

    :param core_func: A 2D or 3D core interpolator.
    :param edge_func: A 2D or 3D edge interpolator.
    :param mask: The 2D or 3D mask function used for blending: (1 - mask) * f_edge + mask * f_core.
    :param return3d: If True, return the 3D functions for 2D interpolators assuming
        rotational symmetry. Default is False.

    :returns: Blended interpolator.
    """

    if core_func is None and edge_func is None:
        return None

    if core_func is not None and not isinstance(core_func, (Function2D, Function3D, VectorFunction2D, VectorFunction3D)):
        raise ValueError("The core_func must be a 2D or 3D function.")

    if edge_func is not None and not isinstance(edge_func, (Function2D, Function3D, VectorFunction2D, VectorFunction3D)):
        raise ValueError("The edge_func must be a 2D or 3D function.")

    if not isinstance(mask, (Function2D, Function3D)):
        raise ValueError("The mask must be a 2D or 3D function.")

    if core_func is None:

        if isinstance(edge_func, Function2D) and return3d:
            return AxisymmetricMapper(edge_func)

        if isinstance(edge_func, VectorFunction2D) and return3d:
            return VectorAxisymmetricMapper(edge_func)

        return edge_func

    if edge_func is None:

        if isinstance(core_func, Function2D) and return3d:
            return AxisymmetricMapper(core_func)

        if isinstance(core_func, VectorFunction2D) and return3d:
            return VectorAxisymmetricMapper(core_func)

        return core_func

    if isinstance(core_func, Function2D) and isinstance(edge_func, Function2D) and isinstance(mask, Function2D):

        blended_func = Blend2D(edge_func, core_func, mask)
        return AxisymmetricMapper(blended_func) if return3d else blended_func

    if isinstance(core_func, VectorFunction2D) and isinstance(edge_func, VectorFunction2D) and isinstance(mask, Function2D):

        blended_func = VectorBlend2D(edge_func, core_func, mask)
        return VectorAxisymmetricMapper(blended_func) if return3d else blended_func

    # unable to return 2D, convert to 3D

    if isinstance(core_func, Function2D):
        core_func = AxisymmetricMapper(core_func)

    if isinstance(core_func, VectorFunction2D):
        core_func = VectorAxisymmetricMapper(core_func)

    if isinstance(edge_func, Function2D):
        edge_func = AxisymmetricMapper(edge_func)

    if isinstance(edge_func, VectorFunction2D):
        edge_func = VectorAxisymmetricMapper(edge_func)

    if isinstance(mask, Function2D):
        mask = AxisymmetricMapper(mask)

    if isinstance(core_func, Function3D) and isinstance(edge_func, Function3D):
        return Blend3D(edge_func, core_func, mask)

    if isinstance(core_func, VectorFunction3D) and isinstance(edge_func, VectorFunction3D):
        return VectorBlend3D(edge_func, core_func, mask)

    raise ValueError("Cannot blend scalar and vector functions.")
