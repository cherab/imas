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

from raysect.core.math.function.float import Interpolator1DArray
from raysect.core.math.function.vector3d import Constant3D as ConstantVector3D
from raysect.core.math import translate, Vector3D
from raysect.primitive import Cylinder, Subtract

from cherab.core import Plasma, Species, Maxwellian
from cherab.core.math import VectorAxisymmetricMapper
from cherab.core.utility import RecursiveDict
from cherab.tools.equilibrium import EFITEquilibrium

from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.core_profiles import load_core_species, load_core_grid
from cherab.imas.plasma.equilibrium import load_equilibrium, load_magnetic_field
from cherab.imas.plasma.utility import warn_unsupported_species


def load_core_plasma(shot, run, user, database, backend=imas.imasdef.MDSPLUS_BACKEND, time=0, occurrence=0,
                     equilibrium=None, b_field=None, psi_interpolator=None, time_threshold=np.inf, parent=None):
    """
    Loads core profiles and creates a Plasma object.
    Prefers 'density_thermal' over 'density' profile.

    :param shot: IMAS shot number.
    :param run: IMAS run number for a given shot.
    :param user: IMAS username.
    :param database: IMAS database name.
    :param backend: IMAS database backend. Default is imas.imasdef.MDSPLUS_BACKEND.
    :param time: Time moment. Default is 0.
    :param occurrence: Instance index of the 'core_profiles' IDS. Default is 0.
    :param equilibrium: Alternative EFITEquilibrium object used to map core profiles.
        Default is None: equilibrium is read from the same shot, run, time, etc. as the core profiles.
    :param b_field: An alternative 2D interpolator of the magnetic field vector (Br, Btor, Bz).
        Default is None: the magnetic field will be taken from the equilibrium.
    :param psi_interpolator: Alternative psi_norm(rho_tor_norm) interpolator.
        Used only if 'psi' is missing in the core grid. Default is None: obtained from the equilibrium IDS.
    :param time_threshold: Sets the maximum allowable difference between the specified time and the nearest
        available time. Default is np.inf.
    :param parent: The parent node in the Raysect scene-graph. Default is None.
    """

    entry = imas.DBEntry(backend, database, shot, run, user)
    core_profiles_ids = get_ids_time_slice(entry, 'core_profiles', time=time, occurrence=occurrence, time_threshold=time_threshold)

    if not len(core_profiles_ids.profiles_1d):
        raise RuntimeError('The profiles_1d AOS in core_profiles IDS is empty.')

    if equilibrium is None:
        equilibrium, psi_interp = load_equilibrium(shot, run, user, database, backend=backend, time=time, with_psi_interpolator=True)
        psi_interpolator = psi_interpolator or psi_interp

    if not isinstance(equilibrium, EFITEquilibrium):
        raise ValueError("Argiment equilibrium must be a EFITEquilibrium instance.")

    if b_field is None:
        try:
            b_field = load_magnetic_field(shot, run, user, database, backend=backend, time=time)
        except RuntimeError:
            b_field = equilibrium.b_field

    core_grid = load_core_grid(core_profiles_ids.profiles_1d[0].grid)

    composition = load_core_species(core_profiles_ids.profiles_1d[0])

    psi_norm = get_psi_norm(core_grid['psi'], equilibrium.psi_axis, equilibrium.psi_lcfs, core_grid['rho_tor_norm'], psi_interpolator)

    name = 'IMAS core plasma: shot {}, run {}, time {}.'.format(shot, run, core_profiles_ids.time[0])
    plasma = Plasma(parent=parent, name=name)

    # Create plasma geometry
    radius_inner, radius_outer = equilibrium.r_range
    zmin, zmax = equilibrium.z_range
    height = zmax - zmin
    plasma.geometry = Subtract(Cylinder(radius_outer, height), Cylinder(radius_inner, height))
    plasma.geometry_transform = translate(0, 0, zmin)

    plasma.b_field = VectorAxisymmetricMapper(b_field)

    # Add electron species
    electrons = get_core_interpolators(psi_norm, composition['electron'], equilibrium, return3d=True)
    if electrons['density_thermal'] is not None:
        electrons['density'] = electrons['density_thermal']

    if electrons['density'] is None:
        print("Unable to create Core Plasma: electron density is not available.")
    if electrons['temperature'] is None:
        print("Unable to create Core Plasma: electron temperature is not available.")

    zero_velocity = ConstantVector3D(Vector3D(0, 0, 0))

    plasma.electron_distribution = Maxwellian(electrons['density'], electrons['temperature'],
                                              zero_velocity, electron_mass)

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

        interp = get_core_interpolators(psi_norm, profiles, equilibrium, return3d=True)
        if interp['density_thermal'] is not None:
            interp['density'] = interp['density_thermal']

        if interp['density'] is None:
            print("Warning! Skipping {} species: density is not available.".format(d['label']))
        if interp['temperature'] is None:
            print("Warning! Skipping {} species: temperature is not available.".format(d['label']))

        distribution = Maxwellian(interp['density'], interp['temperature'],
                                  zero_velocity, species_type.atomic_weight * atomic_mass)

        plasma.composition.add(Species(species_type, charge, distribution))

    return plasma


def get_core_interpolators(psi_norm, profiles, equilibrium, return3d=False):
    """
    Create interpolators for the core profiles.

    :param psi_norm: psi_norm array.
    :param profiles: A dictionary with core plasma profiles.
    :param equilibrium: EFITEquilibrium object used to map core profiles.
    :param return3d: If True, return the 3D interpolators assuming rotational symmetry.
        Default is False.

    :returns: A dictionary with core interpolators.
    """

    if not isinstance(equilibrium, EFITEquilibrium):
        raise ValueError("Argiment equilibrium must be a EFITEquilibrium instance.")

    psi_norm, indx = np.unique(psi_norm, return_index=True)

    interpolators = RecursiveDict()

    for prof_key, profile in profiles.items():
        if profile is not None:
            extrapolation_range = max(0, psi_norm[0], 1. - psi_norm[-1])
            func = Interpolator1DArray(psi_norm, profile[indx], 'cubic', 'nearest', extrapolation_range)
            interpolators[prof_key] = equilibrium.map3d(func) if return3d else equilibrium.map2d(func)
        else:
            interpolators[prof_key] = None

    return interpolators.freeze()


def get_psi_norm(psi, psi_axis, psi_lcfs, rho_tor_norm, psi_interpolator):

    if psi is None:
        if psi_interpolator is None:
            raise RuntimeError('Unable to map rho_tor_norm to psi_norm grid: psi_interpolator is not provided.')

        if rho_tor_norm is None:
            raise RuntimeError('No rho_tor_norm values are available in the core grid: unable to interpolate to psi_norm.')

        return np.array([psi_interpolator(rho) for rho in rho_tor_norm])

    return (-psi / (2 * np.pi) - psi_axis) / (psi_lcfs - psi_axis)
