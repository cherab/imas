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

import imas

from raysect.core.math.function.float import Interpolator1DArray, Interpolator2DArray
from raysect.core.math.function.vector3d import FloatToVector3DFunction2D

from cherab.tools.equilibrium import EFITEquilibrium

from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.equilibrium import load_equilibrium_data, load_magnetic_field_data


def load_equilibrium(shot, run, user, database, backend=imas.imasdef.MDSPLUS_BACKEND, time=0, occurrence=0, time_threshold=np.inf,
                     with_psi_interpolator=False):
    """
    Loads plasma equilibrium from the equilibrium IDS and creates EFITEquilibrium object.

    :param shot: IMAS shot number.
    :param run: IMAS run number for a given shot.
    :param user: IMAS username.
    :param database: IMAS database name.
    :param backend: IMAS database backend. Default is imas.imasdef.MDSPLUS_BACKEND.
    :param time: Time moment. Default is 0.
    :param occurrence: Instance index of the 'equilibrium' IDS. Default is 0.
    :param time_threshold: Sets the maximum allowable difference between the specified time and the nearest
        available time. Default is np.inf.
    :param with_psi_interpolator: If True, return the psi_norm(rho_tor_norm) interpolator. 

    :returns: EFITEquilibrium object and Function1D interpolator for the psi_norm(rho_tor_norm) (optionaly).
    """

    entry = imas.DBEntry(backend, database, shot, run, user)
    equilibrium_ids = get_ids_time_slice(entry, 'equilibrium', time=time, occurrence=occurrence, time_threshold=time_threshold)

    equilibrium_dict = load_equilibrium_data(equilibrium_ids)

    entry.close()

    cocos_11to3(equilibrium_dict)

    equilibrium_dict['psi_norm'][0] = min(0, equilibrium_dict['psi_norm'][0])
    equilibrium_dict['psi_norm'][-1] = max(1., equilibrium_dict['psi_norm'][-1])

    f_profile = np.array([equilibrium_dict['psi_norm'], equilibrium_dict['f']])
    q_profile = np.array([equilibrium_dict['psi_norm'], equilibrium_dict['q']])

    equilibrium = EFITEquilibrium(equilibrium_dict['r'],
                           equilibrium_dict['z'],
                           equilibrium_dict['psi_grid'],
                           equilibrium_dict['psi_axis'],
                           equilibrium_dict['psi_lcfs'],
                           equilibrium_dict['magnetic_axis'],
                           equilibrium_dict['x_points'],
                           equilibrium_dict['strike_points'],
                           f_profile,
                           q_profile,
                           equilibrium_dict['b_vacuum_radius'],
                           equilibrium_dict['b_vacuum_magnitude'],
                           equilibrium_dict['lcfs_polygon'],
                           None,
                           equilibrium_dict['time'])
    
    if not with_psi_interpolator:
        return equilibrium
    
    if equilibrium_dict['rho_tor_norm'] is None:
        return equilibrium, None
    
    psi_interpolator = Interpolator1DArray(equilibrium_dict['rho_tor_norm'], equilibrium_dict['psi_norm'], 'cubic', 'none', 0)

    return equilibrium, psi_interpolator


def load_magnetic_field(shot, run, user, database, backend=imas.imasdef.MDSPLUS_BACKEND, time=0, occurrence=0, time_threshold=np.inf):
    """
    Loads the magnetic field from the equilibrium IDS and returns a VectorFunction2D interpolator.

    :param shot: IMAS shot number.
    :param run: IMAS run number for a given shot.
    :param user: IMAS username.
    :param database: IMAS database name.
    :param backend: IMAS database backend. Default is imas.imasdef.MDSPLUS_BACKEND.
    :param time: Time moment. Default is 0.
    :param occurrence: Instance index of the 'equilibrium' IDS. Default is 0.
    :param time_threshold: Sets the maximum allowable difference between the specified time and the nearest
        available time. Default is np.inf.

    :returns: VectorFunction2D.
    """

    entry = imas.DBEntry(backend, database, shot, run, user)
    equilibrium_ids = get_ids_time_slice(entry, 'equilibrium', time=time, occurrence=occurrence, time_threshold=time_threshold)
    
    if not len(equilibrium_ids.time_slice):
        raise RuntimeError('Equilibrium IDS does not have a time slice.')

    b_dict = load_magnetic_field_data(equilibrium_ids.time_slice[0].profiles_2d)

    br = Interpolator2DArray(b_dict['r'], b_dict['z'], b_dict['b_field_r'], 'cubic', 'none', 0, 0)
    btor = Interpolator2DArray(b_dict['r'], b_dict['z'], b_dict['b_field_tor'], 'cubic', 'none', 0, 0)
    bz = Interpolator2DArray(b_dict['r'], b_dict['z'], b_dict['b_field_z'], 'cubic', 'none', 0, 0)

    return FloatToVector3DFunction2D(br, btor, bz)


def cocos_11to3(equilibrium_dict):
    """
    Converts from COCOS 11 convention used in IMAS to COCOS 3 convention used in EFIT.
    """

    equilibrium_dict['psi_grid'] = -equilibrium_dict['psi_grid'] / (2 * np.pi)
    equilibrium_dict['psi_axis'] = -equilibrium_dict['psi_axis'] / (2 * np.pi)
    equilibrium_dict['psi_lcfs'] = -equilibrium_dict['psi_lcfs'] / (2 * np.pi)
    equilibrium_dict['q'] = -equilibrium_dict['q']
    if equilibrium_dict['phi'] is not None:
        equilibrium_dict['phi'] = -equilibrium_dict['phi']
