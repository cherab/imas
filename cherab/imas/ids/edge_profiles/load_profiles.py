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

from imas.imasdef import EMPTY_DOUBLE, EMPTY_INT

from cherab.imas.ids.common.species import get_element_list, get_ion_state, get_neutral_state, get_ion, get_neutral


def load_edge_profiles(species_struct, grid_subset_index=5, backup_species_struct=None):
    """
    Loads edge profiles from a given species structe (e.g. ggd[i1].electrons, ggd[i1].ion[i2].states[i3])
    for a given grid and subset indices.

    The returned dictionary values for missing profiles are None.

    :param species_struct: The ids structure containing the profiles for a single species.
    :param grid_subset_index: Identifier index of the grid subset. Default is 5 ("Cells").
    :param backup_species_struct: The backup ids structure that is used if the profile is missing
                                  in species_struct. Default is None.

    :returns: A dictionary with the following keys:
        'density',
        'density_fast',
        'temperature',
        'velocity_radial',
        'velocity_parallel',
        'velocity_poloidal',
        'velocity_toroidal',
        'velocity_r',
        'velocity_z',
        'z_average'.
    """

    profiles = {
        'density': None,
        'density_fast': None,
        'temperature': None,
        'velocity_radial': None,
        'velocity_parallel': None,
        'velocity_poloidal': None,
        'velocity_toroidal': None,
        'velocity_r': None,
        'velocity_z': None,
        'z_average': None,
        }

    scalar_profiles = ('density', 'density_fast', 'temperature', 'z_average')

    for name in scalar_profiles:
        profiles[name] = _get_profile(species_struct, name, grid_subset_index)
        if profiles[name] is None and backup_species_struct is not None:
            profiles[name] = _get_profile(backup_species_struct, name, grid_subset_index)

    # velocity
    velocity_profiles = ('radial', 'parallel', 'poloidal', 'toroidal', 'r', 'z')
    for s in species_struct.velocity:
        if s.grid_subset_index == grid_subset_index:
            for name in velocity_profiles:
                prof = getattr(s, name)
                profiles['velocity_' + name] = np.array(prof) if len(prof) else None
            break    
    if all(profiles['velocity_' + name] is None for name in velocity_profiles) and backup_species_struct is not None:
        for s in backup_species_struct.velocity:
            if s.grid_subset_index == grid_subset_index:
                for name in velocity_profiles:
                    prof = getattr(s, name)
                    profiles['velocity_' + name] = np.array(prof) if len(prof) else None
                break

    return profiles

def load_edge_species(ggd_struct, grid_subset_index=5):
    """
    Loads edge plasma species and their profiles from a given GGD structure
    for a given grid and subset indices.

    :param ggd_struct: The ggd ids structure containing the profiles.
    :param grid_subset_index: Identifier index of the grid subset. Default is 5 ("Cells").

    The returned dictionary has the following structure:
    {
        'electron': {
            'density': array,
            'temperature': array,
            ...
        },
        'molecule': {
            molecule_id: {  # frozenset identifier
                'density': array,
                'temperature': array,
                ...
            },
            ...
        'molecular_bundle': {
            molecular_bundle_id: {  # frozenset identifier
                'density': array,
                'temperature': array,
                ...
            },
            ...
        'ion': {
            ion_id: {  # frozenset identifier
                'density': array,
                'temperature': array,
                ...
            },
            ...
        'ion_bundle': {
            ion_bundle_id: {  # frozenset identifier
                'density': array,
                'temperature': array,
                ...
            },
        },
    },
    where species are identified by frozensets with (key, value) pairs with the following keys:
        molecule: 'label', 'elements', 'z', 'electron_configuration', 'vibrational_level', 'vibrational_mode';
        molecular_bundle: 'label', 'elements', 'z_min', 'z_max';
        ion: 'label', 'element', 'z', 'electron_configuration';
        ion_bundle: 'label', 'element', 'z_min', 'z_max';

    :returns: A dictionary with plasma profiles.
    """

    species_types = ('molecule', 'molecular_bundle', 'ion', 'ion_bundle')
    composition = {species_type: {} for species_type in species_types}

    composition['electron'] = load_edge_profiles(ggd_struct.electrons, grid_subset_index)

    # ions
    ion_elements = []
    for ion in ggd_struct.ion:
        elements = tuple(get_element_list(ion.element))
        if not len(elements):
            raise RuntimeError("Unable to determine the ion species, ion.element AOS is empty.")
        ion_elements.append(elements)

        if len(ion.state):
            shared_temperature = _get_profile(ion, 'temperature', grid_subset_index)
            backup_ids = None if len(ion.state) > 1 else ion
            for i, state in enumerate(ion.state):
                species_type, species_id = get_ion_state(state, i, elements, grid_subset_index)
                if species_id in composition[species_type]:
                    print("Warning! Skipping duplicated ion: {}".format(state.label.strip()))
                    continue
                profiles = load_edge_profiles(state, grid_subset_index, backup_ids)
                if backup_ids is None and profiles['temperature'] is None:
                    profiles['temperature'] = shared_temperature
                composition[species_type][species_id] = profiles
        else:
            species_type, species_id = get_ion(ion, elements)
            if species_id in composition[species_type]:
                print("Warning! Skipping duplicated ion: {}".format(ion.label.strip()))
            else:
                composition[species_type][species_id] = load_edge_profiles(ion, grid_subset_index)
    
    # neutrals
    for neutral in ggd_struct.neutral:
        elements = tuple(get_element_list(neutral.element))
        if not len(elements):
            elements = ion_elements[neutral.ion_index - 1]

        if len(neutral.state):
            shared_temperature = _get_profile(neutral, 'temperature', grid_subset_index)
            backup_ids = None if len(neutral.state) > 1 else neutral
            for state in neutral.state:
                species_type, species_id = get_neutral_state(state, elements)
                if species_id in composition[species_type]:
                    print("Warning! Skipping duplicated neutral: {}".format(state.label.strip()))
                    continue
                profiles = load_edge_profiles(state, grid_subset_index, backup_ids)
                if backup_ids is None and profiles['temperature'] is None:
                    profiles['temperature'] = shared_temperature
                composition[species_type][species_id] = profiles
        else:
            species_type, species_id = get_neutral(neutral, elements)
            if species_id in composition[species_type]:
                print("Warning! Skipping duplicated neutral: ".format(neutral.label.strip()))
            else:
                composition[species_type][species_id] = load_edge_profiles(neutral, grid_subset_index)

    # Replace missing species temperature with average ion temperature
    tion = _get_profile(ggd_struct, 't_i_average', grid_subset_index)
    if tion is not None:
        for species_type in species_types:
            for species_id, profiles in composition[species_type].items():
                if profiles['temperature'] is None:
                    d = {first:second for first, second in species_id}
                    print('Warning! Using average ion temperature for the {} {}.'.format(d['label'], species_type))
                    profiles['temperature'] = tion

    return composition


def _get_profile(species_struct, name, grid_subset_index):
    if hasattr(species_struct, name):
        for s in getattr(species_struct, name):
            if s.grid_subset_index == grid_subset_index:
                return np.array(s.values)
