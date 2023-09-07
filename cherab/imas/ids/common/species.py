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

from imas.imasdef import EMPTY_DOUBLE, EMPTY_INT

from cherab.core.atomic.elements import lookup_isotope


def get_ion_state(state, state_index, elements, grid_subset_index=None):

    if state.z_min == EMPTY_DOUBLE or state.z_max == EMPTY_DOUBLE:
        if grid_subset_index is None:  # 1D profiles
            z_average = state.z_average_1d
        else:
            for s in state.z_average:
                if s.grid_subset_index == grid_subset_index:
                    z_average = s.values
        if len(z_average):  # probably, a bundle
            z_min = state.z_min if state.z_min != EMPTY_DOUBLE else z_average.min()
            z_max = state.z_max if state.z_max != EMPTY_DOUBLE else z_average.max()
        else:  # probably, a single ion
            z_min = state.z_min if state.z_min != EMPTY_DOUBLE else state_index + 1
            z_max = state.z_max if state.z_max != EMPTY_DOUBLE else z_min
    else:
        z_min = state.z_min
        z_max = state.z_max

    state_dict = {'label': state.label.strip()}

    if len(elements) > 1:  # molecular ions and bundles
        state_dict['elements'] = elements
        if z_min == z_max:
            species_type = 'molecule'
            state_dict['z'] = z_min
            state_dict['electron_configuration'] = state.electron_configuration if len(state.electron_configuration) else None
            state_dict['vibrational_mode'] = state.vibrational_mode if len(state.vibrational_mode) else None
            state_dict['vibrational_level'] = state.vibrational_level if state.vibrational_level != EMPTY_DOUBLE else None
        else:
            species_type = 'molecular_bundle'
            state_dict['z_min'] = z_min
            state_dict['z_max'] = z_max
    else:  # ions and bundles
        state_dict['element'] = elements[0]
        if z_min == z_max:
            species_type = 'ion'
            state_dict['z'] = z_min
            state_dict['electron_configuration'] = state.electron_configuration if len(state.electron_configuration) else None
        else:
            species_type = 'ion_bundle'
            state_dict['z_min'] = z_min
            state_dict['z_max'] = z_max
    species_id = frozenset(state_dict.items())

    return species_type, species_id


def get_neutral_state(state, elements):
    state_dict = {'label': state.label.strip()}

    if len(elements) > 1:  # molecules
        species_type = 'molecule'
        state_dict['elements'] = elements
        state_dict['z'] = 0
        state_dict['electron_configuration'] = state.electron_configuration if len(state.electron_configuration) else None
        state_dict['vibrational_mode'] = state.vibrational_mode if len(state.vibrational_mode) else None
        state_dict['vibrational_level'] = state.vibrational_level if state.vibrational_level != EMPTY_DOUBLE else None
    else:  # atoms
        species_type = 'ion'
        state_dict['element'] = elements[0]
        state_dict['z'] = 0
        state_dict['electron_configuration'] = state.electron_configuration if len(state.electron_configuration) else None

    species_id = frozenset(state_dict.items())

    return species_type, species_id


def get_ion(ion, elements):

    z_ion = ion.z_ion if ion.z_ion != EMPTY_DOUBLE else elements[0].atomic_number
    if len(elements) > 1:
        species_id = frozenset({('label', ion.label.strip()), ('elements', elements), ('z', z_ion),
                                ('electron_configuration', None), ('vibrational_mode', None), ('vibrational_level', None)})
        return 'molecule', species_id

    species_id = frozenset({('label', ion.label.strip()), ('element', elements[0]), ('z', z_ion),
                            ('electron_configuration', None)})
    return 'ion', species_id


def get_neutral(neutral, elements):

    if len(elements) > 1:
        species_id = frozenset({('label', neutral.label.strip()), ('elements', elements), ('z', 0),
                                ('electron_configuration', None), ('vibrational_mode', None), ('vibrational_level', None)})
        return 'molecule', species_id

    species_id = frozenset({('label', neutral.label.strip()), ('element', elements[0]), ('z', 0),
                            ('electron_configuration', None)})
    return 'ion', species_id


def get_element_list(element_aos):
    elements = []
    for element in element_aos:
        mass_number = int(round(element.a))
        zn = int(round(element.z_n))
        isotope = lookup_isotope(zn, number=mass_number)
        if int(round(isotope.element.atomic_weight)) == mass_number:
            isotope = isotope.element  # prefer element over isotope
        atoms_n = 1 if element.atoms_n == EMPTY_INT else element.atoms_n
        for _ in range(atoms_n):
            elements.append(isotope)

    return elements
