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

def warn_unsupported_species(composition, species_type):

    if species_type in composition and len(composition[species_type]):
        print("Warning! Species of type '{}' are currently not supported. The follwoing species will be skipped:".format(species_type))
        labels = []
        for species_id in composition[species_type]:
            d = {first:second for first, second in species_id}
            labels.append(d['label'])
        print('; '.join(labels))


def get_subset_name_index(subset_id_dict, grid_subset_id):

    subset_id = subset_id_dict.copy()
    subset_id.update({value: key for key, value in subset_id.items()})

    try:
        grid_subset_index = int(grid_subset_id)
        grid_subset_name = subset_id[grid_subset_index]
    except ValueError:
        try:
            grid_subset_name = str(grid_subset_id)
            grid_subset_index = subset_id[grid_subset_name]
        except KeyError:
            raise ValueError("Unable to find a grid subset with ID {}.".format(grid_subset_id))
    except KeyError:
        raise ValueError("Unable to find a grid subset with ID {}.".format(grid_subset_id))

    return grid_subset_name, grid_subset_index
