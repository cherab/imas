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

from numpy import inf

import imas


def get_ids_time_slice(
    entry: imas.DBEntry, ids_name: str, time=0, occurrence=0, time_threshold=inf
):
    if time < 0:
        raise ValueError("Argument 'time' must be >=0.")
    if time_threshold < 0:
        raise ValueError("Argument 'time_threshold' must be >=0.")

    entry.open()
    ids = entry.get_slice(ids_name, time, imas.ids_defs.CLOSEST_INTERP, occurrence=occurrence)
    entry.close()

    if not len(ids.time):
        raise RuntimeError(f"The {ids_name} IDS is empty.")

    if abs(ids.time[0] - time) > time_threshold:
        raise RuntimeError(
            f"The time difference between the actual time ({ids.time[0]} s) of the nearest {ids_name}"
            + f" time slice and the given time ({time} s) exceeds the specified threshold ({time_threshold} s)."
        )

    return ids
