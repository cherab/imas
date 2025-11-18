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
"""Module for common functions used to get IDS time slices."""

import warnings

from numpy import inf

from imas import DBEntry
from imas.ids_defs import CLOSEST_INTERP
from imas.ids_toplevel import IDSToplevel

__all__ = ["get_ids_time_slice"]


def get_ids_time_slice(
    entry: DBEntry,
    ids_name: str,
    time: float = 0,
    occurrence: int = 0,
    time_threshold: float = inf,
) -> IDSToplevel:
    """Get a time slice of the specified IDS from the given IMAS entry.

    .. note::
        If the `~imas.db_entry.DBEntry.get_slice` method is not implemented for the given IMAS entry
        URI, this function will fall back to using the `~imas.db_entry.DBEntry.get` method and
        return the entire IDS.

    Parameters
    ----------
    entry
        The IMAS entry. The entry must be opened in read mode.
    ids_name
        The name of the IDS.
    time
        The time in seconds of the requested time slice, by default is 0.
    occurrence
        The occurrence of the IDS, by default is 0.
    time_threshold
        The maximum allowed time difference in seconds between the actual time of the nearest time
        slice and the given time, by default is infinity.

    Returns
    -------
    `~imas.ids_toplevel.IDSToplevel`
        The requested IDS time slice.

    Raises
    ------
    ValueError
        If `.time` or `.time_threshold` is negative.
    RuntimeError
        If the requested IDS is empty.
    RuntimeError
        If the time difference between the actual time of the nearest time slice and the given time
        exceeds the specified threshold.

    Examples
    --------
    .. code-block:: python

        from imas import DBEntry
        from cherab.imas.ids.common import get_ids_time_slice

        with DBEntry(
            "imas://uda.iter.org/uda?path=/work/imas/shared/imasdb/ITER/3/123072/3&backend=hdf5",
            "r",
        ) as entry:
            ids = get_ids_time_slice(entry, "equilibrium", time=0.0)
    """
    if time < 0:
        raise ValueError(f"Argument 'time' must be >=0 ({time} s).")
    if time_threshold < 0:
        raise ValueError(f"Argument 'time_threshold' must be >=0 ({time_threshold} s).")

    try:
        ids = entry.get_slice(
            ids_name,
            time,
            CLOSEST_INTERP,
            occurrence=occurrence,
            autoconvert=False,
        )
    except NotImplementedError:
        # Fallback to `get` method to retrieve the entire IDS
        warnings.warn(
            f"The 'get_slice' method is not implemented for the URI '{entry.uri}'. "
            + "Falling back to 'get' method to retrieve the entire IDS.",
            RuntimeWarning,
            stacklevel=2,
        )
        ids = entry.get(ids_name, occurrence=occurrence, autoconvert=False)

    if not len(ids.time):
        raise RuntimeError(f"The '{ids_name}' IDS is empty.")

    if abs(ids.time[0] - time) > time_threshold:
        raise RuntimeError(
            f"The time difference between the actual time ({ids.time[0]} s) "
            + f"of the nearest '{ids_name}' time slice and the given time ({time} s) "
            + f"exceeds the specified threshold ({time_threshold} s)."
        )

    return ids
