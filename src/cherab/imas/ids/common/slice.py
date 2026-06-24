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
from uuid import uuid4

from numpy import inf

from imas.db_entry import DBEntry
from imas.ids_defs import CLOSEST_INTERP, MEMORY_BACKEND
from imas.ids_toplevel import IDSToplevel

__all__ = ["get_ids_time_slice"]


def _slice_via_memory_backend(
    ids: IDSToplevel,
    ids_name: str,
    time: float,
    occurrence: int,
) -> IDSToplevel:
    """Re-slice an IDS by round-tripping through the IMAS memory backend.

    Parameters
    ----------
    ids
        The IDS to re-slice.
    ids_name
        The name of the IDS.
    time
        The time in seconds of the requested time slice.
    occurrence
        The occurrence of the IDS.

    Returns
    -------
    `~imas.ids_toplevel.IDSToplevel`
        The re-sliced IDS.
    """
    token = uuid4().int
    # Use per-call identifiers so repeated/concurrent calls do not collide.
    temp_entry = DBEntry(
        MEMORY_BACKEND,
        f"cherab_tmp_{token & 0xFFFF:04x}",
        1 + token % 2_000_000_000,
        1 + (token >> 31) % 2_000_000_000,
    )
    temp_entry.create()
    try:
        temp_entry.put(ids, occurrence=occurrence)
        return temp_entry.get_slice(
            ids_name,
            time,
            CLOSEST_INTERP,
            occurrence=occurrence,
            autoconvert=False,
        )
    finally:
        temp_entry.close()


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
        URI, this function falls back to `~imas.db_entry.DBEntry.get` and tries to re-slice the IDS
        by round-tripping through the IMAS memory backend. If that second step fails, it returns the
        full IDS with a warning.

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

    is_time_sliced = True

    try:
        ids = entry.get_slice(
            ids_name,
            time,
            CLOSEST_INTERP,
            occurrence=occurrence,
            autoconvert=False,
        )
    except NotImplementedError:
        ids = entry.get(ids_name, occurrence=occurrence, autoconvert=False)
        is_time_sliced = len(ids.time) <= 1
        get_slice_unavailable_msg = (
            f"The 'get_slice' method is not implemented for the URI '{entry.uri}'."
        )

        if not is_time_sliced:
            try:
                ids = _slice_via_memory_backend(ids, ids_name, time, occurrence)
                is_time_sliced = True
                warnings.warn(
                    get_slice_unavailable_msg
                    + " "
                    + "Falling back to 'get' and re-slicing via the IMAS memory backend.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except Exception as exc:
                warnings.warn(
                    get_slice_unavailable_msg
                    + " "
                    + "Fallback re-slicing via the IMAS memory backend failed. "
                    + "Returning the full "
                    + f"'{ids_name}' IDS without reducing to a single time slice. Error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                get_slice_unavailable_msg
                + " "
                + "Falling back to 'get' method because the returned IDS contains a single time slice.",
                RuntimeWarning,
                stacklevel=2,
            )

    if not len(ids.time):
        raise RuntimeError(f"The '{ids_name}' IDS is empty.")

    nearest_time = min(ids.time, key=lambda t: abs(float(t) - time))

    if abs(float(nearest_time) - time) > time_threshold:
        raise RuntimeError(
            f"The time difference between the actual time ({nearest_time} s) "
            + f"of the nearest '{ids_name}' time slice and the given time ({time} s) "
            + f"exceeds the specified threshold ({time_threshold} s)."
        )

    if not is_time_sliced:
        warnings.warn(
            f"Returning '{ids_name}' IDS with {len(ids.time)} time slices because a single-time "
            + "fallback could not be constructed.",
            RuntimeWarning,
            stacklevel=2,
        )

    return ids
