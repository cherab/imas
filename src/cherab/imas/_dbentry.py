"""Internal helpers for opening IMAS database entries."""

from __future__ import annotations

from os import PathLike, fspath
from typing import Any
from warnings import warn

from imas import DBEntry  # type: ignore[attr-defined]


def _open_dbentry_for_reading(*args: Any, **kwargs: Any) -> DBEntry:
    """Create a DBEntry for use by a CHERAB-IMAS data loader.

    URI-style DBEntry construction requires an explicit mode in IMAS-Python. CHERAB-IMAS
    loader APIs only read existing data, so this helper supplies ``"r"`` automatically.
    The legacy ``(backend_id, db_name, pulse, run, ...)`` constructor is passed through
    unchanged because it opens an existing entry when used as a context manager.

    An explicitly supplied ``"r"`` is accepted temporarily for backwards compatibility.
    Modes that may create or replace data are rejected.

    Returns
    -------
    `imas.DBEntry`
        Database entry opened or prepared for reading.

    Raises
    ------
    TypeError
        If mode is supplied both positionally and by keyword.
    """
    positional = list(args)

    if positional and isinstance(positional[0], (str, PathLike)):
        uri = fspath(positional.pop(0))
        positional_mode = positional.pop(0) if positional else None
        keyword_mode = kwargs.pop("mode", None)

        if positional_mode is not None and keyword_mode is not None:
            raise TypeError("DBEntry mode was specified both positionally and by keyword.")

        mode = positional_mode if positional_mode is not None else keyword_mode
        _validate_read_mode(mode)
        return DBEntry(uri, "r", *positional, **kwargs)

    if not positional and "uri" in kwargs:
        uri = fspath(kwargs.pop("uri"))
        mode = kwargs.pop("mode", None)
        _validate_read_mode(mode)
        return DBEntry(uri, "r", **kwargs)

    return DBEntry(*positional, **kwargs)


def _validate_read_mode(mode: Any) -> None:
    if mode is None:
        return

    if mode != "r":
        raise ValueError(f"CHERAB-IMAS loader APIs only support mode 'r'; received mode {mode!r}.")

    warn(
        "Passing mode 'r' to CHERAB-IMAS loader APIs is deprecated; "
        "the read mode is now selected automatically.",
        DeprecationWarning,
        stacklevel=3,
    )
