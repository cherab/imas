"""Helpers for parsing and resolving IMAS IDS path references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from imas.db_entry import DBEntry
from imas.ids_struct_array import IDSStructArray
from imas.ids_structure import IDSStructure
from imas.ids_toplevel import IDSToplevel

__all__ = [
    "IDSPathReference",
    "parse_ids_path_fragment",
    "resolve_ids_path_reference",
    "load_ids_path_reference",
]


@dataclass(frozen=True, slots=True)
class IDSPathReference:
    """Representation of an IMAS same-document IDS path reference."""

    ids_name: str
    """Referenced IDS name."""

    occurrence: int | None = None
    """Referenced IDS occurrence, if explicitly provided."""

    idspath: str = ""
    """IDS path fragment that follows the referenced IDS name."""


def parse_ids_path_fragment(reference: str) -> IDSPathReference:
    """Parse a same-document IMAS IDS path fragment.

    Parameters
    ----------
    reference
        A string in the form ``#ids[:occurrence][/idspath]``.

    Returns
    -------
    `.IDSPathReference`
        Parsed IDS path reference.

    Raises
    ------
    ValueError
        If the reference is empty or does not follow the expected fragment syntax.
    """
    if not reference:
        raise ValueError("The IDS path reference cannot be empty.")

    fragment = reference.rsplit("#", 1)[-1].strip()
    if not fragment:
        raise ValueError("The IDS path reference cannot be empty.")

    ids_part, separator, idspath = fragment.partition("/")

    if not ids_part:
        raise ValueError(f"Invalid IDS path reference '{reference}'.")

    ids_name, colon, occurrence_text = ids_part.partition(":")
    if not ids_name:
        raise ValueError(f"Invalid IDS path reference '{reference}'.")

    occurrence: int | None
    if colon:
        if not occurrence_text:
            raise ValueError(f"Invalid IDS path reference '{reference}'.")
        try:
            occurrence = int(occurrence_text)
        except ValueError as exc:
            raise ValueError(f"Invalid IDS occurrence '{occurrence_text}'.") from exc
    else:
        occurrence = None

    return IDSPathReference(
        ids_name=ids_name,
        occurrence=occurrence,
        idspath=f"/{idspath}" if separator else "",
    )


def resolve_ids_path_reference(
    ids: IDSToplevel | IDSStructure | IDSStructArray,
    reference: IDSPathReference | str,
) -> IDSToplevel | IDSStructure | IDSStructArray:
    """Resolve an IDS path reference against an already loaded IDS object.

    Parameters
    ----------
    ids
        The loaded IDS root object to resolve the path against.
    reference
        Parsed IDS path reference or a string in the form ``#ids[:occurrence][/idspath]``.

    Returns
    -------
    `~imas.ids_toplevel.IDSToplevel` | `~imas.ids_structure.IDSStructure` | `~imas.ids_struct_array.IDSStructArray`
        The resolved IDS object.

    Raises
    ------
    ValueError
        If the reference does not target the provided IDS root, or the path syntax is invalid.
    """
    ref = parse_ids_path_fragment(reference) if isinstance(reference, str) else reference

    ids_name = _get_ids_name(ids)
    if ids_name is not None and ref.ids_name != ids_name:
        raise ValueError(
            f"The reference targets IDS '{ref.ids_name}', but the provided IDS root is '{ids_name}'."
        )

    if not ref.idspath:
        return ids

    return _resolve_idspath(ids, ref.idspath)


def load_ids_path_reference(
    entry: DBEntry,
    reference: IDSPathReference | str,
) -> IDSToplevel | IDSStructure | IDSStructArray:
    """Load and resolve an IDS path reference from a data entry.

    Parameters
    ----------
    entry
        Open IMAS data entry.
    reference
        Parsed IDS path reference or a string in the form ``#ids[:occurrence][/idspath]``.

    Returns
    -------
    `~imas.ids_toplevel.IDSToplevel` | `~imas.ids_structure.IDSStructure` | `~imas.ids_struct_array.IDSStructArray`
        The resolved IDS object.
    """
    ref = parse_ids_path_fragment(reference) if isinstance(reference, str) else reference
    ids = entry.get(ref.ids_name, occurrence=ref.occurrence or 0, autoconvert=False, lazy=True)
    if not ref.idspath:
        return ids

    return _resolve_idspath(ids, ref.idspath)


def _get_ids_name(ids: IDSToplevel | IDSStructure | IDSStructArray) -> str | None:
    metadata = getattr(ids, "metadata", None)
    return getattr(metadata, "name", None)


def _resolve_idspath(
    ids: IDSToplevel | IDSStructure | IDSStructArray,
    idspath: str,
) -> IDSToplevel | IDSStructure | IDSStructArray:
    current: IDSToplevel | IDSStructure | IDSStructArray = ids

    for segment in idspath.lstrip("/").split("/"):
        if not segment:
            raise ValueError(f"Invalid IDS path '{idspath}'.")

        name, index_expression = _split_segment(segment)
        current = cast(IDSToplevel | IDSStructure | IDSStructArray, getattr(current, name))

        if index_expression is None:
            continue

        if not isinstance(current, IDSStructArray):
            raise ValueError(f"Segment '{segment}' refers to a non-array IDS node.")

        current = _resolve_struct_array_index(current, index_expression)

    return current


def _split_segment(segment: str) -> tuple[str, str | None]:
    if "(" not in segment:
        return segment, None

    name, _, remainder = segment.partition("(")
    if not name or not remainder.endswith(")"):
        raise ValueError(f"Invalid IDS path segment '{segment}'.")

    return name, remainder[:-1]


def _resolve_struct_array_index(
    array: IDSStructArray, index_expression: str
) -> IDSStructure | IDSStructArray:
    index_expression = index_expression.strip()

    if not index_expression or index_expression == ":":
        return array

    if index_expression.startswith("{") or ":" in index_expression:
        raise NotImplementedError(
            "Resolving IDS path fragments with array slices or index sets is not supported yet."
        )

    index = int(index_expression)
    if index == 0:
        raise ValueError("IMAS IDS path indices are 1-based and cannot be 0.")

    return array[index - 1 if index > 0 else index]
