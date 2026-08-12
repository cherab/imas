import pytest
from imas import DBEntry

from cherab.imas.ids.common import (
    IDSPathReference,
    load_ids_path_reference,
    parse_ids_path_fragment,
    resolve_ids_path_reference,
)


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        (
            "#core_profiles",
            IDSPathReference(ids_name="core_profiles"),
        ),
        (
            "#core_profiles:2",
            IDSPathReference(ids_name="core_profiles", occurrence=2),
        ),
        (
            "imas://example?path=/data#edge_profiles:1/grid_ggd(:)/path",
            IDSPathReference(
                ids_name="edge_profiles",
                occurrence=1,
                idspath="/grid_ggd(:)/path",
            ),
        ),
        (
            "#grid_ggd/space(1)/objects_per_dimension(1)",
            IDSPathReference(
                ids_name="grid_ggd",
                idspath="/space(1)/objects_per_dimension(1)",
            ),
        ),
    ],
)
def test_parse_ids_path_fragment(reference: str, expected: IDSPathReference) -> None:
    assert parse_ids_path_fragment(reference) == expected


@pytest.mark.parametrize("reference", ["", "#", "#:1", "#core_profiles:", "#core_profiles:abc"])
def test_parse_ids_path_fragment_rejects_invalid_input(reference: str) -> None:
    with pytest.raises(ValueError):
        parse_ids_path_fragment(reference)


def test_resolve_ids_path_reference_on_loaded_ids(path_iter_jintrac: str) -> None:
    from cherab.imas.ids.common import get_ids_time_slice

    with DBEntry(path_iter_jintrac, "r") as entry:
        ids = get_ids_time_slice(entry, "edge_profiles", time=0)

    reference = IDSPathReference(ids_name=str(ids.metadata.name), idspath="/grid_ggd")
    resolved = resolve_ids_path_reference(ids, reference)

    assert resolved.metadata.path == ids.grid_ggd.metadata.path
    assert type(resolved) is type(ids.grid_ggd)


def test_load_ids_path_reference_from_entry(path_iter_jintrac: str) -> None:
    with DBEntry(path_iter_jintrac, "r") as entry:
        resolved = load_ids_path_reference(entry, "#edge_profiles/grid_ggd")

    from cherab.imas.ids.common import get_ids_time_slice

    with DBEntry(path_iter_jintrac, "r") as entry:
        ids = get_ids_time_slice(entry, "edge_profiles", time=0)

    assert resolved.metadata.path == ids.grid_ggd.metadata.path
    assert type(resolved) is type(ids.grid_ggd)
