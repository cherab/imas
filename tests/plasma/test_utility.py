from contextlib import suppress

import pytest
from imas import DBEntry
from imas.ids_defs import MEMORY_BACKEND

from cherab.imas.plasma.utility import get_entry_reference


@pytest.mark.parametrize(
    ("constructor", "entry_kwargs", "expected"),
    [
        (
            "uri",
            {
                "uri_builder": lambda _tmp_path, _path_iter_jintrac: (
                    "imas:memory?path=cherab_test_memory_uri"
                ),
                "mode": "w",
            },
            "imas:memory?path=cherab_test_memory_uri",
        ),
        (
            "uri",
            {
                "uri_builder": lambda _tmp_path, path_iter_jintrac: path_iter_jintrac,
                "mode": "r",
            },
            "__PATH_ITER_JINTRAC__",
        ),
        pytest.param(
            "legacy",
            {
                "backend_id": MEMORY_BACKEND,
                "db_name": "ITER",
                "pulse": 116100,
                "run": 1001,
                "data_version": "3",
            },
            f"backend_id={MEMORY_BACKEND!r}, db_name='ITER', pulse=116100, run=1001, "
            "user_name=None, data_version='3'",
            marks=pytest.mark.requires_imas_memory_backend,
            id="memory-legacy-v3",
        ),
        pytest.param(
            "legacy",
            {
                "backend_id": MEMORY_BACKEND,
                "db_name": "ITER",
                "pulse": 134110,
                "run": 111,
                "data_version": "4",
            },
            f"backend_id={MEMORY_BACKEND!r}, db_name='ITER', pulse=134110, run=111, "
            "user_name=None, data_version='4'",
            marks=pytest.mark.requires_imas_memory_backend,
            id="memory-legacy-v4",
        ),
    ],
    ids=["memory-uri", "file-uri", None, None],
)
def test_get_entry_reference_supports_dbentry_constructor_variants(
    tmp_path,
    path_iter_jintrac: str,
    constructor: str,
    entry_kwargs: dict,
    expected: str,
):
    """Test get_entry_reference() for both URI and legacy DBEntry constructors."""
    if constructor == "uri":
        source_uri = entry_kwargs["uri_builder"](tmp_path, path_iter_jintrac)
        entry = DBEntry(source_uri, entry_kwargs["mode"])
    else:
        entry = DBEntry(
            entry_kwargs["backend_id"],
            entry_kwargs["db_name"],
            entry_kwargs["pulse"],
            entry_kwargs["run"],
            data_version=entry_kwargs["data_version"],
        )

    try:
        actual = get_entry_reference(entry)
        expected_value = path_iter_jintrac if expected == "__PATH_ITER_JINTRAC__" else expected

        assert actual == expected_value
    finally:
        with suppress(Exception):
            entry.close()
