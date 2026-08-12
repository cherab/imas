from pathlib import Path

import pytest

from cherab.imas import _dbentry


class _DummyDBEntry:
    pass


@pytest.fixture
def dbentry_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple, dict]]:
    calls = []

    def create_dbentry(*args, **kwargs):
        calls.append((args, kwargs))
        return _DummyDBEntry()

    monkeypatch.setattr(_dbentry, "DBEntry", create_dbentry)
    return calls


@pytest.mark.parametrize("uri", ["imas:hdf5?path=testdb", Path("data.nc")])
def test_uri_mode_is_added_automatically(uri, dbentry_spy):
    entry = _dbentry._open_dbentry_for_reading(uri, dd_version="4.1.0")

    assert isinstance(entry, _DummyDBEntry)
    assert dbentry_spy == [((str(uri), "r"), {"dd_version": "4.1.0"})]


def test_keyword_uri_mode_is_added_automatically(dbentry_spy):
    _dbentry._open_dbentry_for_reading(uri="data.nc", xml_path="IDSDef.xml")

    assert dbentry_spy == [(("data.nc", "r"), {"xml_path": "IDSDef.xml"})]


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("data.nc", "r"), {}),
        (("data.nc",), {"mode": "r"}),
        ((), {"uri": "data.nc", "mode": "r"}),
    ],
)
def test_explicit_read_mode_is_deprecated(args, kwargs, dbentry_spy):
    with pytest.deprecated_call(match="selected automatically"):
        _dbentry._open_dbentry_for_reading(*args, **kwargs)

    assert dbentry_spy == [(("data.nc", "r"), {})]


@pytest.mark.parametrize("mode", ["w", "a", "x", "r+"])
def test_non_read_mode_is_rejected(mode, dbentry_spy):
    with pytest.raises(ValueError, match="only support mode 'r'"):
        _dbentry._open_dbentry_for_reading("data.nc", mode)

    assert not dbentry_spy


def test_duplicate_mode_is_rejected(dbentry_spy):
    with pytest.raises(TypeError, match="both positionally and by keyword"):
        _dbentry._open_dbentry_for_reading("data.nc", "r", mode="r")

    assert not dbentry_spy


def test_legacy_constructor_is_passed_through(dbentry_spy):
    _dbentry._open_dbentry_for_reading(13, "ITER", 12345, 1, "user", dd_version="3.42.0")

    assert dbentry_spy == [((13, "ITER", 12345, 1, "user"), {"dd_version": "3.42.0"})]
