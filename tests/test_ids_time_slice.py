import numpy as np
import pytest
from imas.db_entry import DBEntry
from imas.ids_toplevel import IDSToplevel

from cherab.imas.ids.common import get_ids_time_slice


class _FakeIDS(IDSToplevel):
    def __init__(self, time):
        self.time = np.asarray(time)


class _FallbackOnlyEntry(DBEntry):
    def __init__(self, ids, uri: str = "imas://example"):
        self._ids = ids
        self.uri = uri

    def get_slice(self, *args, **kwargs):
        raise NotImplementedError

    def get(self, *args, **kwargs):
        return self._ids


def test_get_ids_time_slice_warns_for_single_time_fallback() -> None:
    entry = _FallbackOnlyEntry(_FakeIDS([0.25]))

    with pytest.warns(RuntimeWarning, match="Falling back to 'get' method"):
        ids = get_ids_time_slice(entry, "radiation", time=0.25)

    assert ids.time[0] == pytest.approx(0.25)


def test_get_ids_time_slice_reslices_multi_time_fallback(monkeypatch) -> None:
    entry = _FallbackOnlyEntry(_FakeIDS([0.1, 0.2, 0.3]))

    def _fake_reslice(*args, **kwargs):
        return _FakeIDS([0.2])

    monkeypatch.setattr("cherab.imas.ids.common.slice._slice_via_memory_backend", _fake_reslice)

    with pytest.warns(RuntimeWarning, match="re-slicing via the IMAS memory backend"):
        ids = get_ids_time_slice(entry, "radiation", time=0.2)

    assert len(ids.time) == 1
    assert ids.time[0] == pytest.approx(0.2)


def test_get_ids_time_slice_warns_when_multi_time_reslice_fails(monkeypatch) -> None:
    entry = _FallbackOnlyEntry(_FakeIDS([0.1, 0.2, 0.3]))

    def _failing_reslice(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("cherab.imas.ids.common.slice._slice_via_memory_backend", _failing_reslice)

    with pytest.warns(RuntimeWarning, match="Returning 'radiation' IDS with 3 time slices"):
        ids = get_ids_time_slice(entry, "radiation", time=0.2)

    assert len(ids.time) == 3
