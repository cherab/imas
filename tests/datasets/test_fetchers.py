import pytest

from cherab.imas.datasets._fetchers import fetch_data
from cherab.imas.datasets._registry import method_files_map


def test_fetchers_missing_pooch():
    with pytest.raises(ImportError):
        _ = fetch_data("sample_data", data_fetcher=None)


def test_fetchers_without_progress():
    data_path = fetch_data(method_files_map["iter_jintrac"][0], show_progress=False)
    assert method_files_map["iter_jintrac"][0] in data_path


def test_fetchers_error_handling():
    with pytest.raises(RuntimeError):
        _ = fetch_data("non_existent_dataset")
