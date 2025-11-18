"""Utility functions for cherab.imas.datasets module."""

import shutil
from collections.abc import Callable, Collection
from pathlib import Path

from ._registry import method_files_map

try:
    import platformdirs
except ImportError:
    platformdirs = None


def _clear_cache(
    datasets: Callable[..., str] | Collection[Callable[..., str]] | None,
    cache_dir: str | Path | None = None,
    method_map: dict[str, list[str]] | None = None,
) -> None:
    if method_map is None:
        # Use Datasets method map
        method_map = method_files_map
    if cache_dir is None:
        # Use default cache_dir path
        if platformdirs is None:
            # platformdirs is pooch dependency
            raise ImportError(
                "Missing optional dependency 'pooch' required for cherab.imas.datasets module. "
                + "Please use pip or conda to install 'pooch'."
            )
        cache_dir = platformdirs.user_cache_dir("cherab/imas")

    # Ensure cache_dir is a Path object
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
        return

    if datasets is None:
        print(f"Cleaning the cache directory {cache_dir}!")
        shutil.rmtree(cache_dir)
    else:
        if not isinstance(datasets, Collection):
            # single dataset method passed should be converted to list
            datasets = [
                datasets,
            ]
        for dataset in datasets:
            if not callable(dataset):
                raise TypeError(
                    "Expected a callable dataset method or a list/tuple of the same, "
                    + f"but got {type(dataset)}"
                )
            dataset_name = dataset.__name__  # Name of the dataset method
            if dataset_name not in method_map:
                raise ValueError(
                    f"Dataset method {dataset_name} doesn't exist."
                    + "Please check if the passed dataset is a subset of the following dataset "
                    + f"methods: {list(method_map.keys())}"
                )

            data_files = method_map[dataset_name]
            data_filepaths = [cache_dir / file for file in data_files]
            for data_filepath in data_filepaths:
                if data_filepath.exists():
                    print(f"Cleaning the file {data_filepath.name} for dataset {dataset_name}")
                    data_filepath.unlink()
                else:
                    print(f"Path {data_filepath} doesn't exist. Nothing to clear.")


def clear_cache(
    datasets: Callable[..., str] | Collection[Callable[..., str]] | None = None,
) -> None:
    """Clean the cherab-imas datasets cache directory.

    If a `cherab.imas.datasets` method or a list/tuple of the same is provided, then `clear_cache`
    removes all the data files associated to the passed dataset method callable(s).

    By default, it removes all the cached data files.

    Parameters
    ----------
    datasets
        `cherab.imas.datasets` method or a list/tuple of the same whose cached data files are to be
        removed. If `None`, all the cached data files are removed.

    Examples
    --------
    >>> from cherab.imas import datasets
    >>> data_path = datasets.iter_jintrac()
    >>> datasets.clear_cache([datasets.iter_jintrac])
    Cleaning the file iter_scenario_53298_seq1_DD4.nc for dataset iter_jintrac

    If no argument is passed, all cached dataset files are removed.

    >>> datasets.clear_cache()
    Cleaning the cache directory ~/Library/Caches/cherab/imas!
    """
    _clear_cache(datasets)
