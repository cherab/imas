"""This module provides functionality to fetch IMAS sample datasets."""

from ...imas import __version__
from ._registry import registry

try:
    import pooch

except ImportError:
    pooch = None
    data_fetcher = None
    progress = None
else:
    data_fetcher = pooch.create(
        path=pooch.os_cache("cherab/imas"),
        base_url="doi:10.5281/zenodo.17062699",
        registry=registry,
    )

try:
    from ._progress import PoochRichProgress
except ImportError:
    progress_available = False
else:
    progress_available = True


def fetch_data(dataset_name: str, data_fetcher=data_fetcher) -> str:
    if data_fetcher is None:
        raise ImportError(
            "Missing optional dependency 'pooch' required for cherab.imas.datasets module. "
            + "Please use pip or conda to install 'pooch'."
        )
    if progress_available:
        progress = PoochRichProgress(filename=dataset_name)
    else:
        progress = None

    downloader = pooch.DOIDownloader(
        headers={"User-Agent": f"CHERAB-IMAS {__version__}"}, progressbar=progress
    )

    #  The "fetch" method returns the full path to the downloaded data file.
    try:
        return data_fetcher.fetch(dataset_name, downloader=downloader)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch dataset '{dataset_name}'.") from e
    finally:
        # Ensure progress bar is closed properly
        if progress is not None:
            progress.close()


def iter_jintrac() -> str:
    """Fetche and return the path to the ITER JINTRAC sample dataset.

    Returns
    -------
    str
        Path to the downloaded ITER JINTRAC sample dataset file.

    Examples
    --------
    >>> from cherab.imas import datasets
    >>> data_path = datasets.iter_jintrac()
    >>> data_path
    '.../cherab/imas/iter_jintrac/iter_scenario_53298_seq1_DD4.nc'
    """
    return fetch_data("iter_scenario_53298_seq1_DD4.nc")


def iter_solps() -> str:
    """Fetche and return the path to the ITER SOLPS sample dataset.

    Returns
    -------
    str
        Path to the downloaded ITER SOLPS sample dataset file.

    Examples
    --------
    >>> from cherab.imas import datasets
    >>> data_path = datasets.iter_solps()
    >>> data_path
    '.../cherab/imas/iter_solps/iter_scenario_123364_1.nc'
    """
    return fetch_data("iter_scenario_123364_1.nc")


def iter_jorek() -> str:
    """Fetche and return the path to the ITER JOREK sample dataset.

    Returns
    -------
    str
        Path to the downloaded ITER JOREK sample dataset file.

    Examples
    --------
    >>> from cherab.imas import datasets
    >>> data_path = datasets.iter_jorek()
    >>> data_path
    '.../cherab/imas/iter_jorek/iter_disruption_113112_1.nc'
    """
    return fetch_data("iter_disruption_113112_1.nc")
