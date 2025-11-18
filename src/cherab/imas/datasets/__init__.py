r"""Sample Dataset utilities for fetching and processing data.

Usage of Datasets
=================

CHERAB-IMAS dataset methods can be simply called as follows: ``'<dataset-name>()'``
This downloads the dataset files over the network once, and saves the cache,
before returning the path to the downloaded data file.

How dataset retrieval and storage works
=======================================

The dataset files are stored in public repositories on
Zenodo: https://doi.org/10.5281/zenodo.17062700.
The `cherab.imas.datasets` submodule utilizes
and depends on `Pooch <https://www.fatiando.org/pooch/latest/>`_, a Python
package built to simplify fetching data files. Pooch uses these repos to
retrieve the respective dataset files when calling the dataset function.

Datasets available in CHERAB-IMAS are sample datasets resulting from ITER JINTRAC, SOLPS, JOREK,
etc. simulations, and are intended for testing and demonstration purposes.

A registry of all the datasets, essentially a mapping of filenames with their
MD5 hash are maintained, which Pooch uses to handle and verify
the downloads on function call. After downloading the dataset once, the files
are saved in the system cache directory under ``'cherab/imas'``.

Dataset cache locations may vary on different platforms.

For macOS::

    "~/Library/Caches/cherab/imas"

For Linux and other Unix-like platforms::

    "~/.cache/cherab/imas"  # or the value of the XDG_CACHE_HOME env var, if defined

For Windows::

    "C:\Users\<user>\AppData\Local\<AppAuthor>\cherab\imas\Cache"

In environments with constrained network connectivity for various security
reasons or on systems without continuous internet connections, one may manually
load the cache of the datasets by placing the contents of the dataset repo in
the above mentioned cache directory to avoid fetching dataset errors without
the internet connectivity.
"""

from ._fetchers import iter_jintrac, iter_jorek, iter_solps
from ._utils import clear_cache

__all__ = ["iter_jintrac", "iter_solps", "iter_jorek", "clear_cache"]
