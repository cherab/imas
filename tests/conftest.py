import shutil
from pathlib import Path

import pytest

from cherab.core.atomic.elements import neon
from cherab.imas.datasets import iter_jintrac, iter_jorek, iter_solps
from cherab.openadas import OpenADAS
from cherab.openadas.repository import populate


@pytest.fixture(scope="session", autouse=True)
def populate_openadas_repository():
    """Fixture to populate the OpenADAS repository before running tests."""
    adas = OpenADAS(missing_rates_return_null=False)
    try:
        adas.ionisation_rate(neon, 1)
    except Exception:
        print("Populating OpenADAS repository...")
        populate()


def _copy_dataset_to_tmp(path: Path, tmp_path_factory: pytest.TempPathFactory) -> str:
    """Copy a dataset into a temporary location, handling files and directories."""
    tmp_path = tmp_path_factory.mktemp("cherab-imas-data")
    target = tmp_path / path.name
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)
    return str(target)


@pytest.fixture(scope="session")
def path_iter_jintrac(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Fixture to provide the path to a sample JINTRAC IMAS dataset."""
    path = Path(iter_jintrac())
    return _copy_dataset_to_tmp(path, tmp_path_factory)


@pytest.fixture(scope="session")
def path_iter_solps(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Fixture to provide the path to a sample SOLPS IMAS dataset."""
    path = Path(iter_solps())
    return _copy_dataset_to_tmp(path, tmp_path_factory)


@pytest.fixture(scope="session")
def path_iter_jorek(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Fixture to provide the path to a sample JOREK IMAS dataset."""
    path = Path(iter_jorek())
    return _copy_dataset_to_tmp(path, tmp_path_factory)
