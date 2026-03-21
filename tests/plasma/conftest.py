import shutil
from pathlib import Path

import pytest

from cherab.imas.datasets import iter_jintrac, iter_jorek, iter_solps


@pytest.fixture(scope="session")
def path_iter_jintrac(tmp_path_factory) -> str:
    """Fixture to provide the path to a sample JINTRAC IMAS dataset."""
    path = Path(iter_jintrac())
    tmp_path = tmp_path_factory.mktemp("cherab-imas-data")
    shutil.move(path, tmp_path)
    return str(tmp_path / path.name)


@pytest.fixture(scope="session")
def path_iter_solps(tmp_path_factory) -> str:
    """Fixture to provide the path to a sample SOLPS IMAS dataset."""
    path = Path(iter_solps())
    tmp_path = tmp_path_factory.mktemp("cherab-imas-data")
    shutil.move(path, tmp_path)
    return str(tmp_path / path.name)


@pytest.fixture(scope="session")
def path_iter_jorek(tmp_path_factory) -> str:
    """Fixture to provide the path to a sample JOREK IMAS dataset."""
    path = Path(iter_jorek())
    tmp_path = tmp_path_factory.mktemp("cherab-imas-data")
    shutil.move(path, tmp_path)
    return str(tmp_path / path.name)
