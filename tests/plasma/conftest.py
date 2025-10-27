import shutil
from pathlib import Path

import pytest

from cherab.imas.datasets import iter_jintrac


@pytest.fixture(scope="session")
def path_iter_jintrac(tmp_path_factory) -> str:
    """Fixture to provide the path to a sample JINTRAC IMAS dataset."""
    path = Path(iter_jintrac())
    tmp_path = tmp_path_factory.mktemp("cherab-imas-data")
    shutil.move(path, tmp_path)
    return str(tmp_path / path.name)
