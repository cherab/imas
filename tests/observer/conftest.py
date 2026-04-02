import shutil
from pathlib import Path

import pytest

from cherab.imas.datasets import bolometer_moc


@pytest.fixture(scope="session")
def path_bolometer_moc(tmp_path_factory) -> str:
    """Fixture to provide the path to a sample bolometer IMAS dataset."""
    path = Path(bolometer_moc())
    tmp_path = tmp_path_factory.mktemp("cherab-imas-data")
    shutil.copy2(path, tmp_path)
    return str(tmp_path / path.name)
