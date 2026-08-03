import shutil
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import pytest
from imas import DBEntry
from imas.ids_defs import MEMORY_BACKEND

from cherab.core.atomic.elements import neon
from cherab.imas.datasets import (
    iter_jintrac,
    iter_jintrac_radiation_values,
    iter_jorek,
    iter_solps,
)
from cherab.openadas import OpenADAS
from cherab.openadas.repository import populate


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--force-imas-memory-backend-tests",
        action="store_true",
        default=False,
        help="Run tests marked 'requires_imas_memory_backend' even if the IMAS memory backend probe fails.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_imas_memory_backend: test requires a working IMAS memory backend.",
    )


@lru_cache(maxsize=1)
def _probe_imas_memory_backend() -> tuple[bool, str]:
    token = uuid4().int
    entry = None
    try:
        entry = DBEntry(
            MEMORY_BACKEND,
            f"cherab_pytest_{token & 0xFFFF:04x}",
            1 + token % 2_000_000_000,
            1 + (token >> 31) % 2_000_000_000,
        )
        entry.create()
    except Exception as exc:
        return False, f"IMAS memory backend is unavailable on this machine: {exc}"
    finally:
        if entry is not None:
            with suppress(Exception):
                entry.close()

    return True, ""


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--force-imas-memory-backend-tests"):
        return

    available, reason = _probe_imas_memory_backend()
    if available:
        return

    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker("requires_imas_memory_backend") is not None:
            item.add_marker(skip_marker)


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


@pytest.fixture(scope="session")
def path_iter_jintrac_radiation_values(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Fixture to provide the path to a synthetic JINTRAC radiation values dataset."""
    path = Path(iter_jintrac_radiation_values())
    return _copy_dataset_to_tmp(path, tmp_path_factory)
