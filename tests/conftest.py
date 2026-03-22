import pytest

from cherab.core.atomic.elements import neon
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
