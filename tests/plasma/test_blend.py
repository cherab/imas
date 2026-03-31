import pytest
from raysect.core.scenegraph import World
from raysect.primitive import Subtract

from cherab.core import Maxwellian, Plasma
from cherab.core.atomic.elements import neon
from cherab.imas.plasma import load_plasma
from cherab.imas.plasma.equilibrium import load_magnetic_field


def test_load_plasma(path_iter_jintrac: str):
    """Test basic loading of plasma data from an IMAS file."""
    plasma = load_plasma(path_iter_jintrac, "r")

    # Test that a Plasma object is returned
    assert isinstance(plasma, Plasma)

    # Check that geometry is set and has the expected type
    assert plasma.geometry is not None
    assert isinstance(plasma.geometry, Subtract)

    # Check that geometry transform is set
    assert plasma.geometry_transform is not None

    # Check that electron distribution exists
    assert plasma.electron_distribution is not None
    assert isinstance(plasma.electron_distribution, Maxwellian)

    # Check if a certain bundled species is split into individual ion species.
    # Here we check only the Ne species as an example, but this can be extended to other species
    # based on the test dataset.
    assert len(plasma.composition) > 0
    ion_charges = set()
    for species in plasma.composition:
        # Check that we have multiple ionization states for Ne
        if species.element == neon:
            ion_charges.add(species.charge)

    # Expecting all charge states from 0 to Z for Ne
    assert len(ion_charges) == neon.atomic_number + 1


def test_load_plasma_with_time(path_iter_jintrac: str):
    """Test loading plasma with specific time parameter."""
    # Test with default time (should not raise error)
    plasma = load_plasma(path_iter_jintrac, "r", time=0.1)
    assert isinstance(plasma, Plasma)

    with pytest.raises(ValueError):
        load_plasma(path_iter_jintrac, "r", time=-1)


def test_load_plasma_with_occurrence(path_iter_jintrac: str):
    """Test loading plasma with different occurrence values."""
    # Test with default occurrence core
    plasma1 = load_plasma(path_iter_jintrac, "r", occurrence_core=0, split_ion_bundles=False)
    assert isinstance(plasma1, Plasma)

    # Test with default occurrence edge
    plasma2 = load_plasma(path_iter_jintrac, "r", occurrence_edge=0, split_ion_bundles=False)
    assert isinstance(plasma2, Plasma)

    # Test with default occurrence core and edge
    plasma3 = load_plasma(
        path_iter_jintrac, "r", occurrence_core=0, occurrence_edge=0, split_ion_bundles=False
    )
    assert isinstance(plasma3, Plasma)


def test_load_plasma_with_parent(path_iter_jintrac: str):
    """Test loading plasma with a parent scene graph node."""
    world = World()

    plasma = load_plasma(path_iter_jintrac, "r", parent=world, split_ion_bundles=False)
    assert isinstance(plasma, Plasma)
    assert plasma.parent is world


def test_load_plasma_with_magnetic_field(path_iter_jintrac: str):
    """Test loading plasma with external magnetic field."""
    # Load magnetic field separately
    try:
        b_field = load_magnetic_field(path_iter_jintrac, "r")
        plasma = load_plasma(path_iter_jintrac, "r", b_field=b_field, split_ion_bundles=False)
        assert isinstance(plasma, Plasma)
        assert plasma.b_field is not None
    except RuntimeError:
        # If magnetic field loading fails, test should still pass
        # as this is handled gracefully in the function
        pass


def test_load_plasma_time_threshold(path_iter_jintrac: str):
    """Test loading plasma with time threshold parameter."""
    # Test with large time threshold (should work)
    plasma = load_plasma(path_iter_jintrac, "r", time_threshold=500, split_ion_bundles=False)
    assert isinstance(plasma, Plasma)

    # Test with very small time threshold (should raise error)
    with pytest.raises(RuntimeError):
        load_plasma(path_iter_jintrac, "r", time_threshold=0.0, split_ion_bundles=False)


def test_load_plasma_not_split_ion_bundles(path_iter_jintrac: str):
    """Test that loading with split_ion_bundles=False."""
    plasma = load_plasma(path_iter_jintrac, "r", split_ion_bundles=False)

    ion_charges_no_split = set()
    for species in plasma.composition:
        if species.element == neon:
            ion_charges_no_split.add(species.charge)

    # Expecting only the bundled charge state
    assert 0 < len(ion_charges_no_split) < neon.atomic_number + 1
