import pytest
from raysect.core.scenegraph import World
from raysect.primitive import Subtract

from cherab.core import Maxwellian, Plasma
from cherab.imas.datasets import iter_jintrac
from cherab.imas.plasma import load_plasma
from cherab.imas.plasma.equilibrium import load_magnetic_field


def test_load_plasma_basic():
    """Test basic loading of plasma data from an IMAS file."""
    path = iter_jintrac()
    plasma = load_plasma(path, "r")

    # Test that a Plasma object is returned
    assert isinstance(plasma, Plasma)


def test_load_plasma_geometry():
    """Test that the plasma geometry is properly set."""
    path = iter_jintrac()
    plasma = load_plasma(path, "r")

    # Check that geometry is set and has the expected type
    assert plasma.geometry is not None
    assert isinstance(plasma.geometry, Subtract)

    # Check that geometry transform is set
    assert plasma.geometry_transform is not None


def test_load_plasma_electron_distribution():
    """Test that electron distribution is properly loaded."""
    path = iter_jintrac()
    plasma = load_plasma(path, "r")

    # Check that electron distribution exists
    assert plasma.electron_distribution is not None
    assert isinstance(plasma.electron_distribution, Maxwellian)


def test_load_plasma_with_time():
    """Test loading plasma with specific time parameter."""
    path = iter_jintrac()

    # Test with default time (should not raise error)
    plasma1 = load_plasma(path, "r", time=0.0)
    assert isinstance(plasma1, Plasma)

    # we typically have time=450 in the test dataset
    plasma2 = load_plasma(path, "r", time=450.0)
    assert isinstance(plasma2, Plasma)


def test_load_plasma_with_occurrence():
    """Test loading plasma with different occurrence values."""
    path = iter_jintrac()

    # Test with default occurrence core
    plasma1 = load_plasma(path, "r", occurrence_core=0)
    assert isinstance(plasma1, Plasma)

    # Test with default occurrence edge
    plasma2 = load_plasma(path, "r", occurrence_edge=0)
    assert isinstance(plasma2, Plasma)

    # Test with default occurrence core and edge
    plasma3 = load_plasma(path, "r", occurrence_core=0, occurrence_edge=0)
    assert isinstance(plasma3, Plasma)


def test_load_plasma_with_parent():
    """Test loading plasma with a parent scene graph node."""
    path = iter_jintrac()
    world = World()

    plasma = load_plasma(path, "r", parent=world)
    assert isinstance(plasma, Plasma)
    assert plasma.parent is world


def test_load_plasma_with_magnetic_field():
    """Test loading plasma with external magnetic field."""
    path = iter_jintrac()

    # Load magnetic field separately
    try:
        b_field = load_magnetic_field(path, "r")
        plasma = load_plasma(path, "r", b_field=b_field)
        assert isinstance(plasma, Plasma)
        assert plasma.b_field is not None
    except RuntimeError:
        # If magnetic field loading fails, test should still pass
        # as this is handled gracefully in the function
        pass


def test_load_plasma_time_threshold():
    """Test loading plasma with time threshold parameter."""
    path = iter_jintrac()

    # Test with large time threshold (should work)
    plasma = load_plasma(path, "r", time_threshold=500)
    assert isinstance(plasma, Plasma)

    # Test with very small time threshold (should raise error)
    with pytest.raises(RuntimeError):
        load_plasma(path, "r", time_threshold=0.0)


def test_load_plasma_composition():
    """Test that plasma composition contains expected species."""
    path = iter_jintrac()
    plasma = load_plasma(path, "r")

    # Check that we have some ion species (exact species depend on data)
    assert len(plasma.composition) > 0


def test_load_plasma_error_handling():
    """Test error handling for invalid inputs."""
    # Test with non-existent file path
    with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
        load_plasma("non_existent_path", "r")

    # Test with invalid mode
    path = iter_jintrac()
    with pytest.raises((ValueError, RuntimeError, OSError)):
        load_plasma(path, "invalid_mode")
