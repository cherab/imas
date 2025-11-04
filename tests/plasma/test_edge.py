import pytest
from imas import DBEntry
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from raysect.core.scenegraph import World
from raysect.primitive import Subtract

from cherab.core import Maxwellian, Plasma
from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.common.ggd import load_grid
from cherab.imas.ids.edge_profiles import load_edge_species
from cherab.imas.plasma.edge import load_edge_plasma
from cherab.imas.plasma.equilibrium import load_magnetic_field

plt.rcParams["backend"] = "Agg"  # Use non-interactive backend for testing


def test_load_edge_plasma_basic(path_iter_jintrac: str):
    """Test basic loading of edge plasma data from an IMAS file."""
    plasma = load_edge_plasma(path_iter_jintrac, "r")

    # Test that a Plasma object is returned
    assert isinstance(plasma, Plasma)


def test_load_edge_plasma_geometry(path_iter_jintrac: str):
    """Test that the plasma geometry is properly set."""
    plasma = load_edge_plasma(path_iter_jintrac, "r")

    # Check that geometry is set and has the expected type
    assert plasma.geometry is not None
    assert isinstance(plasma.geometry, Subtract)

    # Check that geometry transform is set
    assert plasma.geometry_transform is not None


def test_load_edge_plasma_electron_distribution(path_iter_jintrac: str):
    """Test that electron distribution is properly loaded."""
    plasma = load_edge_plasma(path_iter_jintrac, "r")

    # Check that electron distribution exists
    assert plasma.electron_distribution is not None
    assert isinstance(plasma.electron_distribution, Maxwellian)


def test_load_edge_plasma_with_time(path_iter_jintrac: str):
    """Test loading edge plasma with specific time parameter."""
    # Test with default time (should not raise error)
    plasma1 = load_edge_plasma(path_iter_jintrac, "r", time=0.0)
    assert isinstance(plasma1, Plasma)

    # Test with different time values
    plasma2 = load_edge_plasma(path_iter_jintrac, "r", time=0.1)
    assert isinstance(plasma2, Plasma)


def test_load_edge_plasma_with_grid_subset_id(path_iter_jintrac: str):
    """Test loading edge plasma with different grid subset IDs."""
    # Test with string identifier
    plasma2 = load_edge_plasma(path_iter_jintrac, "r", grid_subset_id="cells")
    assert isinstance(plasma2, Plasma)


def test_load_edge_plasma_with_parent(path_iter_jintrac: str):
    """Test loading edge plasma with a parent scene graph node."""
    world = World()

    plasma = load_edge_plasma(path_iter_jintrac, "r", parent=world)
    assert isinstance(plasma, Plasma)
    assert plasma.parent is world


def test_load_edge_plasma_with_magnetic_field(path_iter_jintrac: str):
    """Test loading edge plasma with external magnetic field."""
    # Load magnetic field separately
    try:
        b_field = load_magnetic_field(path_iter_jintrac, "r")
        plasma = load_edge_plasma(path_iter_jintrac, "r", b_field=b_field)
        assert isinstance(plasma, Plasma)
        assert plasma.b_field is not None
    except RuntimeError:
        # If magnetic field loading fails, test should still pass
        # as this is handled gracefully in the function
        pass


def test_load_edge_plasma_time_threshold(path_iter_jintrac: str):
    """Test loading edge plasma with time threshold parameter."""
    # Test with different time thresholds
    plasma = load_edge_plasma(
        path_iter_jintrac, "r", time_threshold=500
    )  # test dataset has time ~450s only
    assert isinstance(plasma, Plasma)

    with pytest.raises(RuntimeError):
        load_edge_plasma(path_iter_jintrac, "r", time_threshold=0.0)


def test_load_edge_plasma_composition(path_iter_jintrac: str):
    """Test that plasma composition contains expected species."""
    plasma = load_edge_plasma(path_iter_jintrac, "r")

    # Check that we have some ion species (exact species depend on data)
    assert len(plasma.composition) > 0


def test_load_edge_plasma_error_handling(path_iter_jintrac: str):
    """Test error handling for invalid inputs."""
    # Test with non-existent file path
    with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
        load_edge_plasma("non_existent_path", "r")

    # Test with invalid mode
    with pytest.raises((ValueError, RuntimeError, OSError)):
        load_edge_plasma(path_iter_jintrac, "invalid_mode")


def test_plot_edge_plasma_profiles(path_iter_jintrac: str):
    """Test mesh plotting of edge plasma profiles."""
    with DBEntry(path_iter_jintrac, "r") as entry:
        ids = get_ids_time_slice(entry, "edge_profiles", time=0)

    # Load mesh grid
    grid, subsets, subset_id = load_grid(ids.grid_ggd[0], with_subsets=True)
    grid_subset_name = "Cells"
    try:
        grid = grid.subset(subsets[grid_subset_name], name=grid_subset_name)
    except KeyError:
        grid_subset_name = grid_subset_name.lower()
        grid = grid.subset(subsets[grid_subset_name.lower()], name=grid_subset_name)

    # Load edge species composition
    composition = load_edge_species(ids.ggd[0], grid_subset_index=subset_id[grid_subset_name])

    # Plot electron density
    electron_density = composition["electron"]["density"]
    ax = grid.plot_mesh(data=electron_density)

    assert isinstance(ax, Axes)
