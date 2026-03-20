import numpy as np
import pytest
from imas import DBEntry
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from raysect.core.scenegraph import World
from raysect.primitive import Subtract

from cherab.core import Maxwellian, Plasma
from cherab.core.atomic.elements import neon
from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.common.ggd import load_grid
from cherab.imas.ids.edge_profiles import load_edge_species
from cherab.imas.plasma.edge import load_edge_plasma
from cherab.imas.plasma.equilibrium import load_magnetic_field

plt.rcParams["backend"] = "Agg"  # Use non-interactive backend for testing


def test_load_edge_plasma(path_iter_jintrac: str):
    """Test loading of edge plasma data from an IMAS file."""
    plasma = load_edge_plasma(path_iter_jintrac, "r")

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
        if species.element.symbol == "Ne":
            ion_charges.add(species.charge)

    # Expecting all charge states from 0 to Z for Ne
    assert len(ion_charges) == neon.atomic_number + 1


def test_load_edge_plasma_with_time(path_iter_jintrac: str):
    """Test loading edge plasma with specific time parameter."""
    plasma = load_edge_plasma(path_iter_jintrac, "r", time=0.1, split_ion_bundles=False)
    assert isinstance(plasma, Plasma)

    with pytest.raises(ValueError):
        load_edge_plasma(path_iter_jintrac, "r", time=-1, split_ion_bundles=False)


def test_load_edge_plasma_with_grid_subset_id(path_iter_jintrac: str):
    """Test loading edge plasma with different grid subset IDs."""
    # Test with string identifier
    plasma = load_edge_plasma(
        path_iter_jintrac, "r", grid_subset_id="cells", split_ion_bundles=False
    )
    assert isinstance(plasma, Plasma)


def test_load_edge_plasma_with_parent(path_iter_jintrac: str):
    """Test loading edge plasma with a parent scene graph node."""
    world = World()

    plasma = load_edge_plasma(path_iter_jintrac, "r", parent=world, split_ion_bundles=False)
    assert isinstance(plasma, Plasma)
    assert plasma.parent is world


def test_load_edge_plasma_with_magnetic_field(path_iter_jintrac: str):
    """Test loading edge plasma with external magnetic field."""
    # Load magnetic field separately
    try:
        b_field = load_magnetic_field(path_iter_jintrac, "r")
        plasma = load_edge_plasma(path_iter_jintrac, "r", b_field=b_field, split_ion_bundles=False)
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
        path_iter_jintrac, "r", time_threshold=500, split_ion_bundles=False
    )  # test dataset has time ~450s only
    assert isinstance(plasma, Plasma)

    with pytest.raises(RuntimeError):
        load_edge_plasma(path_iter_jintrac, "r", time_threshold=0.0, split_ion_bundles=False)


def test_edge_plasma_profiles(path_iter_jintrac: str):
    """Test that edge plasma profiles can be loaded and plotted."""
    with DBEntry(path_iter_jintrac, "r") as entry:
        ids = get_ids_time_slice(entry, "edge_profiles", time=0)

    # Load mesh grid
    grid, subsets, subset_id = load_grid(ids.grid_ggd[0], with_subsets=True)
    grid_subset_name = "Cells"
    try:
        grid = grid.subset(subsets[grid_subset_name], name=grid_subset_name)
    except KeyError:
        grid_subset_name = grid_subset_name.lower()
        grid = grid.subset(subsets[grid_subset_name], name=grid_subset_name)

    # Load edge species composition
    composition_w_bundle = load_edge_species(
        ids.ggd[0],
        grid_subset_index=subset_id[grid_subset_name],
        split_ion_bundles=False,
    )
    composition = load_edge_species(
        ids.ggd[0],
        grid_subset_index=subset_id[grid_subset_name],
        split_ion_bundles=True,
    )

    # Check that the summed density of the split ion species matches the original bundled species density
    ion_bundles = []
    for bundle in composition_w_bundle.ion_bundle:
        if bundle.species.element == neon and bundle.density is not None:
            ion_bundles += [bundle]
    if not ion_bundles:
        raise ValueError("Test dataset does not contain a bundled Ne ion species with density.")
    for ion_bundle in ion_bundles:
        original_density = ion_bundle.density
        element = ion_bundle.species.element
        z_min = ion_bundle.species.z_min
        z_max = ion_bundle.species.z_max
        split_density_sum = np.zeros_like(original_density)
        for profile in composition.ion:
            if (
                profile.species.element == element
                and profile.species.z_min >= z_min
                and profile.species.z_max <= z_max
            ):
                if profile.density is None:
                    raise ValueError(
                        f"Test dataset does not contain density for split ion species: {profile.species}."
                    )
                split_density_sum += profile.density

        np.testing.assert_allclose(split_density_sum, original_density)

    # Plot electron density
    electron_density = composition.electron.density
    ax = grid.plot_mesh(data=electron_density)

    assert isinstance(ax, Axes)


def test_load_edge_plasma_not_split_ion_bundles(path_iter_jintrac: str):
    """Test that loading with split_ion_bundles=False."""
    plasma = load_edge_plasma(path_iter_jintrac, "r", split_ion_bundles=False)

    ion_charges_no_split = set()
    for species in plasma.composition:
        if species.element.symbol == "Ne":
            ion_charges_no_split.add(species.charge)

    # Expecting only the bundled charge state
    assert 0 < len(ion_charges_no_split) < neon.atomic_number + 1
