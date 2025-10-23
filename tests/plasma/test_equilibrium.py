from matplotlib import pyplot as plt
from raysect.core.math.function.vector3d import FloatToVector3DFunction2D

from cherab.imas.datasets import iter_jintrac
from cherab.imas.plasma.equilibrium import load_equilibrium, load_magnetic_field
from cherab.tools.equilibrium import EFITEquilibrium, plot_equilibrium

plt.rcParams["backend"] = "Agg"  # Use non-interactive backend for testing


def test_load_equilibrium():
    """Test loading of equilibrium data from an IMAS file."""
    path = iter_jintrac()
    equilibrium = load_equilibrium(path, "r")

    # Test that equilibrium object is returned
    assert isinstance(equilibrium, EFITEquilibrium)

    # Test plotting function
    plot_equilibrium(equilibrium)


def test_load_magnetic_field():
    """Test loading of magnetic field data from an IMAS file."""
    path = iter_jintrac()
    magnetic_field = load_magnetic_field(path, "r")

    # Test that Vector3DFunction2D object is returned
    assert isinstance(magnetic_field, FloatToVector3DFunction2D)
