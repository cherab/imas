#!/usr/bin/env -S pixi run python
"""Core plasma profile plotting demo for ITER scenarios.

This demo creates a Plasma object from the core profiles and equilibrium. Then samples and plots the
quantities.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure

from cherab.core.math import sample3d_grid, samplevector2d
from cherab.imas.datasets import iter_jintrac
from cherab.imas.plasma import load_core_plasma, load_equilibrium
from cherab.tools.equilibrium import plot_equilibrium

plt.ion()


def plot_quantity(quantity, extent, title="", logscale=False, symmetric=False) -> Figure:
    """Make a 2D plot of quantity, with a title, optionally on a log scale.

    Parameters
    ----------
    quantity
        2D array of the quantity to plot.
    extent
        The extent of the plot in the form [xmin, xmax, ymin, ymax].
    title
        The title of the plot.
    logscale
        Whether to use a logarithmic scale for the color map.
    symmetric
        Whether to use a symmetric color map around zero.

    Returns
    -------
    The matplotlib Figure object containing the plot.
    """
    fig = plt.figure(figsize=(4.0, 6.0), layout="constrained")
    ax = fig.add_subplot(111)
    if logscale:
        # Plot lowest values (mainly 0's) on linear map, as log(0) = -inf.
        linthresh = np.percentile(np.unique(quantity), 1)
        norm = SymLogNorm(linthresh=float(max(linthresh, 1.0e-10 * quantity.max())), base=10)
    else:
        norm = None
    # Sampled data is indexed as quantity(x, y), but matplotlib's imshow
    # expects quantity(y, x).
    if symmetric and not logscale:
        vmax = np.abs(quantity).max()
        image = ax.imshow(
            quantity.T, extent=extent, origin="lower", vmin=-vmax, vmax=vmax, cmap="coolwarm"
        )
    else:
        image = ax.imshow(quantity.T, extent=extent, origin="lower", norm=norm, cmap="gnuplot")
    fig.colorbar(image, aspect=50)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)

    return fig


demos_path = Path(__file__).parent
plots_path = demos_path / "plots"
plots_path.mkdir(exist_ok=True)

# Load and plot equilibrium
path = iter_jintrac()
equilibrium, psi_interpolator = load_equilibrium(path, "r", with_psi_interpolator=True)
plot_equilibrium(equilibrium)
plt.gcf().savefig(plots_path / "equilibrium.png", dpi=200)

# sampling range
xl, xu = equilibrium.r_range
zl, zu = equilibrium.z_range
nz = 600
nx = 300
extent = [xl, xu, zl, zu]

# Sample and plot magnetic field
xsamp, zsamp, b = samplevector2d(equilibrium.b_field, (xl, xu, nx), (zl, zu, nz))
b_length = np.sqrt((b * b).sum(2))
fig = plot_quantity(b[:, :, 0], extent, title="Brad [T]", symmetric=True)
fig.savefig(plots_path / "brad.png", dpi=200)
fig = plot_quantity(b[:, :, 1], extent, title="Btor [T]")
fig.savefig(plots_path / "btor.png", dpi=200)
fig = plot_quantity(b[:, :, 2], extent, title="Bz [T]", symmetric=True)
fig.savefig(plots_path / "bz.png", dpi=200)

# Load core plasma
plasma = load_core_plasma(path, "r", equilibrium=equilibrium, psi_interpolator=psi_interpolator)

# Sample and plot electron profiles
ne_plasma = sample3d_grid(plasma.electron_distribution.density, xsamp, [0], zsamp)
ne_plasma = ne_plasma.squeeze()
fig = plot_quantity(ne_plasma, extent, title="ne [m-3]")
fig.savefig(plots_path / "core_ne.png", dpi=200)

te_plasma = sample3d_grid(
    plasma.electron_distribution.effective_temperature, xsamp, [0], zsamp
).squeeze()
fig = plot_quantity(te_plasma, extent, title="Te [eV]")
fig.savefig(plots_path / "core_te.png", dpi=200)

# Sample and plot ion and neutral profiles
for species in plasma.composition:
    density = sample3d_grid(species.distribution.density, xsamp, [0], zsamp).squeeze()
    if np.any(density):
        fig = plot_quantity(
            density,
            extent,
            title=f"{species.element.symbol}{species.charge} density [m-3]",
        )
        fig.savefig(
            plots_path / f"core_{species.element.symbol}{species.charge}_density.png",
            dpi=200,
        )

    if species.element.atomic_number == 1:
        temperature = sample3d_grid(
            species.distribution.effective_temperature, xsamp, [0], zsamp
        ).squeeze()
        if np.any(temperature):
            fig = plot_quantity(
                temperature,
                extent,
                title=f"{species.element.symbol}{species.charge} temperature [eV]",
            )
            fig.savefig(
                plots_path / f"core_{species.element.symbol}{species.charge}_temperature.png",
                dpi=200,
            )
