#!/usr/bin/env -S pixi run python
"""Full plasma profile plotting demo for ITER JINTRAC scenarios.

This demo creates a blended Plasma object from the core and edge profiles using the equilibrium.
Then samples and plots the quantities.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm

from cherab.core.math import sample3d, sample3d_grid, samplevector2d, samplevector3d_grid
from cherab.imas.datasets import iter_jintrac
from cherab.imas.plasma import load_equilibrium, load_plasma
from cherab.tools.equilibrium import plot_equilibrium

plt.ion()


def plot_quantity(quantity, extent, title="", logscale=False, symmetric=False):
    """Make a 2D plot of quantity, with a title, optionally on a log scale."""

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

# sampling range
xl, xu = 4.0, 8.5
zl, zu = -4.5, 4.6
nz = 911
nx = 451
extent = [xl, xu, zl, zu]

# Load and plot equilibrium
equilibrium, psi_interpolator = load_equilibrium(iter_jintrac(), "r", with_psi_interpolator=True)
plot_equilibrium(equilibrium)
plt.gcf().savefig(plots_path / "equilibrium.png", dpi=200)

# Sample and plot magnetic field
plot_velocity = False
b_field = equilibrium.b_field
try:
    xsamp, zsamp, b = samplevector2d(b_field, (xl, xu, nx), (zl, zu, nz))
    b_length = np.sqrt((b * b).sum(2))
    fig = plot_quantity(b[:, :, 0], extent, title="Brad [T]", symmetric=True)
    fig.savefig(plots_path / "brad.png", dpi=200)
    fig = plot_quantity(b[:, :, 1], extent, title="Btor [T]")
    fig.savefig(plots_path / "btor.png", dpi=200)
    fig = plot_quantity(b[:, :, 2], extent, title="Bz [T]", symmetric=True)
    fig.savefig(plots_path / "bz.png", dpi=200)
    radial_vector = np.zeros_like(b)
    radial_vector[:, :, 0] = -b[:, :, 2]
    radial_vector[:, :, 2] = b[:, :, 0]
    radial_vector /= np.sqrt((radial_vector * radial_vector).sum(2))[:, :, None]
    poloidal_vector = np.copy(b)
    poloidal_vector[:, :, 1] = 0
    poloidal_vector /= np.sqrt((poloidal_vector * poloidal_vector).sum(2))[:, :, None]
    plot_velocity = True
except ValueError:
    print(
        "Unable to sample magnetic field. "
        + "The magnetic field grid does not cover the edge region."
    )

# Load edge plasma
plasma = load_plasma(iter_jintrac(), "r", equilibrium=equilibrium)

# Sample and plot electron profiles
xsamp, _, zsamp, ne_plasma = sample3d(
    plasma.electron_distribution.density, (xl, xu, nx), (0, 0, 1), (zl, zu, nz)
)
ne_plasma = ne_plasma.squeeze()
fig = plot_quantity(ne_plasma, extent, title="ne [m-3]", logscale=True)
fig.savefig(plots_path / "ne.png", dpi=200)

te_plasma = sample3d_grid(
    plasma.electron_distribution.effective_temperature, xsamp, [0], zsamp
).squeeze()
fig = plot_quantity(te_plasma, extent, title="Te [eV]", logscale=True)
fig.savefig(plots_path / "te.png", dpi=200)

if plot_velocity:
    electron_velocity = samplevector3d_grid(
        plasma.electron_distribution.bulk_velocity, xsamp, [0], zsamp
    ).squeeze()
    electron_velocity_par = (electron_velocity * b).sum(2) / b_length
    fig = plot_quantity(
        electron_velocity_par,
        extent,
        title="Electron Vpar [m/s]",
        symmetric=True,
    )
    fig.savefig(plots_path / "electron_vpar.png", dpi=200)

# Sample and plot ion and neutral profiles
for species in plasma.composition:
    density = sample3d_grid(species.distribution.density, xsamp, [0], zsamp).squeeze()
    fig = plot_quantity(
        density,
        extent,
        title=f"{species.element.symbol}{species.charge} density [m-3]",
        logscale=True,
    )
    fig.savefig(
        plots_path / f"{species.element.symbol}{species.charge}_density.png",
        dpi=200,
    )

    if species.element.atomic_number == 1:
        temperature = sample3d_grid(
            species.distribution.effective_temperature, xsamp, [0], zsamp
        ).squeeze()
        fig = plot_quantity(
            temperature,
            extent,
            title=f"{species.element.symbol}{species.charge} temperature [eV]",
            logscale=True,
        )
        fig.savefig(
            plots_path / f"{species.element.symbol}{species.charge}_temperature.png",
            dpi=200,
        )

        if plot_velocity:
            velocity = samplevector3d_grid(
                species.distribution.bulk_velocity, xsamp, [0], zsamp
            ).squeeze()

            if species.charge:
                vpar = (velocity * b).sum(2) / b_length
                fig = plot_quantity(
                    vpar,
                    extent,
                    title=f"{species.element.symbol}{species.charge} Vpar [m/s]",
                    symmetric=True,
                )
                fig.savefig(
                    plots_path / f"{species.element.symbol}{species.charge}_vpar.png",
                    dpi=200,
                )
            else:
                vrad = (velocity * radial_vector).sum(2)
                vpol = (velocity * poloidal_vector).sum(2)
                vtor = velocity[:, :, 1]
                fig = plot_quantity(
                    vrad,
                    extent,
                    title=f"{species.element.symbol}{species.charge} Vrad [m/s]",
                    symmetric=True,
                )
                fig.savefig(
                    plots_path / f"{species.element.symbol}{species.charge}_vrad.png",
                    dpi=200,
                )
                fig = plot_quantity(
                    vpol,
                    extent,
                    title=f"{species.element.symbol}{species.charge} Vpol [m/s]",
                    symmetric=True,
                )
                fig.savefig(
                    plots_path / f"{species.element.symbol}{species.charge}_vpol.png",
                    dpi=200,
                )
                fig = plot_quantity(
                    vtor,
                    extent,
                    title=f"{species.element.symbol}{species.charge} Vtor [m/s]",
                    symmetric=True,
                )
                fig.savefig(
                    plots_path / f"{species.element.symbol}{species.charge}_vtor.png",
                    dpi=200,
                )
