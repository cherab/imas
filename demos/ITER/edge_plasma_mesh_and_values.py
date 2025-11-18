#!/usr/bin/env -S pixi run python
"""Edge plasma grid plotting demo for ITER-like scenarios.

This demo reads edge plasma grid and profiles and plots the plasma profiles on the grid.
"""

from pathlib import Path

import imas
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure

from cherab.imas.datasets import iter_jintrac
from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.common.ggd import load_grid
from cherab.imas.ids.edge_profiles import load_edge_species

plt.ion()

grid_subset_name = "Cells"


def plot_grid_quantity(grid, quantity, title="", logscale=False, symmetric=False) -> Figure:
    """Make a 2D plot of a grid quantity, with a title, optionally on a log scale.

    Parameters
    ----------
    grid
        The grid object.
    quantity
        1D array of the quantity to plot on the grid.
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
    ax = grid.plot_mesh(data=quantity)

    if logscale:
        linthresh = np.percentile(np.unique(quantity), 1)
        norm = SymLogNorm(linthresh=float(max(linthresh, 1.0e-10 * quantity.max())), base=10)
        ax.collections[0].set_norm(norm)

    if symmetric:
        vmax = np.abs(quantity.max())
        ax.collections[0].set_clim(-vmax, vmax)
        ax.collections[0].set_cmap("coolwarm")
    else:
        ax.collections[0].set_cmap("gnuplot")

    ax.set_title(title)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    fig = ax.get_figure()
    fig.colorbar(ax.collections[0], ax=ax, aspect=40)
    fig.set_size_inches((4, 6))

    return fig


demos_path = Path(__file__).parent
plots_path = demos_path / "plots"
plots_path.mkdir(exist_ok=True)

with imas.DBEntry(iter_jintrac(), "r") as entry:
    ids = get_ids_time_slice(entry, "edge_profiles", time=0)

grid, subsets, subset_id = load_grid(ids.grid_ggd[0], with_subsets=True)

try:
    grid = grid.subset(subsets[grid_subset_name], name=grid_subset_name)
except KeyError:
    grid_subset_name = grid_subset_name.lower()
    grid = grid.subset(subsets[grid_subset_name], name=grid_subset_name)

composition = load_edge_species(ids.ggd[0], grid_subset_index=subset_id[grid_subset_name])

# Plot mesh
ax = grid.plot_mesh()
ax.set_xlabel("R [m]")
ax.set_ylabel("Z [m]")
fig = ax.get_figure()
fig.set_size_inches((4, 6))
fig.savefig(plots_path / "edge_mesh.png", dpi=300)

# Plot electron profiles
fig = plot_grid_quantity(
    grid, composition["electron"]["density"], title="Electron density [m-3]", logscale=True
)
fig.savefig(plots_path / "edge_mesh_ne.png", dpi=200)

fig = plot_grid_quantity(
    grid, composition["electron"]["temperature"], title="Electron temperature [eV]", logscale=True
)
fig.savefig(plots_path / "edge_mesh_te.png", dpi=200)

vpar = composition["electron"]["velocity_parallel"]
if np.any(vpar):
    fig = plot_grid_quantity(grid, vpar, title="Electron parallel velocity [m/s]", symmetric=True)
    fig.savefig(plots_path / "edge_mesh_electron_vpar.png", dpi=200)

# Plot species profiles
for stype in ("ion", "molecule"):
    for species_id, profiles in composition[stype].items():
        d = {first: second for first, second in species_id}
        charge = int(round(d["z"]))
        if stype == "ion":
            element = d["element"]
            name = d["name"] if len(d["name"]) else f"{element.symbol}{charge}"
        else:
            element = d["elements"][0]
            name = (
                d["name"]
                if len(d["name"])
                else "{}{}".format("".join([el.symbol for el in d["elements"]]), charge)
            )

        fig = plot_grid_quantity(
            grid, profiles["density"], title=f"{name} density [m-3]", logscale=True
        )
        fig.savefig(plots_path / f"edge_mesh_{name}_density.png", dpi=200)

        if element.atomic_number == 1:
            temperature = profiles["temperature"]
            if np.any(temperature):
                fig = plot_grid_quantity(
                    grid, profiles["temperature"], title=f"{name} temperature [eV]", logscale=True
                )
                fig.savefig(
                    plots_path / f"edge_mesh_{name}_temperature.png",
                    dpi=200,
                )

            if charge:
                vpar = profiles["velocity_parallel"]
                if np.any(vpar):
                    fig = plot_grid_quantity(
                        grid, vpar, title=f"{name} parallel velocity [m/s]", symmetric=True
                    )
                    fig.savefig(
                        plots_path / f"edge_mesh_{name}_vpar.png",
                        dpi=200,
                    )
            else:
                vrad = profiles["velocity_radial"]
                if np.any(vrad):
                    fig = plot_grid_quantity(
                        grid, vrad, title=f"{name} radial velocity [m/s]", symmetric=True
                    )
                    fig.savefig(
                        plots_path / f"edge_mesh_{name}_vrad.png",
                        dpi=200,
                    )
                vpol = profiles["velocity_poloidal"]
                if np.any(vpol):
                    fig = plot_grid_quantity(
                        grid, vrad, title=f"{name} poloidal velocity [m/s]", symmetric=True
                    )
                    fig.savefig(
                        plots_path / f"edge_mesh_{name}_vpol.png",
                        dpi=200,
                    )
                vtor = profiles["velocity_phi"]
                if np.any(vtor):
                    fig = plot_grid_quantity(
                        grid, vtor, title=f"{name} toroidal velocity [m/s]", symmetric=True
                    )
                    fig.savefig(
                        plots_path / f"edge_mesh_{name}_vtor.png",
                        dpi=200,
                    )
