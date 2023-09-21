
"""
This demo reads edge plasma grid and profiles and plots the plasma profiles on the grid.

Tested on SOLPS 4.3, SOLPS-ITER, SOLEDGE3X, JINTRAC and DINA-JINTRAC scenarios.
"""

import os
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib import pyplot as plt

import imas

from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.common.ggd import load_grid
from cherab.imas.ids.edge_profiles import load_edge_species


DATABASE = 'ITER'
USER = 'public'
BACKEND = imas.imasdef.MDSPLUS_BACKEND

# SOLPS-ITER
shot, run, time = 123001, 3, 0
# JINTRAC mkimas
# shot, run, time = 134000, 45, 300
# SOLEDGE3X
# shot, run, time = 106000, 1, 0
# DINA-JINTRAC
# shot, run, time = 134174, 117, 49.5

grid_subset_name = 'Cells'


def plot_grid_quantity(grid, quantity, title='', logscale=False, symmetric=False):

    ax = grid.plot_mesh(data=quantity)

    if logscale:
        linthresh = np.percentile(np.unique(quantity), 1)
        norm = SymLogNorm(linthresh=max(linthresh, 1.e-10 * quantity.max()), base=10)
        ax.collections[0].set_norm(norm)

    if symmetric:
        vmax = np.abs(quantity.max())
        ax.collections[0].set_clim(-vmax, vmax)
        ax.collections[0].set_cmap('coolwarm')
    else:
        ax.collections[0].set_cmap('gnuplot')

    ax.set_title(title)
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    fig = ax.get_figure()
    fig.colorbar(ax.collections[0], ax=ax, aspect=40)
    fig.set_size_inches((4, 6))

    return fig


demos_path = os.path.dirname(__file__)
plots_path = os.path.join(demos_path, 'plots')
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

entry = imas.DBEntry(BACKEND, DATABASE, shot, run, USER)
edge_profiles_ids = get_ids_time_slice(entry, 'edge_profiles', time=time)

grid, subsets, subset_id = load_grid(edge_profiles_ids.grid_ggd[0], with_subsets=True)

try:
    grid = grid.subset(subsets[grid_subset_name], name=grid_subset_name)
except KeyError:
    grid_subset_name = grid_subset_name.lower()
    grid = grid.subset(subsets[grid_subset_name.lower()], name=grid_subset_name)

composition = load_edge_species(edge_profiles_ids.ggd[0], grid_subset_index=subset_id[grid_subset_name])

# Plot mesh
ax = grid.plot_mesh()
ax.set_xlabel('R [m]')
ax.set_ylabel('Z [m]')
fig = ax.get_figure()
fig.set_size_inches((4, 6))
fig.savefig(os.path.join(plots_path, '{}_{}_{}_edge_mesh.png'.format(shot, run, time)), dpi=300)

# Plot electron profiles
fig = plot_grid_quantity(grid, composition['electron']['density'], title="Electron density [m-3]", logscale=True)
fig.savefig(os.path.join(plots_path, '{}_{}_{}_edge_mesh_ne.png'.format(shot, run, time)), dpi=200)

fig = plot_grid_quantity(grid, composition['electron']['temperature'], title="Electron temperature [eV]", logscale=True)
fig.savefig(os.path.join(plots_path, '{}_{}_{}_edge_mesh_te.png'.format(shot, run, time)), dpi=200)

vpar = composition['electron']['velocity_parallel']
if np.any(vpar):
    fig = plot_grid_quantity(grid, vpar, title="Electron parallel velocity [m/s]", symmetric=True)
    fig.savefig(os.path.join(plots_path, '{}_{}_{}_edge_mesh_electron_vpar.png'.format(shot, run, time)), dpi=200)

# Plot species profiles
for stype in ('ion', 'molecule'):
    for species_id, profiles in composition[stype].items():
        d = {first: second for first, second in species_id}
        charge = int(round(d['z']))
        if stype == 'ion':
            element = d['element']
            label = d['label'] if len(d['label']) else '{}{}'.format(element.symbol, charge)
        else:
            element = d['elements'][0]
            label = d['label'] if len(d['label']) else '{}{}'.format(''.join([el.symbol for el in d['elements']]), charge)

        fig = plot_grid_quantity(grid, profiles['density'], title="{} density [m-3]".format(label), logscale=True)
        fig.savefig(os.path.join(plots_path, '{}_{}_{}_edge_mesh_{}_density.png'.format(shot, run, time, label)), dpi=200)

        if element.atomic_number == 1:
            temperature = profiles['temperature']
            if np.any(temperature):
                fig = plot_grid_quantity(grid, profiles['temperature'], title="{} temperature [eV]".format(label), logscale=True)
                fig.savefig(os.path.join(plots_path, '{}_{}_{}_edge_mesh_{}_temperature.png'.format(shot, run, time, label)), dpi=200)

            if charge:
                vpar = profiles['velocity_parallel']
                if np.any(vpar):
                    fig = plot_grid_quantity(grid, vpar, title="{} parallel velocity [m/s]".format(label), symmetric=True)
                    fig.savefig(os.path.join(plots_path, '{}_{}_{}_edge_mesh_{}_vpar.png'.format(shot, run, time, label)), dpi=200)
            else:
                vrad = profiles['velocity_radial']
                if np.any(vrad):
                    fig = plot_grid_quantity(grid, vrad, title="{} radial velocity [m/s]".format(label), symmetric=True)
                    fig.savefig(os.path.join(plots_path, '{}_{}_{}_edge_mesh_{}_vrad.png'.format(shot, run, time, label)), dpi=200)
                vpol = profiles['velocity_poloidal']
                if np.any(vpol):
                    fig = plot_grid_quantity(grid, vrad, title="{} poloidal velocity [m/s]".format(label), symmetric=True)
                    fig.savefig(os.path.join(plots_path, '{}_{}_{}_edge_mesh_{}_vpol.png'.format(shot, run, time, label)), dpi=200)
                vtor = profiles['velocity_toroidal']
                if np.any(vtor):
                    fig = plot_grid_quantity(grid, vtor, title="{} toroidal velocity [m/s]".format(label), symmetric=True)
                    fig.savefig(os.path.join(plots_path, '{}_{}_{}_edge_mesh_{}_vtor.png'.format(shot, run, time, label)), dpi=200)
