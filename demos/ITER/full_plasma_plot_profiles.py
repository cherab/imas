
import os
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib import pyplot as plt

import imas

from cherab.core.math import samplevector2d, sample3d, sample3d_grid, samplevector3d_grid, samplevector2d_grid
from cherab.tools.equilibrium import plot_equilibrium
from cherab.imas import load_equilibrium, load_plasma


DATABASE = 'ITER'
USER = 'public'

# JINTRAC mkimas
shot, run, time = 134000, 45, 300
# DINA-JINTRAC
# shot, run, time = 134174, 117, 49.5


def plot_quantity(quantity, extent, title='', logscale=False, symmetric=False):
    """
    Make a 2D plot of quantity, with a title, optionally on a log scale.
    """

    fig = plt.figure(figsize=(4., 6.), tight_layout=True)
    ax = fig.add_subplot(111)
    if logscale:
        # Plot lowest values (mainly 0's) on linear map, as log(0) = -inf.
        linthresh = np.percentile(np.unique(quantity), 1)
        norm = SymLogNorm(linthresh=max(linthresh, 1.e-10 * quantity.max()), base=10)
    else:
        norm = None
    # Sampled data is indexed as quantity(x, y), but matplotlib's imshow
    # expects quantity(y, x).
    if symmetric and not logscale:
        vmax = np.abs(quantity).max()
        image = ax.imshow(quantity.T, extent=extent, origin='lower', vmin=-vmax, vmax=vmax, cmap='coolwarm')
    else:
        image = ax.imshow(quantity.T, extent=extent, origin='lower', norm=norm, cmap='gnuplot')
    fig.colorbar(image, aspect=50)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title(title)

    return fig


demos_path = os.path.dirname(__file__)
plots_path = os.path.join(demos_path, 'plots')
if not os.path.exists(plots_path ):
    os.makedirs(plots_path)

# sampling range
xl, xu = 4., 8.5
zl, zu = -4.5, 4.6
nz = 911
nx = 451
extent = [xl, xu, zl, zu]

# Load and plot equilibrium
equilibrium, psi_interpolator = load_equilibrium(shot, run, USER, DATABASE, imas.imasdef.MDSPLUS_BACKEND, time=time,
                                                 with_psi_interpolator=True)
plot_equilibrium(equilibrium)
plt.gcf().savefig(os.path.join(plots_path, "{}_{}_{}_equilibrium.png".format(shot, run, time)), dpi=200)

# Sample and plot magnetic field
plot_velocity = False
b_field = equilibrium.b_field
try:
    xsamp, zsamp, b = samplevector2d(b_field, (xl, xu, nx), (zl, zu, nz))
    b_length = np.sqrt((b * b).sum(2))
    fig = plot_quantity(b[:, :, 0], extent, title='Brad [T]', symmetric=True)
    fig.savefig(os.path.join(plots_path, '{}_{}_{}_brad.png'.format(shot, run, time)), dpi=200)
    fig = plot_quantity(b[:, :, 1], extent, title='Btor [T]')
    fig.savefig(os.path.join(plots_path, '{}_{}_{}_btor.png'.format(shot, run, time)), dpi=200)
    fig = plot_quantity(b[:, :, 2], extent, title='Bz [T]', symmetric=True)
    fig.savefig(os.path.join(plots_path, '{}_{}_{}_bz.png'.format(shot, run, time)), dpi=200)
    radial_vector = np.zeros_like(b)
    radial_vector[:, :, 0] = -b[:, :, 2]
    radial_vector[:, :, 2] = b[:, :, 0]
    radial_vector /= np.sqrt((radial_vector * radial_vector).sum(2))[:, :, None]
    poloidal_vector = np.copy(b)
    poloidal_vector[:, :, 1] = 0
    poloidal_vector /= np.sqrt((poloidal_vector * poloidal_vector).sum(2))[:, :, None]
    plot_velocity = True
except ValueError:
    print('Unable to sample magnetic field. The magnetic field grid does not cover the edge region.')

# Load edge plasma
plasma = load_plasma(shot, run, USER, DATABASE, imas.imasdef.MDSPLUS_BACKEND, equilibrium=equilibrium, time=time, time_threshold=0.99)

# Sample and plot electron profiles
xsamp, _, zsamp, ne_plasma = sample3d(plasma.electron_distribution.density, (xl, xu, nx), (0, 0, 1), (zl, zu, nz))
ne_plasma = ne_plasma.squeeze()
fig = plot_quantity(ne_plasma, extent, title='ne [m-3]', logscale=True)
fig.savefig(os.path.join(plots_path, '{}_{}_{}_ne.png'.format(shot, run, time)), dpi=200)

te_plasma = sample3d_grid(plasma.electron_distribution.effective_temperature, xsamp, [0], zsamp).squeeze()
fig = plot_quantity(te_plasma, extent, title='Te [eV]', logscale=True)
fig.savefig(os.path.join(plots_path, '{}_{}_{}_te.png'.format(shot, run, time)), dpi=200)

if plot_velocity:
    electron_velocity = samplevector3d_grid(plasma.electron_distribution.bulk_velocity, xsamp, [0], zsamp).squeeze()
    electron_velocity_par = (electron_velocity * b).sum(2) / b_length
    fig = plot_quantity(electron_velocity_par, extent, title='Electron Vpar [m/s]', symmetric=True)
    fig.savefig(os.path.join(plots_path, '{}_{}_{}_electron_vpar.png'.format(shot, run, time)), dpi=200)

# Sample and plot ion and neutral profiles
for species in plasma.composition:
    density = sample3d_grid(species.distribution.density, xsamp, [0], zsamp).squeeze()
    fig = plot_quantity(density, extent, title='{}{} density [m-3]'.format(species.element.symbol, species.charge), logscale=True)
    fig.savefig(os.path.join(plots_path, '{}_{}_{}_{}{}_density.png'.format(shot, run, time, species.element.symbol, species.charge)), dpi=200)

    if species.element.atomic_number == 1:
        temperature = sample3d_grid(species.distribution.effective_temperature, xsamp, [0], zsamp).squeeze()
        fig = plot_quantity(temperature, extent, title='{}{} temperature [eV]'.format(species.element.symbol, species.charge), logscale=True)
        fig.savefig(os.path.join(plots_path, '{}_{}_{}_{}{}_temperature.png'.format(shot, run, time, species.element.symbol, species.charge)), dpi=200)

        if plot_velocity:
            velocity = samplevector3d_grid(species.distribution.bulk_velocity, xsamp, [0], zsamp).squeeze()

            if species.charge:
                vpar = (velocity * b).sum(2) / b_length
                fig = plot_quantity(vpar, extent, title='{}{} Vpar [m/s]'.format(species.element.symbol, species.charge), symmetric=True)
                fig.savefig(os.path.join(plots_path, '{}_{}_{}_{}{}_vpar.png'.format(shot, run, time, species.element.symbol, species.charge)), dpi=200)
            else:
                vrad = (velocity * radial_vector).sum(2)
                vpol = (velocity * poloidal_vector).sum(2)
                vtor = velocity[:, :, 1]
                fig = plot_quantity(vrad, extent, title='{}{} Vrad [m/s]'.format(species.element.symbol, species.charge), symmetric=True)
                fig.savefig(os.path.join(plots_path, '{}_{}_{}_{}{}_vrad.png'.format(shot, run, time, species.element.symbol, species.charge)), dpi=200)
                fig = plot_quantity(vpol, extent, title='{}{} Vpol [m/s]'.format(species.element.symbol, species.charge), symmetric=True)
                fig.savefig(os.path.join(plots_path, '{}_{}_{}_{}{}_vpol.png'.format(shot, run, time, species.element.symbol, species.charge)), dpi=200)
                fig = plot_quantity(vtor, extent, title='{}{} Vtor [m/s]'.format(species.element.symbol, species.charge), symmetric=True)
                fig.savefig(os.path.join(plots_path, '{}_{}_{}_{}{}_vtor.png'.format(shot, run, time, species.element.symbol, species.charge)), dpi=200)
