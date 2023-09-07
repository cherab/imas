
import os
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib import pyplot as plt

import imas

from cherab.core.math import samplevector2d, sample3d_grid
from cherab.tools.equilibrium import plot_equilibrium
from cherab.imas import load_equilibrium, load_core_plasma


DATABASE = 'ITER'
USER = 'public'

# CORSICA
shot, run, time = 100506, 3, 100
# ASTRA
# shot, run, time = 101001, 60, 0
# JETTO mkimas
# shot, run, time = 104010, 1, 0
# JINTRAC mkimas
# shot, run, time = 134000, 45, 300
# DINA-IMAS
# shot, run, time = 105030, 1, 100
# DINA-JINTRAC
# shot, run, time = 134174, 117, 70.


def plot_quantity(quantity, extent, title='', logscale=False, symmetric=False):
    """
    Make a 2D plot of quantity, with a title, optionally on a log scale.
    """

    fig = plt.figure(figsize=(4.5, 6.), tight_layout=True)
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
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

# Load and plot equilibrium
equilibrium, psi_interpolator = load_equilibrium(shot, run, USER, DATABASE, imas.imasdef.MDSPLUS_BACKEND, time=time,
                                                 with_psi_interpolator=True)
plot_equilibrium(equilibrium)
plt.gcf().savefig(os.path.join(plots_path, "{}_{}_{}_equilibrium.png".format(shot, run, time)), dpi=200)

# sampling range
xl, xu = equilibrium.r_range
zl, zu = equilibrium.z_range
nz = 600
nx = 300
extent = [xl, xu, zl, zu]

# Sample and plot magnetic field
xsamp, zsamp, b = samplevector2d(equilibrium.b_field, (xl, xu, nx), (zl, zu, nz))
b_length = np.sqrt((b * b).sum(2))
fig = plot_quantity(b[:, :, 0], extent, title='Brad [T]', symmetric=True)
fig.savefig(os.path.join(plots_path, '{}_{}_{}_brad.png'.format(shot, run, time)), dpi=200)
fig = plot_quantity(b[:, :, 1], extent, title='Btor [T]')
fig.savefig(os.path.join(plots_path, '{}_{}_{}_btor.png'.format(shot, run, time)), dpi=200)
fig = plot_quantity(b[:, :, 2], extent, title='Bz [T]', symmetric=True)
fig.savefig(os.path.join(plots_path, '{}_{}_{}_bz.png'.format(shot, run, time)), dpi=200)

# Load core plasma
plasma = load_core_plasma(shot, run, USER, DATABASE, imas.imasdef.MDSPLUS_BACKEND, time=time,
                          equilibrium=equilibrium, psi_interpolator=psi_interpolator)

# Sample and plot electron profiles
ne_plasma = sample3d_grid(plasma.electron_distribution.density, xsamp, [0], zsamp)
ne_plasma = ne_plasma.squeeze()
fig = plot_quantity(ne_plasma, extent, title='ne [m-3]')
fig.savefig(os.path.join(plots_path, '{}_{}_{}_core_ne.png'.format(shot, run, time)), dpi=200)

te_plasma = sample3d_grid(plasma.electron_distribution.effective_temperature, xsamp, [0], zsamp).squeeze()
fig = plot_quantity(te_plasma, extent, title='Te [eV]')
fig.savefig(os.path.join(plots_path, '{}_{}_{}_core_te.png'.format(shot, run, time)), dpi=200)

# Sample and plot ion and neutral profiles
for species in plasma.composition:
    density = sample3d_grid(species.distribution.density, xsamp, [0], zsamp).squeeze()
    if np.any(density):
        fig = plot_quantity(density, extent, title='{}{} density [m-3]'.format(species.element.symbol, species.charge))
        fig.savefig(os.path.join(plots_path, '{}_{}_{}_core_{}{}_density.png'.format(shot, run, time, species.element.symbol, species.charge)), dpi=200)

    if species.element.atomic_number == 1:
        temperature = sample3d_grid(species.distribution.effective_temperature, xsamp, [0], zsamp).squeeze()
        if np.any(temperature):
            fig = plot_quantity(temperature, extent, title='{}{} temperature [eV]'.format(species.element.symbol, species.charge))
            fig.savefig(os.path.join(plots_path, '{}_{}_{}_core_{}{}_temperature.png'.format(shot, run, time, species.element.symbol, species.charge)), dpi=200)
