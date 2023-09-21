import os
import numpy as np
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
from multiprocessing import get_context

import imas

from raysect.core.math import Point3D, Vector3D
from raysect.optical import Spectrum
from raysect.optical.colour import spectrum_to_ciexyz, ciexyz_to_srgb, resample_ciexyz
from cherab.core.atomic import Line, beryllium, deuterium
from cherab.core.model import ExcitationLine, RecombinationLine
from cherab.openadas import OpenADAS

from cherab.imas import load_plasma


NPROC = 8  # number of parallel processes

DATABASE = 'ITER'
DATABASE_MD = 'ITER_MD'
USER = 'public'
USER_MD = 'public'

shot, run = 134174, 117
time = np.arange(10.5, 75.1, 0.5)

# 55.G1
geom_matrix_shot = 150701

# EP03 Divertor
# geom_matrix_run = 1000
# EP03 Left
# geom_matrix_run = 1010
# EP03 Right
geom_matrix_run = 1020
# EP03 Upper
# geom_matrix_run = 1030

# 55.E2
# geom_matrix_shot = 150502

# EP11 Lower (A0)
# geom_matrix_run = 1000
# EP11 Upper (B0)
# geom_matrix_run = 1010
# EP12 (C0)
# geom_matrix_run = 1020

unsaturated_fraction = 0.95


def read_geometry_matrix_without_reflections(shot, run, user, database='ITER_MD', backend=imas.imasdef.HDF5_BACKEND):

    geometry_matrix = {}

    entry = imas.DBEntry(backend, database, shot, run, user)
    entry.open()
    camera = entry.get('camera_visible')
    entry.close()

    geometry_matrix['camera_name'] = camera.channel[0].name
    geometry_matrix['pixel_to_alpha'] = camera.channel[0].detector[0].pixel_to_alpha
    geometry_matrix['pixel_to_beta'] = camera.channel[0].detector[0].pixel_to_beta

    geometry_matrix['voxel_map'] = camera.channel[0].detector[0].geometry_matrix.voxel_map

    emission_grid = {}
    emission_grid['r'] = camera.channel[0].detector[0].geometry_matrix.emission_grid.dim1
    emission_grid['z'] = camera.channel[0].detector[0].geometry_matrix.emission_grid.dim2
    geometry_matrix['emission_grid'] = emission_grid

    without_reflections = {}
    without_reflections['data'] = camera.channel[0].detector[0].geometry_matrix.without_reflections.data
    without_reflections['voxel_indices'] = camera.channel[0].detector[0].geometry_matrix.without_reflections.voxel_indices
    without_reflections['pixel_indices'] = camera.channel[0].detector[0].geometry_matrix.without_reflections.pixel_indices
    geometry_matrix['without_reflections'] = without_reflections

    return geometry_matrix


def read_geometry_matrix_interpolated(shot, run, user, database='ITER_MD', backend=imas.imasdef.HDF5_BACKEND):

    geometry_matrix = {}

    entry = imas.DBEntry(imas.imasdef.HDF5_BACKEND, 'ITER_MD', shot, run, user)
    entry.open()
    camera = entry.get('camera_visible')
    entry.close()

    geometry_matrix['camera_name'] = camera.channel[0].name
    geometry_matrix['pixel_to_alpha'] = camera.channel[0].detector[0].pixel_to_alpha
    geometry_matrix['pixel_to_beta'] = camera.channel[0].detector[0].pixel_to_beta

    interpolated = {}
    interpolated['r'] = camera.channel[0].detector[0].geometry_matrix.interpolated.r
    interpolated['z'] = camera.channel[0].detector[0].geometry_matrix.interpolated.z
    interpolated['data'] = camera.channel[0].detector[0].geometry_matrix.interpolated.data
    geometry_matrix['interpolated'] = interpolated

    return geometry_matrix


# Define spectral lines
dalpha = Line(deuterium, 0, (3, 2))
dbeta = Line(deuterium, 0, (4, 2))
dgamma = Line(deuterium, 0, (5, 2))
ddelta = Line(deuterium, 0, (6, 2))
be0_457nm = Line(beryllium, 0, ("2s1 3d1 1d2.0", "2s1 2p1 1p1.0"))
be1_436nm = Line(beryllium, 1, ("4d1 2d4.5", "3p1 2p2.5"))
be1_467nm = Line(beryllium, 1, ("4f1 2f6.5", "3d1 2d4.5"))
be1_527nm = Line(beryllium, 1, ("4s1 2s0.5", "3p1 2p2.5"))

spectral_lines = [dalpha, dbeta, dgamma, ddelta, be0_457nm, be1_436nm, be1_467nm, be1_527nm]

# Sample emission profiles

# sampling points
r = np.linspace(4., 8.5, 451)
z = np.linspace(-4.5, 4.6, 911)
volume = np.pi * (r[1:]**2 - r[:-1]**2)[:, None] * (z[1:] - z[:-1])[None, :]
r = 0.5 * (r[1:] + r[:-1])
z = 0.5 * (z[1:] + z[:-1])
r2d, z2d = np.meshgrid(r, z, indexing='ij')

atomic_data = OpenADAS(permit_extrapolation=True)

line_emission = np.zeros((len(spectral_lines), time.size, r.size, z.size))
line_wavelength = np.array([atomic_data.wavelength(line.element, line.charge, line.transition) for line in spectral_lines])

wvl_delta = 3.
direction = Vector3D(0, 0, 1)
print("Sampling emission profiles...")
for it, t in enumerate(time):
    print("Processing time moment: {} s".format(t))
    plasma = load_plasma(shot, run, USER, DATABASE, imas.imasdef.MDSPLUS_BACKEND, time=t)
    plasma.atomic_data = atomic_data

    for il, line in enumerate(spectral_lines):
        plasma.models = [ExcitationLine(line), RecombinationLine(line)]
        line_wvl = line_wavelength[il]
        spectrum = Spectrum(line_wvl - 0.5 * wvl_delta, line_wvl + 0.5 * wvl_delta, 1)
        for ir, r1 in enumerate(r):
            for iz, z1 in enumerate(z):
                point = Point3D(r1, 0, z1)
                for model in plasma.models:
                    line_emission[il, it, ir, iz] += model.emission(point, direction, spectrum.new_spectrum()).total()

# Load geometry matrix without reflections
geom_matrix_norefl = read_geometry_matrix_without_reflections(geom_matrix_shot, geom_matrix_run, USER_MD, DATABASE_MD)
npix_x = geom_matrix_norefl['pixel_to_alpha'].size
npix_y = geom_matrix_norefl['pixel_to_beta'].size
npix = npix_x * npix_y
pixels = geom_matrix_norefl['without_reflections']['pixel_indices']
ipix = npix_x * pixels[:, 0] + pixels[:, 1]
jvox = geom_matrix_norefl['without_reflections']['voxel_indices']
nvox = geom_matrix_norefl['voxel_map'].max() + 1
geom_matrix = coo_matrix((geom_matrix_norefl['without_reflections']['data'], (ipix, jvox)), shape=(npix, nvox)).tocsr()

voxel_map = geom_matrix_norefl['voxel_map'][:, :, 0]
r_gm = geom_matrix_norefl['emission_grid']['r']
z_gm = geom_matrix_norefl['emission_grid']['z']
ivox, indx_v = np.unique(voxel_map, return_index=True)

extent = (np.rad2deg(geom_matrix_norefl['pixel_to_alpha'][1]),
          np.rad2deg(geom_matrix_norefl['pixel_to_alpha'][-2]),
          np.rad2deg(geom_matrix_norefl['pixel_to_beta'][-2]),
          np.rad2deg(geom_matrix_norefl['pixel_to_beta'][1]))

# Wavelengths of the interpolated geometry matrices
geom_matrix_wvl = np.arange(400., 701., 50.)

# Obtain images
images = np.zeros((len(spectral_lines), time.size, npix_y, npix_x))

# Parallel tasks
npix_step = min(4096, npix // NPROC + bool(npix % NPROC))
tasks = [(ibegin, min(ibegin + npix_step, npix)) for ibegin in range(0, npix, npix_step)]

print("Obtaining images for each emission profile...")
gm_index_prev = -1
for il_indx, il in enumerate(np.argsort(line_wavelength)):
    # Load interpolated geometry matrix closest to this wavelength
    gm_index = np.argmin(np.abs(geom_matrix_wvl - line_wavelength[il])) + 1
    if gm_index != gm_index_prev:
        gm_interp = read_geometry_matrix_interpolated(geom_matrix_shot, geom_matrix_run + gm_index, USER_MD, DATABASE_MD)
        node_points = np.array([gm_interp['interpolated']['r'], gm_interp['interpolated']['z']]).T
        interp_data = np.moveaxis(gm_interp['interpolated']['data'], -1, 0).reshape((node_points.shape[0], npix))
        gm_index_prev = gm_index

    for it, t in enumerate(time):
        print("Processing time moment: {} s for emission profile {} out of {}.".format(t, il_indx + 1, len(spectral_lines)))
        # without reflections
        emission_interp = RectBivariateSpline(r, z, line_emission[il, it], kx=1, ky=1)
        emission_grid = emission_interp(r_gm, z_gm, grid=True)
        emission_prof = emission_grid.flatten()[indx_v[1:]] if ivox[0] == -1 else emission_grid.flatten()[indx_v]
        images[il, it] = (geom_matrix @ emission_prof).reshape(npix_y, npix_x)

        # add reflections
        indx = np.where(line_emission[il, it] > 0)
        power = volume[indx] * line_emission[il, it][indx]        
        points = np.array([r2d[indx], z2d[indx]]).T

        def f(task):
            ibegin = task[0]
            iend = task[1]
            interp_matrix = griddata(node_points, interp_data[:, ibegin:iend], points, fill_value=0, method='linear')

            return power @ interp_matrix

        ctx = get_context(method='fork')

        with ctx.Pool(processes=NPROC) as pool:
            results = pool.map(f, tasks)

        reflected_image = np.zeros(npix)
        for task, res in zip(tasks, results):
            reflected_image[task[0]:task[1]] = res

        images[il, it] += reflected_image.reshape((npix_y, npix_x))

# Convert to CIEXYZ
print("Converting to CIEXYZ...")
delta_wl = 1.
min_wavelength = line_wavelength.min() - 0.5 * delta_wl
max_wavelength = line_wavelength.max() + 0.5 * delta_wl
bins = int(round((max_wavelength - min_wavelength) / delta_wl))
delta_wl = (max_wavelength - min_wavelength) / bins
bin_index = ((line_wavelength - min_wavelength) / delta_wl).astype(int)
resampled_xyz = resample_ciexyz(min_wavelength, max_wavelength, bins)

images_ciexyz = np.zeros((time.size, npix_y, npix_x, 3))
for it in range(time.size):
    spectral_image = np.zeros((npix_y, npix_x, bins))
    for il in range(len(spectral_lines)):
        spectral_image[:, :, bin_index[il]] += images[il, it] / delta_wl
    for iy in range(npix_y):
        for ix in range(npix_x):
            spectrum = Spectrum(min_wavelength, max_wavelength, bins)
            spectrum.samples[:] += spectral_image[iy, ix]
            images_ciexyz[it, iy, ix] = spectrum_to_ciexyz(spectrum, resampled_xyz)

luminance = np.copy(images_ciexyz[:, :, :, 1]).flatten()
luminance.sort()
peak_luminance = luminance[min(luminance.size - 1, int(luminance.size * unsaturated_fraction))]
images_ciexyz /= peak_luminance

demos_path = os.path.dirname(__file__)
frames_path = os.path.join(demos_path, 'frames')
if not os.path.exists(frames_path):
    os.makedirs(frames_path)

# Convert to SRGB
print("Converting to SRGB and saving...")
camera_name = geom_matrix_norefl['camera_name'].replace(' ', '_')
fig = plt.figure(figsize=(5., 6.), tight_layout=True)
for it, t in enumerate(time):
    ciexyz = images_ciexyz[it]
    rgb_image = np.zeros(ciexyz.shape)
    for iy in range(npix_y):
        for ix in range(npix_x):
            rgb_image[iy, ix] = ciexyz_to_srgb(ciexyz[iy, ix, 0], ciexyz[iy, ix, 1], ciexyz[iy, ix, 2])

    ax = fig.add_subplot(111)
    im = ax.imshow(rgb_image[1:-1, 1:-1], extent=extent)
    ax.set_xlabel(r'$\alpha$, $\degree$')
    ax.set_ylabel(r'$\beta$, $\degree$')
    ax.set_title('{}/{}: {:.1f} s'.format(shot, run, t))
    fig.savefig(os.path.join(frames_path, '{}_rgb_{}_{}_time_{:.1f}s.png'.format(camera_name, shot, run, t)), dpi=300)
    fig.clear()

os.system('gm convert -delay 12 -loop 0 {}/*.png {}/{}_rgb_{}_{}.gif'.format(frames_path, demos_path, camera_name, shot, run))
