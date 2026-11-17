#!/usr/bin/env -S pixi run python
"""Calculate incident radiative power on each bolometer foil from IMAS radiation data.

Write power [W], Monte Carlo standard errors and the selected time to an IMAS
bolometer IDS in netCDF format, retaining camera/channel names and order.
"""

# %%
# Imports
# -------
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from imas import DBEntry, IDSFactory
from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS
from raysect.optical import World
from raysect.optical.material import AbsorbingSurface
from raysect.optical.observer import PowerPipeline0D

from cherab.imas import __version__
from cherab.imas.emitter import load_radiation_emitter
from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.observer import load_bolometers
from cherab.imas.wall import load_wall_mesh

# %%
# Configuration
# -------------
# Input IMAS URIs (bolometer, radiation, wall)
BOLOMETER_URI: str = "imas:hdf5?path=/work/imas/shared/imasdb/ITER_MD/4/150401/5"
RADIATION_URI: str = "imas:hdf5?path=/work/imas/shared/imasdb/ITER/3/134110/111"
WALL_URIS: tuple[str, ...] = (
    "imas:hdf5?path=/work/imas/shared/imasdb/ITER_MD/3/116100/1001",  # First wall
    "imas:hdf5?path=/work/imas/shared/imasdb/ITER_MD/3/116100/2001",  # Divertor
)
OUTPUT_PATH = Path(__file__).parent / "bolometer_incident_power.nc"

TIME = 0.0  # Requested radiation time [s]; use the nearest available slice.
NUM_SAMPLES = 1000  # Number of samples per foil.
INTEGRATION_STEP = 1e-3  # Volume integration step [m].

# Dummy spectral interval for bolometric RadiationFunction emission.
MIN_WAVELENGTH = 600.0  # [nm]
MAX_WAVELENGTH = 601.0  # [nm]
SPECTRAL_BINS = 1
SPECTRAL_RAYS = 1

# %%
# Validate configuration
# ----------------------
if not BOLOMETER_URI.strip() or not RADIATION_URI.strip():
    raise ValueError("Set both BOLOMETER_URI and RADIATION_URI before running this script")
if not WALL_URIS or any(not uri.strip() for uri in WALL_URIS):
    raise ValueError("Set WALL_URIS to at least one non-empty wall IDS URI")
if NUM_SAMPLES < 2 or not np.isfinite(INTEGRATION_STEP) or INTEGRATION_STEP <= 0:
    raise ValueError("NUM_SAMPLES must be >= 2 and INTEGRATION_STEP must be finite and positive")
if not np.isfinite(TIME) or TIME < 0:
    raise ValueError("TIME must be finite and non-negative")
if OUTPUT_PATH.suffix != ".nc":
    raise ValueError("OUTPUT_PATH must have the .nc extension")
if OUTPUT_PATH.exists():
    raise FileExistsError(f"Output already exists: {OUTPUT_PATH}")

# %%
# Load input IDSs and select the radiation time
# -------------------------------------------
with DBEntry(BOLOMETER_URI, "r") as entry:
    source = entry.get("bolometer", lazy=True, autoconvert=False)
    bolometer_dd_version = source._version
    # Extract names while the DBEntry is open; do not read geometry or existing signals.
    camera_metadata: list[tuple[str, list[str]]] = [
        (str(camera.name), [str(channel.name) for channel in camera.channel])
        for camera in source.camera
    ]
with DBEntry(RADIATION_URI, "r") as entry:
    radiation = get_ids_time_slice(entry, "radiation", time=TIME)
if len(radiation.time) != 1:
    raise RuntimeError("A single radiation time slice is required for this demo.")
time = float(radiation.time[0])

# %%
# Create scenegraph: machine, radiation emitter, and cameras
# ---------------------------------------------------------
world = World()
for uri in WALL_URIS:
    meshes = load_wall_mesh(uri, time=time, parent=world)
    if not meshes:
        raise RuntimeError(f"No wall meshes loaded from {uri}")
    for mesh in meshes.values():
        mesh.material = AbsorbingSurface()

load_radiation_emitter(RADIATION_URI, time=time, step=INTEGRATION_STEP, parent=world)
cameras = load_bolometers(BOLOMETER_URI, dd_version=bolometer_dd_version, parent=world)

# %%
# Prepare the output bolometer IDS
# -------------------------------
# Create only camera/channel identifiers; geometry and existing signals are not copied.
result = IDSFactory(bolometer_dd_version).new("bolometer")
result.camera.resize(len(camera_metadata))
for camera, (camera_name, channel_names) in zip(result.camera, camera_metadata, strict=True):
    camera.name = camera_name
    camera.channel.resize(len(channel_names))
    for channel, channel_name in zip(camera.channel, channel_names, strict=True):
        channel.name = channel_name
# Homogeneous time mode uses the root time, rather than channel.power.time.
result.time = [time]
result.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
result.ids_properties.creation_date = datetime.now(timezone.utc).isoformat()
result.ids_properties.comment = (
    "Synthetic bolometer incident power; power.data_error_upper contains symmetric "
    "Monte Carlo standard errors (one sigma). "
    f"Geometry: {BOLOMETER_URI}; radiation: {RADIATION_URI}; "
    f"requested time: {TIME} s; selected time: {time} s; "
    f"samples/foil: {NUM_SAMPLES}; integration step: {INTEGRATION_STEP} m; "
    f"Absorbing wall meshes: {WALL_URIS}."
)
result.code.name = "cherab-imas bolometer_incident_power"
result.code.version = __version__

# %%
# Render incident power for each foil
# -----------------------------------
# The loader skips empty cameras, but otherwise preserves camera/channel order.
output_cameras = [camera for camera in result.camera if len(camera.channel)]
for camera, output_camera in zip(cameras, output_cameras, strict=True):
    print(f"Rendering {camera.name}: {len(camera)} channels at {time:g} s")
    for foil, channel in zip(camera, output_camera.channel, strict=True):
        pipeline = PowerPipeline0D(accumulate=False)
        foil.pipelines = [pipeline]
        foil.pixel_samples = NUM_SAMPLES
        foil.min_wavelength = MIN_WAVELENGTH
        foil.max_wavelength = MAX_WAVELENGTH
        foil.spectral_bins = SPECTRAL_BINS
        foil.spectral_rays = SPECTRAL_RAYS
        foil.observe()

        power = pipeline.value.mean
        error = pipeline.value.error()
        if not np.isfinite([power, error]).all():
            raise RuntimeError(f"Non-finite power/error for {foil.name}")
        channel.power.data = [power]
        # For symmetric errors, data_error_upper represents both sides; leave lower empty.
        channel.power.data_error_upper = [error]
        channel.validity = 0
        print(f"  {foil.name}: {power:.6g} +/- {error:.3g} W")

# %%
# Validate and save the IMAS netCDF output
# ---------------------------------------
result.validate()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with DBEntry(str(OUTPUT_PATH), "x", dd_version=result._version) as entry:
    entry.put(result)
print(f"Saved IMAS bolometer IDS: {OUTPUT_PATH.resolve()}")
