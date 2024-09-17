"""This demo creates a BolometerCamera object and plots the lines of sight with machine wall outline.

Tested on ITER IMAS database.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from raysect.core import Ray
from raysect.optical import World

from cherab.imas.observers import load_bolometers
from cherab.imas.wall import load_wall_outline

# Default ITER IMAS quaries
DATABASE = "ITER_MD"
USER = "public"
SHOT_BOLO = 150401
SHOT_WALL = 116000
RUN_BOLO = 3
RUN_WALL = 4

# Raysect root scene-graph
world = World()

# Load bolometer camera
bolos = load_bolometers(SHOT_BOLO, RUN_BOLO, USER, DATABASE)

# Load machine wall outline
wall = load_wall_outline(SHOT_WALL, RUN_WALL, USER, DATABASE)
first_wall = wall["First Wall"]
divertor = wall["Divertor"]

# %%
# Plot the bolometer lines of sight
# ---------------------------------
fig, ax = plt.subplots(layout="constrained")

# Plot the machine wall outline
ax.plot(first_wall[:, 0], first_wall[:, 1], color="black", zorder=-1)
ax.plot(divertor[:, 0], divertor[:, 1], color="black", zorder=-1)

# Plot the bolometer lines of sight
for bolo in bolos:
    origin = bolo.foil_detectors[0].centre_point
    los_vector = bolo.foil_detectors[0].sightline_vector
    los_ray = Ray(origin, los_vector)

    los_line = np.array([[*los_ray.point_on(i)] for i in range(0, 20)])

    # Project the line of sight onto the R-Z plane
    r, z = np.hypot(los_line[:, 0], los_line[:, 1]), los_line[:, 2]
    ax.plot(r, z, color="red", lw=0.25)

ax.set_xlabel("$R$ [m]")
ax.set_ylabel("$Z$ [m]")
ax.set_aspect("equal")

# Save the plot
plots_path = Path(__file__).parent / "plots"
plots_path.mkdir(exist_ok=True)
fig.savefig(
    plots_path / f"bolometer_lines_of_sight_{SHOT_BOLO}_{RUN_BOLO}.png",
    bbox_inches="tight",
    dpi=200,
)
