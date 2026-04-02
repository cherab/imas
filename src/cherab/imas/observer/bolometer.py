# Copyright 2023 Euratom
# Copyright 2023 United Kingdom Atomic Energy Authority
# Copyright 2023 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.
"""Module for loading bolometer cameras from IMAS bolometer IDS."""

from __future__ import annotations

from itertools import cycle
from typing import Literal, overload

import numpy as np
from raysect.core.constants import ORIGIN
from raysect.core.math import AffineMatrix3D, Point3D, Vector3D, rotate_basis, translate
from raysect.core.scenegraph._nodebase import _NodeBase
from raysect.optical.loggingray import LoggingRay
from raysect.optical.material import AbsorbingSurface
from raysect.optical.scenegraph import World
from raysect.primitive import Box, Subtract, Union
from raysect.primitive.csg import CSGPrimitive

from cherab.tools.observers.bolometry import BolometerCamera, BolometerFoil, BolometerSlit
from imas.db_entry import DBEntry

from ..ids.bolometer import load_cameras
from ..ids.bolometer._camera import BoloCamera, Geometry
from ..ids.bolometer.utility import CameraType, GeometryType
from ..ids.common import get_ids_time_slice

__all__ = ["load_bolometers", "visualize"]

X_AXIS = Vector3D(1, 0, 0)
Y_AXIS = Vector3D(0, 1, 0)
Z_AXIS = Vector3D(0, 0, 1)
THICKNESS = 2.0e-3  # Thickness of camera box walls
THICKNESS_INNER_LAYER = 0.5e-3  # Thickness of inner aperture layers
EPS = 1e-4  # Small epsilon value to avoid numerical issues


@overload
def load_bolometers(
    backend_id: int,
    db_name: str,
    pulse: int,
    run: int,
    user_name: str | None = None,
    data_version: str | None = None,
    *,
    parent: _NodeBase | None = None,
    shot: int | None = None,
    dd_version: str | None = None,
    xml_path: str | None = None,
) -> list[BolometerCamera]: ...


@overload
def load_bolometers(
    uri: str,
    mode: str,
    *,
    parent: _NodeBase | None = None,
    dd_version: str | None = None,
    xml_path: str | None = None,
) -> list[BolometerCamera]: ...


def load_bolometers(
    *args,
    parent: _NodeBase | None = None,
    **kwargs,
) -> list[BolometerCamera]:
    """Load bolometer cameras from IMAS bolometer IDS.

    .. note::
        This function requires the Data dictionary v4.1.0 or later.

    Parameters
    ----------
    *args
        Arguments passed to `~imas.db_entry.DBEntry`.
    parent
        The parent node of `~cherab.tools.observers.bolometry.BolometerCamera` in the Raysect
        scene-graph, by default None.
    **kwargs
        Keyword arguments passed to `~imas.db_entry.DBEntry` constructor.

    Returns
    -------
    `list[BolometerCamera]`
        List of `~cherab.tools.observers.bolometry.BolometerCamera` objects.

    Raises
    ------
    ValueError
        If slit data is required for a pinhole camera but not provided.
    NotImplementedError
        If an unsupported camera type or geometry type is encountered.

    Examples
    --------
    >>> from raysect.optical import World
    >>> world = World()

    If you have a local IMAS database and store the "bolometer.h5" file there:

    >>> bolometers = load_bolometers("imas:hdf5?path=path/to/db/", "r", parent=world)

    If you want to load netCDF files directly:

    >>> bolometers = load_bolometers("path/to/bolometer_file.nc", "r", parent=world)
    """
    # Load bolometer IDS
    with DBEntry(*args, **kwargs) as entry:
        # Get available time slices
        ids = get_ids_time_slice(entry, "bolometer")

    # Extract bolometer data
    bolo_data = load_cameras(ids)

    bolometers: list[BolometerCamera] = []

    for data in bolo_data:
        # Skip empty cameras
        if len(data.channels) == 0:
            continue

        # ------------------
        # === Camera Box ===
        # ------------------
        camera_box = _create_camera_box(data)
        camera = BolometerCamera(camera_geometry=camera_box, name=data.name, parent=parent)

        match data.type:
            case CameraType.PINHOLE:
                # ----------------------
                # === Slit (Pinhole) ===
                # ----------------------
                # Pick up only first aperture and use it for all channels
                slit_data = data.channels[0].slits[0]
                slit = BolometerSlit(
                    f"slit-{data.name}",
                    slit_data.centre,
                    slit_data.basis_x,
                    slit_data.dx,
                    slit_data.basis_y,
                    slit_data.dy,
                    curvature_radius=(slit_data.radius or 0.0)
                    if slit_data.type == GeometryType.CIRCULAR
                    else 0.0,
                    parent=camera,
                )
            case CameraType.COLLIMATOR:
                slit = None  # Defined per channel below
            case _:
                raise NotImplementedError(f"Camera type {data.type} not supported yet.")

        for i_channel, channel in enumerate(data.channels):
            # -------------------------
            # === Slit (Collimator) ===
            # -------------------------
            # Concatenate top plate slits into one slit if multiple sub-collimators
            # NOTE: Top plate slits are accumulated in the beginning of the slits list
            # NOTE: All slit geometry types are assumed to be the RECTANGLE.
            # NOTE: Sub-collimators are assumed to be aligned in a row along the basis_y direction.
            if (num_subcol := channel.num_subcollimator) > 1:
                _slit_data = channel.slits[:num_subcol]
                centre = (
                    ORIGIN
                    + sum([ORIGIN.vector_to(s.centre) for s in _slit_data], Vector3D(0, 0, 0))
                    / num_subcol
                )
                basis_x = _slit_data[0].basis_x
                basis_y = _slit_data[0].basis_y
                dx = _slit_data[0].dx
                dy = (
                    ORIGIN
                    + ORIGIN.vector_to(_slit_data[0].centre)
                    - _slit_data[0].dy * 0.5 * basis_y
                ).distance_to(
                    ORIGIN
                    + ORIGIN.vector_to(_slit_data[-1].centre)
                    + _slit_data[-1].dy * 0.5 * basis_y
                )

                slit_data = Geometry(
                    type=GeometryType.RECTANGLE,
                    centre=centre,
                    basis_x=basis_x,
                    basis_y=basis_y,
                    dx=dx,
                    dy=dy,
                )
            else:
                slit_data = channel.slits[0]

            if data.type == CameraType.COLLIMATOR:
                # Create slit object
                match slit_data.type:
                    case GeometryType.CIRCULAR | GeometryType.RECTANGLE:
                        slit = BolometerSlit(
                            f"slit-{data.name}-ch{i_channel}",
                            slit_data.centre,
                            slit_data.basis_x,
                            slit_data.dx,
                            slit_data.basis_y,
                            slit_data.dy,
                            curvature_radius=(slit_data.radius or 0.0)
                            if slit_data.type == GeometryType.CIRCULAR
                            else 0.0,
                            parent=camera,
                        )
                    case _:
                        raise NotImplementedError("Outline geometry not supported yet.")
            else:
                if slit is None:
                    raise ValueError("Slit data is required for pinhole camera.")

            # ------------
            # === Foil ===
            # ------------
            match channel.foil.type:
                case GeometryType.CIRCULAR | GeometryType.RECTANGLE:
                    foil = BolometerFoil(
                        f"foil-{data.name}-ch{i_channel}",
                        channel.foil.centre,
                        channel.foil.basis_x,
                        channel.foil.dx,
                        channel.foil.basis_y,
                        channel.foil.dy,
                        slit,
                        curvature_radius=(channel.foil.radius or 0.0)
                        if channel.foil.type == GeometryType.CIRCULAR
                        else 0.0,
                        parent=camera,
                    )
                case _:
                    raise NotImplementedError("Outline geometry not supported yet.")

            # Add foil to the camera
            camera.add_foil_detector(foil)

        bolometers.append(camera)

    return bolometers


def _create_camera_box(bolo_data: BoloCamera) -> CSGPrimitive:
    """Create a camera housing box geometry.

    This box is represented as a Subtract primitive, where a smaller inner box is subtracted
    from a larger outer box to create a hollow structure.
    Additionally, top plate apertures and other apertures (inner layers) are included as Unions.

    Returns
    -------
    `CSGPrimitive`
        The camera housing box geometry.
    """
    # Extract slits data
    slits_top: list[Geometry]
    slits_inner: list[list[Geometry]] = []
    if bolo_data.type == CameraType.PINHOLE:
        # Pick up only first aperture and use it for all channels
        slits_top = [bolo_data[0].slits[0]]
    else:
        # Consider sub-collimators for collimator cameras
        num_subcol = bolo_data[0].num_subcollimator
        num_channel = len(bolo_data)
        _s: list[Geometry] = []
        slits_top = sum(
            [bolo_data[i_ch].slits[:num_subcol] for i_ch in range(num_channel)],
            _s,
        )
        slits_inner = [
            sum(
                [
                    bolo_data[i_ch].slits[i_slit : i_slit + num_subcol]
                    for i_ch in range(num_channel)
                ],
                _s,
            )
            for i_slit in range(num_subcol, len(bolo_data[0].slits), num_subcol)
        ]

    # -------------------------------
    # === Local coordinate system ===
    # -------------------------------
    # The local origin is placed at the average position of top plate slits of all foil detectors,
    # and the basis vectors are defined based on the average normal and basis_x vectors of the slits.
    origin = ORIGIN + sum(
        [ORIGIN.vector_to(slit.centre) for slit in slits_top],
        Vector3D(0, 0, 0),
    ) / len(slits_top)

    basis_z: Vector3D = sum(
        [slit.basis_z for slit in slits_top],
        Vector3D(0, 0, 0),
    ).normalise()

    basis_x: Vector3D = sum(
        [slit.basis_x for slit in slits_top],
        Vector3D(0, 0, 0),
    ).normalise()

    basis_y = basis_z.cross(basis_x).normalise()

    # Transformation matrix from local to global coordinate system
    to_root = translate(*origin) * rotate_basis(basis_z, basis_y)

    # ------------------------------
    # === Determine box geometry ===
    # ------------------------------
    EPS_WIDTH = 1e-3
    EPS_HEIGHT = 1e-3
    EPS_DEPTH = 1e-3

    # Collect all corner points of foils and slits
    pts: list[Point3D] = []
    for geometry in [bolo_data[i].foil for i in range(len(bolo_data))] + slits_top:
        pts += _get_verts(geometry)

    # Camera box dimensions (in local coordinate system)
    box_width_u = max([origin.vector_to(p).dot(basis_x) for p in pts]) + EPS_WIDTH
    box_width_l = min([origin.vector_to(p).dot(basis_x) for p in pts]) - EPS_WIDTH
    box_height_u = max([origin.vector_to(p).dot(basis_y) for p in pts]) + EPS_HEIGHT
    box_height_l = min([origin.vector_to(p).dot(basis_y) for p in pts]) - EPS_HEIGHT
    box_depth_u = max([origin.vector_to(p).dot(basis_z) for p in pts])  # slit top plate
    box_depth_l = min([origin.vector_to(p).dot(basis_z) for p in pts]) - EPS_DEPTH

    # -------------------------
    # === Create Hollow Box ===
    # -------------------------
    camera_box = Box(
        lower=ORIGIN + box_width_l * X_AXIS + box_height_l * Y_AXIS + box_depth_l * Z_AXIS,
        upper=ORIGIN + box_width_u * X_AXIS + box_height_u * Y_AXIS + box_depth_u * Z_AXIS,
        name=f"inner-box-{bolo_data.name}",
    )
    outside_box = Box(
        lower=camera_box.lower - Vector3D(THICKNESS, THICKNESS, THICKNESS),
        upper=camera_box.upper + Vector3D(THICKNESS, THICKNESS, THICKNESS),
        name=f"outside-box-{bolo_data.name}",
    )
    camera_box = Subtract(outside_box, camera_box, name=f"hollow-box-{bolo_data.name}")

    # ------------------------------------------
    # === Clip out Top Plate Apertures Holes ===
    # ------------------------------------------
    for slit in slits_top:
        slit_verts = _get_verts(slit)
        aperture_box = Box(
            lower=ORIGIN
            + min([origin.vector_to(p).dot(basis_x) for p in slit_verts]) * X_AXIS
            + min([origin.vector_to(p).dot(basis_y) for p in slit_verts]) * Y_AXIS
            + (box_depth_u - EPS) * Z_AXIS,
            upper=ORIGIN
            + max([origin.vector_to(p).dot(basis_x) for p in slit_verts]) * X_AXIS
            + max([origin.vector_to(p).dot(basis_y) for p in slit_verts]) * Y_AXIS
            + (box_depth_u + THICKNESS + EPS) * Z_AXIS,
            name=f"aperture-box-top-{bolo_data.name}",
        )
        camera_box = Subtract(camera_box, aperture_box, name=f"aperture-top-{bolo_data.name}")

    # ----------------------------------
    # === Add Inner Apertures Plates ===
    # ----------------------------------
    # NOTE: Assume all inner apertures use the same local coordinate system as the top plate.
    for slits in slits_inner:
        # Create Inner Aperture Plate layer
        layer_depth_z = basis_z.dot(origin.vector_to(slits[0].centre))
        layer = Box(
            lower=Point3D(
                outside_box.lower.x,
                outside_box.lower.y,
                layer_depth_z - THICKNESS_INNER_LAYER * 0.5,
            ),
            upper=Point3D(
                outside_box.upper.x,
                outside_box.upper.y,
                layer_depth_z + THICKNESS_INNER_LAYER * 0.5,
            ),
            name=f"inner-plate-{bolo_data.name}",
        )

        for slit in slits:
            slit_verts = _get_verts(slit)
            aperture_box = Box(
                lower=ORIGIN
                + min([origin.vector_to(p).dot(basis_x) for p in slit_verts]) * X_AXIS
                + min([origin.vector_to(p).dot(basis_y) for p in slit_verts]) * Y_AXIS
                + (layer_depth_z - THICKNESS_INNER_LAYER * 0.5 - EPS) * Z_AXIS,
                upper=ORIGIN
                + max([origin.vector_to(p).dot(basis_x) for p in slit_verts]) * X_AXIS
                + max([origin.vector_to(p).dot(basis_y) for p in slit_verts]) * Y_AXIS
                + (layer_depth_z + THICKNESS_INNER_LAYER * 0.5 + EPS) * Z_AXIS,
                name=f"inner-aperture-box-{bolo_data.name}",
            )
            layer = Subtract(layer, aperture_box, name=f"inner-aperture-plate-{bolo_data.name}")

        camera_box = Union(camera_box, layer)

    # Transform to global coordinate system
    camera_box.transform = to_root

    # Apply absorbing material
    camera_box.material = AbsorbingSurface()

    # Name the camera box
    camera_box.name = f"camera-box-{bolo_data.name}"

    return camera_box


def _get_verts(geometry: Geometry) -> list[Point3D]:
    """Get the geometry vertices.

    For example, if the geometry is rectangular, return 4 corner vertices.

    Parameters
    ----------
    geometry
        Geometry structure object.

    Returns
    -------
    `list[Point3D]`
        List of geometry vertices.
    """
    dx = geometry.dx
    dy = geometry.dy
    basis_x = geometry.basis_x
    basis_y = geometry.basis_y
    center = geometry.centre
    match geometry.type:
        case GeometryType.RECTANGLE:
            verts = [
                center + 0.5 * dx * basis_x + 0.5 * dy * basis_y,
                center - 0.5 * dx * basis_x + 0.5 * dy * basis_y,
                center - 0.5 * dx * basis_x - 0.5 * dy * basis_y,
                center + 0.5 * dx * basis_x - 0.5 * dy * basis_y,
            ]
        case _:
            raise NotImplementedError(f"Geometry type {geometry.type} not implemented yet.")

    return verts


def visualize(
    camera: BolometerCamera,
    fig=None,
    num_rays: int | None = None,
    ray_from_channel: int | list[int] | None = None,
    ray_terminate_distance: float = 5.0e-2,
    aspect: Literal["data", "auto"] = "data",
    show: bool = True,
):
    """Visualize a bolometer camera geometry in 3D using Plotly.

    .. note::
        This function requires `plotly` to be installed.

    Parameters
    ----------
    camera
        The BolometerCamera object to visualize.
    fig : plotly.graph_objects.Figure | None
        An existing Plotly figure to add the camera visualization to, by default None.
    num_rays
        Number of rays to trace for visualizing the field of view, by default None.
    ray_from_channel
        Channel index or list of channel indices from which to trace rays, by default None.
    ray_terminate_distance
        Distance at which rays are terminated, by default 5.0e-2.
    aspect
        Aspect ratio of the plot, by default "data".
    show
        Whether to display the plot, by default True.

    Returns
    -------
    `plotly.graph_objects.Figure`
        The Plotly figure object.

    Raises
    ------
    ImportError
        If Plotly is not installed.
    TypeError
        If the provided fig is not a Plotly Figure instance.

    Examples
    --------
    >>> bolometers = load_bolometers("imas:hdf5?path=path/to/db/", "r")
    >>> fig = visualize(bolometers[0], num_rays=100, ray_from_channel=[0, 3])
    """
    try:
        from plotly import graph_objects as go
        from plotly.colors import qualitative
    except ImportError as e:
        raise ImportError("Plotly is required for visualization.") from e

    if fig is None:
        fig = go.Figure()
    else:
        if not isinstance(fig, go.Figure):
            raise TypeError("fig must be a plotly.graph_objects.Figure instance.")

    # Set up rays to be traced
    foils_ray_triggered: list[BolometerFoil] = []
    if isinstance(num_rays, int) and num_rays > 0:
        if isinstance(ray_from_channel, int):
            foils_ray_triggered = [camera.foil_detectors[ray_from_channel]]
        elif isinstance(ray_from_channel, list):
            foils_ray_triggered = [camera.foil_detectors[ch] for ch in ray_from_channel]
        else:
            foils_ray_triggered = camera.foil_detectors

    # Create scene graph temporally
    world = World()
    prev_parent = camera.parent
    try:
        camera.parent = world

        # === Local Axis ===
        local_origin = ORIGIN + sum(
            [ORIGIN.vector_to(foil.slit.centre_point) for foil in camera.foil_detectors],
            Vector3D(0, 0, 0),
        ) / len(camera.foil_detectors)
        local_z_axis: Vector3D = sum(
            [foil.slit.normal_vector.normalise() for foil in camera.foil_detectors],
            Vector3D(0, 0, 0),
        ).normalise()
        local_x_axis: Vector3D = sum(
            [foil.slit.basis_x.normalise() for foil in camera.foil_detectors],
            Vector3D(0, 0, 0),
        ).normalise()

        local_y_axis = local_z_axis.cross(local_x_axis).normalise()

        # --------------------------
        # === Plot Slits & Foils ===
        # --------------------------
        slit_name = ""
        for i_ch, foil in enumerate(camera.foil_detectors):
            foil: BolometerFoil
            for rect in [foil.slit, foil]:
                center: Point3D = rect.centre_point

                if isinstance(rect, BolometerSlit):
                    if rect.name == slit_name:
                        continue
                    else:
                        slit_name = rect.name
                        dx = rect.dx
                        dy = rect.dy
                elif isinstance(rect, BolometerFoil):
                    dx = rect.x_width
                    dy = rect.y_width
                else:
                    raise TypeError("Unknown rectangle type")

                basis_x: Vector3D = rect.basis_x.normalise()
                basis_y: Vector3D = rect.basis_y.normalise()

                # Calculate the foil edge coordinates
                vertices = [
                    center + 0.5 * dx * basis_x + 0.5 * dy * basis_y,
                    center - 0.5 * dx * basis_x + 0.5 * dy * basis_y,
                    center - 0.5 * dx * basis_x - 0.5 * dy * basis_y,
                    center + 0.5 * dx * basis_x - 0.5 * dy * basis_y,
                    center + 0.5 * dx * basis_x + 0.5 * dy * basis_y,  # Close the loop
                ]
                color = "blue" if rect is foil else "green"
                fig.add_trace(
                    go.Scatter3d(
                        x=[v.x for v in vertices],
                        y=[v.y for v in vertices],
                        z=[v.z for v in vertices],
                        mode="lines",
                        name=f"{'Foil' if rect is foil else 'Slit'} - {i_ch} CH",
                        line=dict(color=color, width=3),
                        hovertemplate=f"Width: {dx * 1e3:.2f} mm<br>Height: {dy * 1e3:.2f} mm",
                    )
                )

        # ------------------------------
        # === Camera Box (Outer Box) ===
        # ------------------------------
        if isinstance(camera._camera_geometry, CSGPrimitive):
            boxes = _extract_box_primitive(camera._camera_geometry)
            if boxes:
                for box in boxes:
                    fig: go.Figure = _plot_box(
                        box, transform=camera._camera_geometry.transform, fig=fig
                    )

        # -----------------
        # === Plot Rays ===
        # -----------------
        text_num_rays_passed = ""
        if isinstance(num_rays, int) and num_rays > 0:
            # Add terminating box to avoid rays escaping to infinity
            terminate_board = Box(
                lower=Point3D(-1e9, -1e9, -1e-3),
                upper=Point3D(1e9, 1e9, 0),
                parent=world,
                name="terminating_board",
            )
            terminate_board.material = AbsorbingSurface()
            terminate_board.transform = (
                translate(*local_origin)
                * rotate_basis(local_z_axis, local_y_axis)
                * translate(0, 0, ray_terminate_distance)
            )

            # Generate and trace rays
            count = 0
            for foil, color in zip(foils_ray_triggered, cycle(qualitative.Set1), strict=False):
                for ray in foil._generate_rays(LoggingRay(), num_rays):
                    origin = ray[0].origin.transform(foil.to_root())
                    direction = ray[0].direction.transform(foil.to_root())
                    ray = LoggingRay(origin=origin, direction=direction)
                    ray.trace(world)
                    hit_points = [origin, ray.path_vertices[-1]]
                    if ray.log[-1].primitive.name == "terminating_board":
                        count += 1
                    fig.add_trace(
                        go.Scatter3d(
                            x=[p.x for p in hit_points],
                            y=[p.y for p in hit_points],
                            z=[p.z for p in hit_points],
                            mode="lines+markers",
                            marker=dict(size=1.5, color=color),
                            line=dict(color=color, width=1),
                            name="rays",
                            showlegend=False,
                            hovertemplate=(
                                f"From: {foil.name}<br>"
                                f"Hit Object: {ray.log[-1].primitive.name}<extra></extra>"
                            ),
                        )
                    )
            total_rays = num_rays * len(foils_ray_triggered)
            if total_rays > 0:
                text_num_rays_passed = f" ({count / total_rays:.2%} Rays Passed)"

        # -----------------------
        # === Plot local axes ===
        # -----------------------
        scale = 0.01
        axes = [
            (local_x_axis, "X Axis", "rgb(255, 0, 0)"),
            (local_y_axis, "Y Axis", "rgb(0, 255, 0)"),
            (local_z_axis, "Z Axis", "rgb(0, 0, 255)"),
        ]
        for axis, name, color in axes:
            point = local_origin + scale * axis
            fig.add_trace(
                go.Scatter3d(
                    x=[local_origin.x, point.x],
                    y=[local_origin.y, point.y],
                    z=[local_origin.z, point.z],
                    name=name,
                    marker=dict(color=color, size=2),
                    line=dict(color=color),
                    showlegend=False,
                )
            )

        fig.update_layout(
            title=f"Bolometer Camera: {camera.name}{text_num_rays_passed}",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode=aspect,
            ),
            showlegend=True,
            width=700,
            height=600,
            margin=dict(r=10, l=10, b=10, t=35),
        )

        # restore previous parent node
        camera.parent = prev_parent

        if show:
            fig.show()

    finally:
        camera.parent = prev_parent

    return fig


def _extract_box_primitive(primitive: CSGPrimitive) -> list[Box]:
    """Recursively extract all Box primitives from CSGPrimitive.

    Parameters
    ----------
    primitive
        CSGPrimitive object to extract Box primitives from.

    Returns
    -------
    `list[Box]`
        List of Box primitives found within the CSGPrimitive.
    """
    boxes = []

    # Check primitive_a
    primitive_a = getattr(primitive, "primitive_a", None)
    if primitive_a is not None:
        if isinstance(primitive_a, Box):
            boxes.append(primitive_a)
        elif isinstance(primitive_a, CSGPrimitive):
            boxes.extend(_extract_box_primitive(primitive_a))

    # Check primitive_b
    primitive_b = getattr(primitive, "primitive_b", None)
    if primitive_b is not None:
        if isinstance(primitive_b, Box):
            boxes.append(primitive_b)
        elif isinstance(primitive_b, CSGPrimitive):
            boxes.extend(_extract_box_primitive(primitive_b))

    return boxes


def _plot_box(
    box: Box,
    transform: AffineMatrix3D,
    fig,
    color: str = "#7d7d7d",
):
    """Plot a box in a given figure.

    Parameters
    ----------
    box
        Box primitive to plot.
    transform
        Affine transformation to apply to the box vertices.
    fig
        Plotly figure to add the box plot to.
    color
        Color of the box, by default "#7d7d7d".

    Returns
    -------
    `plotly.graph_objects.Figure`
        The updated Plotly figure with the box plotted.
    """
    from plotly import graph_objects as go

    # Get box vertices
    _xaxis = X_AXIS.transform(transform)
    _yaxis = Y_AXIS.transform(transform)
    _zaxis = Z_AXIS.transform(transform)
    lower = box.lower.transform(transform)
    upper = box.upper.transform(transform)
    lower_to_upper = lower.vector_to(upper)
    box_width = abs(lower_to_upper.dot(_xaxis))
    box_height = abs(lower_to_upper.dot(_yaxis))
    box_depth = abs(lower_to_upper.dot(_zaxis))
    vertices = [
        lower,
        lower + box_width * _xaxis,
        lower + box_width * _xaxis + box_height * _yaxis,
        lower + box_height * _yaxis,
        lower + box_depth * _zaxis,
        lower + box_depth * _zaxis + box_width * _xaxis,
        upper,
        upper - box_width * _xaxis,
    ]
    verts = np.array([[*vertex] for vertex in vertices])

    # Plot box surfaces
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=0.2,
            color=color,
            flatshading=True,
            name="Camera Box",
            showlegend=True,
        )
    )

    # Plot box edges
    # Define the 12 edges of the box
    edges = [
        # bottom face
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        # top face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        # vertical edges
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    # Plot edges as lines
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        fig.add_trace(
            go.Scatter3d(
                x=[v1.x, v2.x],
                y=[v1.y, v2.y],
                z=[v1.z, v2.z],
                mode="lines",
                line=dict(color=color, width=2),
                name="Camera Box Edge",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    return fig
