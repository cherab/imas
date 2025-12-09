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

from raysect.core.constants import ORIGIN
from raysect.core.math import Point3D, Vector3D, rotate_basis, translate
from raysect.core.scenegraph._nodebase import _NodeBase
from raysect.optical.material import AbsorbingSurface
from raysect.primitive import Box, Subtract, Union
from raysect.primitive.csg import CSGPrimitive

from cherab.tools.observers.bolometry import BolometerCamera, BolometerFoil, BolometerSlit
from imas.db_entry import DBEntry

from ..ids.bolometer import load_cameras
from ..ids.bolometer._camera import BoloCamera, Geometry
from ..ids.bolometer.utility import CameraType, GeometryType
from ..ids.common import get_ids_time_slice

__all__ = ["load_bolometers"]

X_AXIS = Vector3D(1, 0, 0)
Y_AXIS = Vector3D(0, 1, 0)
Z_AXIS = Vector3D(0, 0, 1)
THICKNESS = 2.0e-3  # Thickness of camera box walls
THICKNESS_INNER_LAYER = 0.5e-3  # Thickness of inner aperture layers
EPS = 1e-4  # Small epsilon value to avoid numerical issues


def load_bolometers(*args, parent: _NodeBase | None = None, **kwargs) -> list[BolometerCamera]:
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
    # NOTO: Assume all inner apertures use the same local coordinate system as the top plate.
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
