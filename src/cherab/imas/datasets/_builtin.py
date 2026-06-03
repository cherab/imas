"""Provide functionality to create builtin IMAS sample datasets."""

import datetime

import numpy as np
from raysect.core.math import Point3D, Vector3D, rotate_z, to_cylindrical

from imas import DBEntry, IDSFactory
from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

try:
    import pooch

except ImportError:
    pooch = None

N_CH = 5  # Number of channels per camera
N_APERTURE = 3  # Number of apertures per channel (for collimator cameras)
N_SUBCOL = 3  # Number of subcollimators (for collimator cameras with subcollimator)
POSITION = (9.0, 0.0)  # (R, Z)
SLIT_WIDTH = 4.0e-3
SLIT_HEIGHT = 5.0e-3
FOIL_WIDTH = 1.3e-3
FOIL_HEIGHT = 3.8e-3
SLIT_SENSOR_SEPARATION = 4.0e-2
FOIL_SEPARATION = 5.08e-3
SLIT_SEPARATION = 7.5e-3
SUBCOL_SEPARATION = 1.0e-3

DISTANCE_LOS_SCALE = 5.0

Y_AXIS = Vector3D(0, 1, 0)


def _bolo_data():
    """
    Create a mock bolometer IDS dataset.

    Returns
    -------
    IDSToplevel
        Mock bolometer IDS dataset.
    """
    ids = IDSFactory().new("bolometer")

    # Set properties
    ids.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    ids.ids_properties.comment = "Test bolometer IDS"
    ids.ids_properties.creation_date = datetime.date.today().isoformat()

    ids.time = [0.0]

    # Set the number of cameras
    ids.camera.resize(3)

    # ----------------------
    # === Pinhole camera ===
    # ----------------------
    camera = ids.camera[0]
    camera.name = "Pinhole Camera"
    camera.type = "pinhole"

    origin_slit = Point3D(POSITION[0], 0.0, POSITION[1])
    origin_foil = Point3D(POSITION[0] + SLIT_SENSOR_SEPARATION, 0.0, POSITION[1])
    basis_z = origin_foil.vector_to(origin_slit).normalise()
    basis_y = Y_AXIS.copy()
    basis_x = basis_y.cross(basis_z).normalise()

    camera.channel.resize(N_CH)
    for i_ch in range(N_CH):
        channel = camera.channel[i_ch]

        # Detector
        pos_foil = origin_foil + basis_x * (i_ch - (N_CH - 1) * 0.5) * FOIL_SEPARATION
        channel.detector.geometry_type = 3
        r, z, phi = to_cylindrical(pos_foil)
        channel.detector.centre.r = r
        channel.detector.centre.z = z
        channel.detector.centre.phi = np.deg2rad(phi)
        channel.detector.x1_width = FOIL_HEIGHT
        channel.detector.x2_width = FOIL_WIDTH
        for xyz in ["x", "y", "z"]:
            setattr(channel.detector.x1_unit_vector, xyz, getattr(basis_y, xyz))
            setattr(channel.detector.x2_unit_vector, xyz, getattr(basis_x, xyz))
            setattr(channel.detector.x3_unit_vector, xyz, getattr(basis_z, xyz))

        # Slit
        channel.aperture.resize(1)
        aperture = channel.aperture[0]
        aperture.geometry_type = 3
        r, z, phi = to_cylindrical(origin_slit)
        aperture.centre.r = r
        aperture.centre.z = z
        aperture.centre.phi = np.deg2rad(phi)
        aperture.x1_width = SLIT_HEIGHT
        aperture.x2_width = SLIT_WIDTH
        for xyz in ["x", "y", "z"]:
            setattr(aperture.x1_unit_vector, xyz, getattr(basis_y, xyz))
            setattr(aperture.x2_unit_vector, xyz, getattr(basis_x, xyz))
            setattr(aperture.x3_unit_vector, xyz, getattr(basis_z, xyz))

        # Line of sight (from detector's centre to slit centre)
        # Its length is set to DISTANCE_LOS_SCALE times the distance between the slit and the foil.
        terminal = pos_foil + pos_foil.vector_to(origin_slit) * DISTANCE_LOS_SCALE

        channel.line_of_sight.first_point.r = channel.detector.centre.r
        channel.line_of_sight.first_point.z = channel.detector.centre.z
        channel.line_of_sight.first_point.phi = channel.detector.centre.phi
        r, z, phi = to_cylindrical(terminal)
        channel.line_of_sight.second_point.r = r
        channel.line_of_sight.second_point.z = z
        channel.line_of_sight.second_point.phi = np.deg2rad(phi)

    # ---------------------------------------------
    # === Collimator camera (w/o subcollimator) ===
    # ---------------------------------------------
    camera = ids.camera[1]
    camera.name = "Collimator Camera"
    camera.type = "collimator"

    angle = 90.0  # [deg]  Angle of the collimator camera in toroidal

    origin_slit = Point3D(POSITION[0], 0.0, POSITION[1]).transform(rotate_z(angle))
    origin_foil = Point3D(POSITION[0] + SLIT_SENSOR_SEPARATION, 0.0, POSITION[1]).transform(
        rotate_z(angle)
    )
    basis_z = origin_foil.vector_to(origin_slit).normalise()
    basis_y = Y_AXIS.transform(rotate_z(angle))
    basis_x = basis_y.cross(basis_z).normalise()

    camera.channel.resize(N_CH)
    for i_ch in range(N_CH):
        channel = camera.channel[i_ch]

        # Detector
        pos_foil = origin_foil + basis_x * (i_ch - (N_CH - 1) * 0.5) * FOIL_SEPARATION
        channel.detector.geometry_type = 3
        r, z, phi = to_cylindrical(pos_foil)
        channel.detector.centre.r = r
        channel.detector.centre.z = z
        channel.detector.centre.phi = np.deg2rad(phi)
        channel.detector.x1_width = FOIL_HEIGHT
        channel.detector.x2_width = FOIL_WIDTH
        for xyz in ["x", "y", "z"]:
            setattr(channel.detector.x1_unit_vector, xyz, getattr(basis_y, xyz))
            setattr(channel.detector.x2_unit_vector, xyz, getattr(basis_x, xyz))
            setattr(channel.detector.x3_unit_vector, xyz, getattr(basis_z, xyz))

        # Slit (w/ inner apertures)
        pos_slit = origin_slit + basis_x * (i_ch - (N_CH - 1) * 0.5) * SLIT_SEPARATION
        _v = pos_foil.vector_to(pos_slit)

        channel.aperture.resize(N_APERTURE)
        for i_ap in range(N_APERTURE):
            pos_ap = pos_slit - _v * i_ap / N_APERTURE

            aperture = channel.aperture[i_ap]
            aperture.geometry_type = 3
            r, z, phi = to_cylindrical(pos_ap)
            aperture.centre.r = r
            aperture.centre.z = z
            aperture.centre.phi = np.deg2rad(phi)
            aperture.x1_width = FOIL_HEIGHT + (SLIT_HEIGHT - FOIL_HEIGHT) * (1 - i_ap / N_APERTURE)
            aperture.x2_width = FOIL_WIDTH + (SLIT_WIDTH - FOIL_WIDTH) * (1 - i_ap / N_APERTURE)
            for xyz in ["x", "y", "z"]:
                setattr(aperture.x1_unit_vector, xyz, getattr(basis_y, xyz))
                setattr(aperture.x2_unit_vector, xyz, getattr(basis_x, xyz))
                setattr(aperture.x3_unit_vector, xyz, getattr(basis_z, xyz))

        # Line of sight (from detector's centre to slit centre)
        # Its length is set to DISTANCE_LOS_SCALE times the distance between the slit and the foil.
        terminal = pos_foil + pos_foil.vector_to(pos_slit) * DISTANCE_LOS_SCALE
        channel.line_of_sight.first_point.r = channel.detector.centre.r
        channel.line_of_sight.first_point.z = channel.detector.centre.z
        channel.line_of_sight.first_point.phi = channel.detector.centre.phi
        r, z, phi = to_cylindrical(terminal)
        channel.line_of_sight.second_point.r = r
        channel.line_of_sight.second_point.z = z
        channel.line_of_sight.second_point.phi = np.deg2rad(phi)

    # --------------------------------------------
    # === Collimator camera (w/ subcollimator) ===
    # --------------------------------------------
    camera = ids.camera[2]
    camera.name = "Collimator Camera (w/ subcollimator)"
    camera.type = "collimator"

    angle = 180.0  # [deg]  Angle of the collimator camera in toroidal

    origin_slit = Point3D(POSITION[0], 0.0, POSITION[1]).transform(rotate_z(angle))
    origin_foil = Point3D(POSITION[0] + SLIT_SENSOR_SEPARATION, 0.0, POSITION[1]).transform(
        rotate_z(angle)
    )
    basis_z = origin_foil.vector_to(origin_slit).normalise()
    basis_y = Y_AXIS.transform(rotate_z(angle))
    basis_x = basis_y.cross(basis_z).normalise()

    camera.channel.resize(N_CH)
    for i_ch in range(N_CH):
        channel = camera.channel[i_ch]

        # Detector
        pos_foil = origin_foil + basis_x * (i_ch - (N_CH - 1) * 0.5) * FOIL_SEPARATION
        channel.detector.geometry_type = 3
        r, z, phi = to_cylindrical(pos_foil)
        channel.detector.centre.r = r
        channel.detector.centre.z = z
        channel.detector.centre.phi = np.deg2rad(phi)
        channel.detector.x1_width = FOIL_HEIGHT
        channel.detector.x2_width = FOIL_WIDTH
        for xyz in ["x", "y", "z"]:
            setattr(channel.detector.x1_unit_vector, xyz, getattr(basis_y, xyz))
            setattr(channel.detector.x2_unit_vector, xyz, getattr(basis_x, xyz))
            setattr(channel.detector.x3_unit_vector, xyz, getattr(basis_z, xyz))

        # Slit (w/ inner apertures & subcollimator)
        pos_slit = origin_slit + basis_x * (i_ch - (N_CH - 1) * 0.5) * SLIT_SEPARATION
        _v = pos_foil.vector_to(pos_slit)

        channel.subcollimators_n = N_SUBCOL
        channel.subcollimators_separation = SUBCOL_SEPARATION

        channel.aperture.resize(N_APERTURE * N_SUBCOL)
        for i_ap in range(N_APERTURE):
            pos_ap = pos_slit - _v * i_ap / N_APERTURE
            width = FOIL_WIDTH + (SLIT_WIDTH - FOIL_WIDTH) * (1 - i_ap / N_APERTURE)
            height = FOIL_HEIGHT + (SLIT_HEIGHT - FOIL_HEIGHT) * (1 - i_ap / N_APERTURE)

            for i_subcol in range(N_SUBCOL):
                pos_ap_subcol = (
                    pos_ap
                    + basis_y
                    * (i_subcol - (N_SUBCOL - 1) * 0.5)
                    * (height + SUBCOL_SEPARATION)
                    / N_SUBCOL
                )

                aperture = channel.aperture[i_ap * N_SUBCOL + i_subcol]
                aperture.geometry_type = 3
                r, z, phi = to_cylindrical(pos_ap_subcol)
                aperture.centre.r = r
                aperture.centre.z = z
                aperture.centre.phi = np.deg2rad(phi)
                aperture.x1_width = (height - SUBCOL_SEPARATION * (N_SUBCOL - 1.0)) / N_SUBCOL
                aperture.x2_width = width
                for xyz in ["x", "y", "z"]:
                    setattr(aperture.x1_unit_vector, xyz, getattr(basis_y, xyz))
                    setattr(aperture.x2_unit_vector, xyz, getattr(basis_x, xyz))
                    setattr(aperture.x3_unit_vector, xyz, getattr(basis_z, xyz))

        # Line of sight (from detector's centre to slit centre)
        # Its length is set to DISTANCE_LOS_SCALE times the distance between the slit and the foil.
        terminal = pos_foil + pos_foil.vector_to(pos_slit) * DISTANCE_LOS_SCALE
        channel.line_of_sight.first_point.r = channel.detector.centre.r
        channel.line_of_sight.first_point.z = channel.detector.centre.z
        channel.line_of_sight.first_point.phi = channel.detector.centre.phi
        r, z, phi = to_cylindrical(terminal)
        channel.line_of_sight.second_point.r = r
        channel.line_of_sight.second_point.z = z
        channel.line_of_sight.second_point.phi = np.deg2rad(phi)

    return ids


def bolometer_moc() -> str:
    """Return the path to a mock bolometer dataset for testing purposes.

    Returns
    -------
    str
        Path to the mock bolometer dataset file.

    Raises
    ------
    ImportError
        If the `pooch` library is not installed, which is required to fetch the dataset.

    Examples
    --------
    >>> from cherab.imas import datasets
    >>> data_path = datasets.bolometer_moc()
    >>> data_path
    '.../cherab/imas/bolometer_moc.nc'
    """
    if pooch is None:
        raise ImportError("The 'pooch' library is required to fetch the bolometer dataset.")

    path = pooch.os_cache("cherab/imas") / "bolometer_moc.nc"

    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        # Create the mock bolometer dataset and save it to the cache path
        ids = _bolo_data()
        with DBEntry(str(path), "w") as entry:
            entry.put(ids)

    return str(path)
