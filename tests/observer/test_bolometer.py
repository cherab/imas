import numpy as np
from imas import DBEntry
from plotly import graph_objs as go
from pytest import approx
from raysect.core.math import to_cylindrical

from cherab.imas.ids.bolometer import get_los, get_los_interp
from cherab.imas.observer.bolometer import load_bolometers, visualize


def test_load_bolometers(path_bolometer_moc: str) -> None:
    """Test loading bolometer data from an IDS dataset."""
    bolometers = load_bolometers(path_bolometer_moc, "r")

    # Check that the bolometer cameras are loaded correctly
    assert len(bolometers) == 3

    # Check the visualization (this is a smoke test to ensure it runs without errors)
    for bolo in bolometers:
        fig = visualize(
            bolo, num_rays=100, ray_from_channel=0, ray_terminate_distance=1e-2, show=False
        )
        assert isinstance(fig, go.Figure)

    fig = visualize(
        bolometers[-1],
        num_rays=100,
        ray_from_channel=[0, 3],
        ray_terminate_distance=1e-2,
        show=False,
    )
    assert isinstance(fig, go.Figure)


def test_los(path_bolometer_moc: str) -> None:
    """Test loading and interpolating line of sight information from an IDS dataset."""
    with DBEntry(path_bolometer_moc, "r") as entry:
        ids = entry.get("bolometer")
    los_dict = get_los(ids)

    ds = 1e-2
    los_interps = get_los_interp(ids, ds=ds)

    # Check that the line of sight information is loaded correctly
    for camera in ids.camera:
        name = str(camera.name)
        assert name in los_dict
        assert name in los_interps

        los_list = los_dict[name]
        los_interp = los_interps[name]

        assert len(camera.channel) == len(los_list)
        assert len(camera.channel) == len(los_interp)

        for (origin, terminal), los in zip(los_list, los_interp, strict=True):
            assert to_cylindrical(origin) == approx((los[0, 0], los[1, 0], np.rad2deg(los[2, 0])))
            assert to_cylindrical(terminal) == approx(
                (los[0, -1], los[1, -1], np.rad2deg(los[2, -1]))
            )

            # Check the spacing of the interpolated points
            los_xyz = np.vstack((los[0] * np.cos(los[2]), los[0] * np.sin(los[2]), los[1]))
            distances = np.linalg.norm(np.diff(los_xyz, axis=1), axis=0)
            assert distances.max() <= ds
