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
        assert fig is not None

    fig = visualize(
        bolometers[-1],
        num_rays=100,
        ray_from_channel=[0, 3],
        ray_terminate_distance=1e-2,
        show=False,
    )
