import os
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pytest
from raysect.primitive import Cylinder, Subtract

import cherab.imas.emitter.radiation as radiation_module
from cherab.imas.emitter import load_radiation_emitter


class _EmitterCacheKwargs(TypedDict, total=False):
    interpolator_cache: Literal["memory", "disk"]
    interpolator_cache_dir: str | Path


@pytest.fixture(scope="session")
def radiation_interpolator_cache(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Literal["memory", "disk"], Path | None]:
    """Choose interpolator cache mode for radiation tests.

    Default is "memory" for speed. Set CHERAB_IMAS_RADIATION_TEST_CACHE=disk to
    reuse cache artifacts across test cases inside one session.
    """
    mode = os.getenv("CHERAB_IMAS_RADIATION_TEST_CACHE", "memory")
    if mode not in {"memory", "disk"}:
        mode = "memory"

    if mode == "disk":
        return mode, tmp_path_factory.mktemp("cherab-imas-radiation-cache")

    return mode, None


def _cache_kwargs(
    cache_cfg: tuple[Literal["memory", "disk"], Path | None],
) -> _EmitterCacheKwargs:
    mode, cache_dir = cache_cfg
    kwargs: _EmitterCacheKwargs = {"interpolator_cache": mode}
    if cache_dir is not None:
        kwargs["interpolator_cache_dir"] = cache_dir
    return kwargs


def test_load_radiation_emitter_coefficients_uses_default_phis(
    path_iter_jorek: str,
    monkeypatch: pytest.MonkeyPatch,
    radiation_interpolator_cache: tuple[Literal["memory", "disk"], Path | None],
):
    captured: dict[str, np.ndarray] = {}
    original_constructor = radiation_module.FourierBezierConstructor

    class _SpyFourierBezierConstructor:
        def __init__(self, *args, **kwargs):
            self._inner = original_constructor(*args, **kwargs)

        def average_gaussian_faces_per_toroidal(self, phis):
            captured["phis"] = np.asarray(phis, dtype=np.float64).copy()
            return self._inner.average_gaussian_faces_per_toroidal(phis)

    monkeypatch.setattr(radiation_module, "FourierBezierConstructor", _SpyFourierBezierConstructor)

    primitive = load_radiation_emitter(
        path_iter_jorek,
        "r",
        source="coefficients",
        **_cache_kwargs(radiation_interpolator_cache),
    )

    assert isinstance(primitive, (Subtract, Cylinder))
    assert primitive.material is not None
    assert primitive.name.startswith("RadiationEmitter_")

    phis_used = captured["phis"]
    assert phis_used.ndim == 1
    assert phis_used.size > 0
    assert np.all(np.diff(phis_used) > 0)
    assert 0.0 < phis_used[0] < 360.0
    assert 0.0 < phis_used[-1] < 360.0


def test_load_radiation_emitter_auto_falls_back_to_coefficients(
    path_iter_jorek: str,
    radiation_interpolator_cache: tuple[Literal["memory", "disk"], Path | None],
):
    primitive = load_radiation_emitter(
        path_iter_jorek,
        "r",
        source="auto",
        **_cache_kwargs(radiation_interpolator_cache),
    )

    assert isinstance(primitive, (Subtract, Cylinder))
    assert primitive.material is not None


def test_load_radiation_emitter_coefficients_uses_given_phis(
    path_iter_jorek: str,
    monkeypatch: pytest.MonkeyPatch,
    radiation_interpolator_cache: tuple[Literal["memory", "disk"], Path | None],
):
    captured: dict[str, np.ndarray] = {}
    original_constructor = radiation_module.FourierBezierConstructor

    class _SpyFourierBezierConstructor:
        def __init__(self, *args, **kwargs):
            self._inner = original_constructor(*args, **kwargs)

        def average_gaussian_faces_per_toroidal(self, phis):
            captured["phis"] = np.asarray(phis, dtype=np.float64).copy()
            return self._inner.average_gaussian_faces_per_toroidal(phis)

    monkeypatch.setattr(radiation_module, "FourierBezierConstructor", _SpyFourierBezierConstructor)

    expected_phis = np.array([15.0, 105.0, 195.0, 285.0], dtype=np.float64)
    primitive = load_radiation_emitter(
        path_iter_jorek,
        "r",
        source="coefficients",
        phis=expected_phis,
        **_cache_kwargs(radiation_interpolator_cache),
    )

    assert isinstance(primitive, (Subtract, Cylinder))
    assert primitive.material is not None
    np.testing.assert_allclose(captured["phis"], expected_phis)


def test_load_radiation_emitter_values_raises_for_jorek(
    path_iter_jorek: str,
    radiation_interpolator_cache: tuple[Literal["memory", "disk"], Path | None],
):
    with pytest.raises(RuntimeError, match="The 'ggd' AOS of the radiation IDS is empty"):
        load_radiation_emitter(
            path_iter_jorek,
            "r",
            source="values",
            **_cache_kwargs(radiation_interpolator_cache),
        )


def test_load_radiation_emitter_invalid_source_raises(
    path_iter_jorek: str,
    radiation_interpolator_cache: tuple[Literal["memory", "disk"], Path | None],
):
    with pytest.raises(RuntimeError, match="Unable to load emissivity"):
        load_radiation_emitter(
            path_iter_jorek,
            "r",
            source="unsupported",  # type: ignore[arg-type]
            **_cache_kwargs(radiation_interpolator_cache),
        )
