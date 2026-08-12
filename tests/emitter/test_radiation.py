import os
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

import numpy as np
import pytest
from imas import DBEntry, IDSFactory
from imas.ids_defs import MEMORY_BACKEND
from raysect.primitive import Cylinder, Subtract

import cherab.imas.emitter.radiation as radiation_module
from cherab.imas import _dbentry as dbentry_module
from cherab.imas.emitter import load_radiation_emitter
from cherab.imas.plasma.equilibrium import load_equilibrium


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


class _OpenEntryContext:
    def __init__(self, entry: DBEntry):
        self.entry = entry

    def __enter__(self) -> DBEntry:
        return self.entry

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _patch_dbentry_for_open_entries(
    monkeypatch: pytest.MonkeyPatch,
    entries: dict[tuple, DBEntry],
) -> None:
    original_dbentry = dbentry_module.DBEntry

    def _dbentry_router(*args, **kwargs):
        if not kwargs and args in entries:
            return _OpenEntryContext(entries[args])
        return original_dbentry(*args, **kwargs)

    monkeypatch.setattr(dbentry_module, "DBEntry", _dbentry_router)


def _write_split_radiation_to_memory(
    path_values_dataset: str,
    *,
    include_core: bool,
    include_ggd: bool,
    name: str,
    pulse: int,
    run: int,
) -> tuple[tuple, DBEntry]:
    with DBEntry(path_values_dataset, "r") as entry:
        equilibrium = entry.get("equilibrium", autoconvert=False)
        radiation = entry.get("radiation", autoconvert=False)

    split_radiation = IDSFactory(equilibrium._version).new("radiation")
    split_radiation.ids_properties.homogeneous_time = equilibrium.ids_properties.homogeneous_time
    split_radiation.ids_properties.comment = radiation.ids_properties.comment
    split_radiation.ids_properties.creation_date = radiation.ids_properties.creation_date
    split_radiation.time = np.asarray(radiation.time, dtype=np.float64)

    split_radiation.grid_ggd.resize(1)
    split_radiation.grid_ggd[0] = radiation.grid_ggd[0]

    split_radiation.process.resize(1)
    proc_src = radiation.process[0]
    proc_dst = split_radiation.process[0]
    proc_dst.identifier.index = int(np.asarray(proc_src.identifier.index).item())
    proc_dst.identifier.name = proc_src.identifier.name

    if include_core:
        proc_dst.profiles_1d.resize(1)
        src = proc_src.profiles_1d[0]
        dst = proc_dst.profiles_1d[0]
        dst.grid.rho_tor_norm = np.asarray(src.grid.rho_tor_norm, dtype=np.float64)
        dst.grid.psi = np.asarray(src.grid.psi, dtype=np.float64)
        dst.electrons.emissivity = np.asarray(src.electrons.emissivity, dtype=np.float64)

    if include_ggd:
        proc_dst.ggd.resize(1)
        src = proc_src.ggd[0].electrons.emissivity[0]
        dst = proc_dst.ggd[0].electrons.emissivity
        dst.resize(1)
        dst[0].grid_subset_index = int(np.asarray(src.grid_subset_index).item())
        dst[0].values = np.asarray(src.values, dtype=np.float64)

    args = (MEMORY_BACKEND, name, pulse, run)
    entry = DBEntry(*args, data_version=equilibrium._version)
    entry.create()
    entry.put(equilibrium)
    entry.put(split_radiation)

    return args, entry


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
    d_phi = 360.0 / phis_used.size
    np.testing.assert_allclose(phis_used, np.arange(0.5 * d_phi, 360.0, d_phi, dtype=np.float64))


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


def test_load_radiation_emitter_values_raises_for_jorek(
    path_iter_jorek: str,
    radiation_interpolator_cache: tuple[Literal["memory", "disk"], Path | None],
):
    with pytest.raises(
        RuntimeError,
        match="No emissivity values are available in either core or GGD radiation data.",
    ):
        load_radiation_emitter(
            path_iter_jorek,
            "r",
            source="values",
            **_cache_kwargs(radiation_interpolator_cache),
        )


def test_load_radiation_emitter_values_with_core_and_ggd(
    path_iter_jintrac_radiation_values: str,
    radiation_interpolator_cache: tuple[Literal["memory", "disk"], Path | None],
):
    primitive = load_radiation_emitter(
        path_iter_jintrac_radiation_values,
        "r",
        source="values",
        **_cache_kwargs(radiation_interpolator_cache),
    )

    assert isinstance(primitive, (Subtract, Cylinder))
    assert primitive.material is not None


@pytest.mark.requires_imas_memory_backend
def test_load_radiation_emitter_values_completes_from_second_ids_core_then_ggd(
    path_iter_jintrac_radiation_values: str,
    monkeypatch: pytest.MonkeyPatch,
    radiation_interpolator_cache: tuple[Literal["memory", "disk"], Path | None],
):
    seed = uuid4().int
    name = f"cherab_radiation_values_{seed & 0xFFFF:04x}"
    pulse = 500000 + seed % 100000
    args1, entry1 = _write_split_radiation_to_memory(
        path_iter_jintrac_radiation_values,
        include_core=True,
        include_ggd=False,
        name=name,
        pulse=pulse,
        run=1,
    )
    args2, entry2 = _write_split_radiation_to_memory(
        path_iter_jintrac_radiation_values,
        include_core=False,
        include_ggd=True,
        name=name,
        pulse=pulse,
        run=2,
    )

    try:
        _patch_dbentry_for_open_entries(monkeypatch, {args1: entry1, args2: entry2})
        equilibrium = load_equilibrium(path_iter_jintrac_radiation_values, "r")

        primitive = load_radiation_emitter(
            *args1,
            args2=args2,
            source="values",
            equilibrium=equilibrium,
            **_cache_kwargs(radiation_interpolator_cache),
        )

        assert isinstance(primitive, (Subtract, Cylinder))
        assert primitive.material is not None
    finally:
        entry1.close()
        entry2.close()


@pytest.mark.requires_imas_memory_backend
def test_load_radiation_emitter_values_completes_from_second_ids_ggd_then_core(
    path_iter_jintrac_radiation_values: str,
    monkeypatch: pytest.MonkeyPatch,
    radiation_interpolator_cache: tuple[Literal["memory", "disk"], Path | None],
):
    seed = uuid4().int
    name = f"cherab_radiation_values_{seed & 0xFFFF:04x}"
    pulse = 600000 + seed % 100000
    args1, entry1 = _write_split_radiation_to_memory(
        path_iter_jintrac_radiation_values,
        include_core=False,
        include_ggd=True,
        name=name,
        pulse=pulse,
        run=3,
    )
    args2, entry2 = _write_split_radiation_to_memory(
        path_iter_jintrac_radiation_values,
        include_core=True,
        include_ggd=False,
        name=name,
        pulse=pulse,
        run=4,
    )

    try:
        _patch_dbentry_for_open_entries(monkeypatch, {args1: entry1, args2: entry2})
        equilibrium = load_equilibrium(path_iter_jintrac_radiation_values, "r")

        primitive = load_radiation_emitter(
            *args1,
            args2=args2,
            source="values",
            equilibrium=equilibrium,
            **_cache_kwargs(radiation_interpolator_cache),
        )

        assert isinstance(primitive, (Subtract, Cylinder))
        assert primitive.material is not None
    finally:
        entry1.close()
        entry2.close()


@pytest.mark.requires_imas_memory_backend
def test_load_radiation_emitter_values_raises_for_duplicate_core_in_args2(
    path_iter_jintrac_radiation_values: str,
    monkeypatch: pytest.MonkeyPatch,
    radiation_interpolator_cache: tuple[Literal["memory", "disk"], Path | None],
):
    seed = uuid4().int
    name = f"cherab_radiation_values_{seed & 0xFFFF:04x}"
    pulse = 700000 + seed % 100000
    args1, entry1 = _write_split_radiation_to_memory(
        path_iter_jintrac_radiation_values,
        include_core=True,
        include_ggd=False,
        name=name,
        pulse=pulse,
        run=5,
    )
    args2, entry2 = _write_split_radiation_to_memory(
        path_iter_jintrac_radiation_values,
        include_core=True,
        include_ggd=False,
        name=name,
        pulse=pulse,
        run=6,
    )

    try:
        _patch_dbentry_for_open_entries(monkeypatch, {args1: entry1, args2: entry2})
        equilibrium = load_equilibrium(path_iter_jintrac_radiation_values, "r")

        with pytest.raises(
            RuntimeError,
            match="Duplicate core emissivity values are available in both radiation IDSs.",
        ):
            load_radiation_emitter(
                *args1,
                args2=args2,
                source="values",
                equilibrium=equilibrium,
                **_cache_kwargs(radiation_interpolator_cache),
            )
    finally:
        entry1.close()
        entry2.close()


@pytest.mark.requires_imas_memory_backend
def test_load_radiation_emitter_values_raises_for_duplicate_ggd_in_args2(
    path_iter_jintrac_radiation_values: str,
    monkeypatch: pytest.MonkeyPatch,
    radiation_interpolator_cache: tuple[Literal["memory", "disk"], Path | None],
):
    seed = uuid4().int
    name = f"cherab_radiation_values_{seed & 0xFFFF:04x}"
    pulse = 800000 + seed % 100000
    args1, entry1 = _write_split_radiation_to_memory(
        path_iter_jintrac_radiation_values,
        include_core=False,
        include_ggd=True,
        name=name,
        pulse=pulse,
        run=7,
    )
    args2, entry2 = _write_split_radiation_to_memory(
        path_iter_jintrac_radiation_values,
        include_core=False,
        include_ggd=True,
        name=name,
        pulse=pulse,
        run=8,
    )

    try:
        _patch_dbentry_for_open_entries(monkeypatch, {args1: entry1, args2: entry2})
        equilibrium = load_equilibrium(path_iter_jintrac_radiation_values, "r")

        with pytest.raises(
            RuntimeError,
            match="Duplicate GGD emissivity values are available in both radiation IDSs.",
        ):
            load_radiation_emitter(
                *args1,
                args2=args2,
                source="values",
                equilibrium=equilibrium,
                **_cache_kwargs(radiation_interpolator_cache),
            )
    finally:
        entry1.close()
        entry2.close()


def test_load_radiation_emitter_invalid_source_raises_early(
    monkeypatch: pytest.MonkeyPatch,
):
    class _UnexpectedDBEntry:
        def __init__(self, *args, **kwargs):
            raise AssertionError("DBEntry must not be instantiated for invalid source.")

    monkeypatch.setattr(dbentry_module, "DBEntry", _UnexpectedDBEntry)

    with pytest.raises(
        ValueError,
        match="Invalid source 'unsupported'. Expected one of: 'auto', 'values', 'coefficients'.",
    ):
        load_radiation_emitter(
            "dummy_path",
            "r",
            source="unsupported",  # type: ignore[arg-type]
        )
