from uuid import uuid4

import pytest
from imas import DBEntry
from imas.ids_defs import MEMORY_BACKEND
from imas.ids_factory import IDSFactory
from imas.ids_toplevel import IDSToplevel

from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.common.ggd import load_grid


def _build_jintrac_external_grid_reference_entry(
    path_iter_jintrac: str,
) -> tuple[DBEntry, IDSToplevel, IDSToplevel]:
    with DBEntry(path_iter_jintrac, "r") as source_entry:
        try:
            edge_profiles = source_entry.get("edge_profiles", autoconvert=False)
        except Exception as exc:
            raise RuntimeError(f"JINTRAC dataset does not provide required IDSs: {exc}") from exc

    if not len(edge_profiles.grid_ggd):
        raise RuntimeError(
            "JINTRAC dataset does not provide grid_ggd data for this integration test."
        )

    # Make radiation point to edge_profiles grid and remove its local space.
    radiation = IDSFactory(edge_profiles._version).new("radiation")
    radiation.ids_properties.homogeneous_time = edge_profiles.ids_properties.homogeneous_time
    radiation.time = edge_profiles.time
    radiation.grid_ggd.resize(1)
    radiation.grid_ggd[0].space.resize(0)
    radiation.grid_ggd[0].path = "#edge_profiles/grid_ggd(1)"

    token = uuid4().int
    entry = DBEntry(
        MEMORY_BACKEND,
        f"cherab_grid_ref_{token & 0xFFFF:04x}",
        1 + token % 2_000_000_000,
        1 + (token >> 31) % 2_000_000_000,
        data_version=radiation._version,
    )
    entry.create()
    entry.put(edge_profiles)
    entry.put(radiation)

    radiation_ids = get_ids_time_slice(entry, "radiation", time=0)
    edge_ids = get_ids_time_slice(entry, "edge_profiles", time=0)
    return entry, radiation_ids, edge_ids


@pytest.mark.requires_imas_memory_backend
def test_load_grid_resolves_external_reference_on_jintrac_dataset(path_iter_jintrac: str):
    entry, radiation_ids, edge_ids = _build_jintrac_external_grid_reference_entry(path_iter_jintrac)

    try:
        resolved_grid = load_grid(radiation_ids.grid_ggd[0], with_subsets=False, entry=entry)
        expected_grid = load_grid(edge_ids.grid_ggd[0], with_subsets=False)

        assert type(resolved_grid) is type(expected_grid)
        assert resolved_grid.num_cell == expected_grid.num_cell
    finally:
        entry.close()


@pytest.mark.requires_imas_memory_backend
def test_load_grid_requires_entry_for_external_reference_on_jintrac_dataset(path_iter_jintrac: str):
    entry, radiation_ids, _ = _build_jintrac_external_grid_reference_entry(path_iter_jintrac)

    try:
        with pytest.raises(RuntimeError, match="without a DBEntry"):
            load_grid(radiation_ids.grid_ggd[0], with_subsets=False)
    finally:
        entry.close()
