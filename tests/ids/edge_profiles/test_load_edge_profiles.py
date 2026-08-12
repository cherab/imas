from imas import DBEntry

from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.common.species import SpeciesType
from cherab.imas.ids.edge_profiles import load_edge_species


def test_load_edge_species_preserves_solps_molecules(path_iter_solps: str) -> None:
    with DBEntry(path_iter_solps, "r") as entry:
        ids = get_ids_time_slice(entry, "edge_profiles", time=0)

    composition = load_edge_species(ids.ggd[0], split_ion_bundles=False)
    d2_charge_states = {
        profile.species.z_min
        for profile in composition.molecule
        if profile.species.species_type is SpeciesType.MOLECULE
        and len(profile.species.elements) == 2
    }

    assert d2_charge_states == {0, 1}
    assert not composition.neutral_bundle
