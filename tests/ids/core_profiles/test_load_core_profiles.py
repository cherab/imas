from types import SimpleNamespace
from typing import cast

import numpy as np
from imas.ids_defs import EMPTY_FLOAT
from imas.ids_structure import IDSStructure

from cherab.imas.ids.common.species import ProfileData, SpeciesType
from cherab.imas.ids.core_profiles import load_profiles


def test_load_core_species_preserves_molecular_ion(monkeypatch) -> None:
    state = SimpleNamespace(
        z_min=1.0,
        z_max=1.0,
        electron_configuration="",
        vibrational_mode="",
        vibrational_level=EMPTY_FLOAT,
    )
    molecular_ion = SimpleNamespace(
        element=[SimpleNamespace(a=2.0, z_n=1.0, atoms_n=2)],
        state=[state],
    )
    profile_1d = SimpleNamespace(electrons=object(), ion=[molecular_ion], neutral=[])

    def fake_load_core_profiles(_structure, species, backup_species_struct=None):
        return ProfileData(
            species=species,
            density=np.ones(1),
            temperature=np.ones(1),
        )

    monkeypatch.setattr(load_profiles, "load_core_profiles", fake_load_core_profiles)
    monkeypatch.setattr(load_profiles, "_get_profile", lambda *_args, **_kwargs: None)

    composition = load_profiles.load_core_species(
        cast(IDSStructure, profile_1d), split_ion_bundles=False
    )

    assert len(composition.molecule) == 1
    assert composition.molecule[0].species.species_type is SpeciesType.MOLECULE
    assert composition.molecule[0].species.z_min == 1
