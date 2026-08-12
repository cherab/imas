from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from imas.ids_defs import EMPTY_FLOAT
from imas.ids_structure import IDSStructure
from rich.text import Text
from rich.tree import Tree

from cherab.core.atomic.elements import deuterium, helium, hydrogen, neon, tritium
from cherab.imas.ids.common.species import (
    ProfileData,
    SpeciesComposition,
    SpeciesData,
    SpeciesType,
    get_ion_state,
    get_neutral_state,
)


@dataclass
class ExtendedProfileData(ProfileData):
    pressure: np.ndarray | None = None


@dataclass
class ExtendedSpeciesComposition(SpeciesComposition):
    diagnostic: list[ExtendedProfileData] = field(default_factory=list)


@pytest.mark.parametrize(
    ("vibrational_mode", "vibrational_level"),
    [
        ("", EMPTY_FLOAT),
        ("A_g", 1.0),
    ],
)
def test_get_neutral_state_classifies_d2_as_molecule(
    vibrational_mode: str,
    vibrational_level: float,
) -> None:
    state = SimpleNamespace(
        electron_configuration="",
        vibrational_mode=vibrational_mode,
        vibrational_level=vibrational_level,
    )

    species = get_neutral_state(cast(IDSStructure, state), (deuterium, deuterium))

    assert species.species_type is SpeciesType.MOLECULE
    assert species.elements == (deuterium, deuterium)
    assert species.vibrational_mode == (vibrational_mode or None)
    assert species.vibrational_level == (
        vibrational_level if vibrational_level != EMPTY_FLOAT else None
    )


@pytest.mark.parametrize(
    ("z_min", "z_max", "species_type"),
    [
        (0.0, 0.0, SpeciesType.MOLECULE),
        (1.0, 1.0, SpeciesType.MOLECULE),
        (1.0, 2.0, SpeciesType.MOLECULAR_BUNDLE),
    ],
)
def test_get_ion_state_classifies_molecular_species(
    z_min: float,
    z_max: float,
    species_type: SpeciesType,
) -> None:
    state = SimpleNamespace(
        z_min=z_min,
        z_max=z_max,
        electron_configuration="",
        vibrational_mode="",
        vibrational_level=EMPTY_FLOAT,
    )

    species = get_ion_state(cast(IDSStructure, state), 0, (deuterium, deuterium))

    assert species.species_type is species_type
    assert species.elements == (deuterium, deuterium)
    assert species.z_min == int(z_min)
    assert species.z_max == int(z_max)


def test_molecular_ion_string_includes_charge() -> None:
    state = SimpleNamespace(
        z_min=1.0,
        z_max=1.0,
        electron_configuration="",
        vibrational_mode="",
        vibrational_level=EMPTY_FLOAT,
    )

    species = get_ion_state(cast(IDSStructure, state), 0, (deuterium, deuterium))

    assert str(species) == "D-D molecule (z=+1)"
    assert species.compact_label() == "D-D +"


@pytest.mark.parametrize(
    ("z_min", "z_max", "expected"),
    [
        (0, 0, "Ne"),
        (1, 1, "Ne +"),
        (3, 3, "Ne 3+"),
        (1, 10, "Ne 1+–10+"),
    ],
)
def test_compact_label_uses_linearized_ionic_charge(z_min: int, z_max: int, expected: str) -> None:
    species_type = SpeciesType.ION if z_min == z_max else SpeciesType.ION_BUNDLE
    species = SpeciesData(
        z_min=z_min,
        z_max=z_max,
        element=neon,
        species_type=species_type,
    )

    assert species.compact_label() == expected


def test_species_rich_color_is_shared_across_charge_and_molecular_states() -> None:
    d_ion = SpeciesData(
        z_min=1,
        z_max=1,
        element=deuterium,
        species_type=SpeciesType.ION,
    )
    d2_ion = SpeciesData(
        z_min=1,
        z_max=1,
        elements=(deuterium, deuterium),
        species_type=SpeciesType.MOLECULE,
    )
    he_ion = SpeciesData(z_min=1, z_max=1, element=helium, species_type=SpeciesType.ION)

    assert d_ion.__rich__().style == d2_ion.__rich__().style
    assert d_ion.__rich__().style != he_ion.__rich__().style


def test_species_rich_color_distinguishes_isotopes() -> None:
    styles = {
        SpeciesData(z_min=0, z_max=0, element=isotope, species_type=SpeciesType.NEUTRAL)
        .__rich__()
        .style
        for isotope in (hydrogen, deuterium, tritium)
    }

    assert len(styles) == 3
    assert all(str(style).startswith("bold #") for style in styles)


def test_species_rich_color_includes_charge() -> None:
    species = SpeciesData(z_min=3, z_max=3, element=neon, species_type=SpeciesType.ION)

    label = species.rich_label(compact=True)

    assert label.plain == "Ne 3+"
    assert str(label.style).startswith("bold #")
    assert not label.spans


def test_species_composition_string_summarizes_species_and_profile_shapes() -> None:
    density = np.ones(3)
    electron = ProfileData(SpeciesData(z_min=-1, z_max=-1), density=density)
    d2 = ProfileData(
        SpeciesData(
            z_min=0,
            z_max=0,
            elements=(deuterium, deuterium),
            species_type=SpeciesType.MOLECULE,
        ),
        density=density,
    )
    d2_ion = ProfileData(
        SpeciesData(
            z_min=1,
            z_max=1,
            elements=(deuterium, deuterium),
            species_type=SpeciesType.MOLECULE,
        ),
        density=density,
    )
    composition = SpeciesComposition(electron=electron, molecule=[d2, d2_ion])

    expected = """SpeciesComposition
  electron
    density: shape=(3,)
  molecule (2)
    - D-D
      density: shape=(3,)
    - D-D +
      density: shape=(3,)"""

    assert str(composition) == expected
    assert "[1. 1. 1.]" not in str(composition)
    assert repr(composition).startswith("SpeciesComposition(electron=ProfileData(")

    rich_tree = composition.__rich__()
    assert isinstance(rich_tree, Tree)
    assert isinstance(rich_tree.label, Text)
    assert rich_tree.label.plain == "SpeciesComposition"
    assert all(isinstance(branch.label, Text) for branch in rich_tree.children)
    assert [cast(Text, branch.label).plain for branch in rich_tree.children] == [
        "electron",
        "molecule (2)",
    ]
    assert all(isinstance(node.label, Text) for node in rich_tree.children[1].children)
    molecule_labels = [cast(Text, node.label).plain for node in rich_tree.children[1].children]
    assert molecule_labels == ["D-D", "D-D +"]


def test_species_summary_discovers_new_dataclass_fields() -> None:
    electron = ProfileData(SpeciesData(z_min=-1, z_max=-1))
    diagnostic = ExtendedProfileData(
        SpeciesData(z_min=1, z_max=1, element=deuterium, species_type=SpeciesType.ION),
        pressure=np.ones((2, 4)),
    )
    composition = ExtendedSpeciesComposition(electron=electron, diagnostic=[diagnostic])

    summary = str(composition)

    assert "  diagnostic\n" in summary
    assert "diagnostic (1)" not in summary
    assert "pressure: shape=(2, 4)" in summary
