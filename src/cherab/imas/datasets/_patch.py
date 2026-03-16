"""Module including patch functionalities for datasets."""

from cherab.imas.ids.common import get_ids_time_slice
from imas import DBEntry


def fix_jintract(path_in: str, path_out: str) -> None:
    """Fix the JINTRAC IDS.

    This function modifies the JINTRAC IDS by updating the `z_min` and `z_max` values for the ion
    states of Neon and Tungsten in the `core_profiles` IDS.

    The modified IDS is then saved to a new file with a "_mod" suffix.

    Parameters
    ----------
    path_in
        The file path to the original JINTRAC IDS.
    path_out
        The file path to save the modified IDS.
    """
    print("=== Apply patch to fix ===")
    # %%
    # Define the bundles for Neon and Tungsten
    # ----------------------------------------
    BUNDLES_W = [
        (1, 1),
        (2, 6),
        (7, 12),
        (13, 22),
        (23, 73),
        (74, 74),
    ]
    BUNDLES_NE = [
        (1, 1),
        (2, 3),
        (4, 6),
        (7, 9),
        (10, 10),
    ]

    # %%
    # Load the IDS from the JINTRAC simulation
    # ----------------------------------------
    with DBEntry(path_in, "r+") as entry:
        ids_core = get_ids_time_slice(entry, "core_profiles", 0.0)
        ids_others = [
            entry.get("edge_profiles", autoconvert=False),
            entry.get("core_sources", autoconvert=False),
            entry.get("edge_sources", autoconvert=False),
            entry.get("edge_transport", autoconvert=False),
            entry.get("equilibrium", autoconvert=False),
            entry.get("ntms", autoconvert=False),
            entry.get("summary", autoconvert=False),
        ]

    # %%
    # Update the z_min and z_max values for the ion states based on the defined bundles
    # ---------------------------------------------------------------------------------
    for i_ion in [3, 4]:
        ion = ids_core.profiles_1d[0].ion[i_ion]
        bundles = BUNDLES_NE if i_ion == 3 else BUNDLES_W
        for i, state in enumerate(ion.state):
            state.z_min = bundles[i][0]
            state.z_max = bundles[i][1]

    # %%
    # Save the modified IDS to a new file
    # -----------------------------------
    with DBEntry(path_out, "w", dd_version=ids_core._version) as entry:
        entry.put(ids_core)
        for ids in ids_others:
            entry.put(ids)
