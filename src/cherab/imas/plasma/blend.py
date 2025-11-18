# Copyright 2023 Euratom
# Copyright 2023 United Kingdom Atomic Energy Authority
# Copyright 2023 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.
"""Module to offer the plasma object, blending core with edge profiles."""

from collections.abc import Callable

import numpy as np
from raysect.core.math import Vector3D, translate
from raysect.core.math.function.float import Blend2D, Blend3D, Function2D, Function3D
from raysect.core.math.function.vector3d import Blend2D as VectorBlend2D
from raysect.core.math.function.vector3d import Blend3D as VectorBlend3D
from raysect.core.math.function.vector3d import Constant3D as ConstantVector3D
from raysect.core.math.function.vector3d import Function2D as VectorFunction2D
from raysect.core.math.function.vector3d import Function3D as VectorFunction3D
from raysect.core.scenegraph._nodebase import _NodeBase
from raysect.primitive import Cylinder, Subtract
from scipy.constants import atomic_mass, electron_mass

from cherab.core import Maxwellian, Plasma, Species
from cherab.core.math import AxisymmetricMapper, VectorAxisymmetricMapper
from cherab.tools.equilibrium import EFITEquilibrium
from imas import DBEntry
from imas.ids_structure import IDSStructure

from ..ids.common import get_ids_time_slice
from ..ids.common.ggd import load_grid
from ..ids.core_profiles import load_core_grid, load_core_species
from ..ids.edge_profiles import load_edge_species
from .core import get_core_interpolators, get_psi_norm, load_core_plasma
from .edge import get_edge_interpolators, load_edge_plasma
from .equilibrium import load_equilibrium, load_magnetic_field
from .utility import get_subset_name_index, warn_unsupported_species

__all__ = ["load_plasma"]


def load_plasma(
    *args,
    time: float = 0,
    occurrence_core: int = 0,
    edge_args: tuple | None = None,
    edge_kwargs: dict | None = None,
    time_edge: float | None = None,
    occurrence_edge: int = 0,
    grid_ggd: IDSStructure | None = None,
    grid_subset_id: int | str = 5,
    equilibrium: EFITEquilibrium | None = None,
    b_field: VectorFunction2D | None = None,
    psi_interpolator: Callable[[float], float] | None = None,
    mask: Function2D | Function3D | None = None,
    time_threshold: float = np.inf,
    parent: _NodeBase | None = None,
    **kwargs,
) -> Plasma:
    """Load core and edge profiles and create a `~cherab.core.plasma.node.Plasma` object.

    If the ``edge_profiles`` IDS is empty, returns only the core plasma.
    If the ``core_profiles`` IDS is empty, returns only the edge plasma.

    To load the edge plasma from a different IMAS entry, use `edge_args` and `edge_kwargs`
    to pass different arguments to the `~imas.db_entry.DBEntry` constructor.

    Parameters
    ----------
    *args
        Arguments passed to the `~imas.db_entry.DBEntry` constructor.
    time
        Time for the core plasma, by default 0.
    occurrence_core
        Occurrence index of the ``core_profiles`` IDS, by default 0.
    edge_args
        Arguments passed to the `~imas.db_entry.DBEntry` constructor for the edge plasma
        if different from the core plasma. By default None: uses the same as `*args`.
    edge_kwargs
        Keyword arguments passed to the `~imas.db_entry.DBEntry` constructor for the edge plasma
        if different from the core plasma. By default None: uses the same as `**kwargs`.
    time_edge
        Time for the edge plasma. If None, uses `time`. By default None.
    occurrence_edge
        Occurrence index of the ``edge_profiles`` IDS, by default 0.
    grid_ggd
        Alternative ``grid_ggd`` structure describing the grid. By default None.
    grid_subset_id
        Identifier of the grid subset (index or name). By default 5 (``"Cells"``).
    equilibrium
        Alternative `~cherab.tools.equilibrium.efit.EFITEquilibrium` used to map core
        profiles. By default None: the equilibrium is read from the same IMAS query as the
        core profiles. Ignored if the core plasma is not available.
    b_field
        Alternative 2D interpolator of the magnetic field vector (Br, Bphi, Bz).
        By default None: the magnetic field is loaded from the ``equilibrium`` IDS.
    psi_interpolator
        Alternative ``psi_norm(rho_tor_norm)`` interpolator.
        Used only if ``psi`` is missing in the core grid. By default None.
        Obtained from the ``equilibrium`` IDS.
    mask
        Mask function used for blending: ``(1 - mask) * f_edge + mask * f_core``.
        By default, uses `~cherab.tools.equilibrium.efit.EFITEquilibrium`'s `inside_lcfs`.
    time_threshold
        Maximum allowed difference between the requested time and the nearest
        available time, by default `numpy.inf`.
    parent
        Parent node in the Raysect scene graph, by default None.
        Typically a `~raysect.optical.scenegraph.world.World` instance.
    **kwargs
        Keyword arguments passed to the `~imas.db_entry.DBEntry` constructor.

    Returns
    -------
    `~cherab.core.plasma.node.Plasma`
        Plasma object with core and/or edge profiles.

    Raises
    ------
    RuntimeError
        If neither core nor edge profiles are available.
    """
    edge_args = edge_args or args
    edge_kwargs = edge_kwargs or kwargs
    if time_edge is None:
        time_edge = time

    try:
        with DBEntry(*args, **kwargs) as entry_core:
            core_profiles_ids = get_ids_time_slice(
                entry_core,
                "core_profiles",
                time=time,
                occurrence=occurrence_core,
                time_threshold=time_threshold,
            )
    except RuntimeError:
        return load_edge_plasma(
            *edge_args,
            time=time_edge,
            occurrence=occurrence_edge,
            grid_ggd=grid_ggd,
            grid_subset_id=grid_subset_id,
            b_field=b_field,
            time_threshold=time_threshold,
            parent=parent,
            **edge_kwargs,
        )

    try:
        with DBEntry(*edge_args, **edge_kwargs) as entry_edge:
            edge_profiles_ids = get_ids_time_slice(
                entry_edge,
                "edge_profiles",
                time=time_edge,
                occurrence=occurrence_edge,
                time_threshold=time_threshold,
            )
    except RuntimeError:
        return load_core_plasma(
            *args,
            time=0,
            occurrence=0,
            equilibrium=equilibrium,
            b_field=b_field,
            psi_interpolator=psi_interpolator,
            time_threshold=time_threshold,
            parent=parent,
            **kwargs,
        )

    if not len(core_profiles_ids.profiles_1d):
        raise RuntimeError("The profiles_1d AOS in core_profiles IDS is empty.")

    if not len(edge_profiles_ids.grid_ggd) and grid_ggd is None:
        raise RuntimeError(
            "The 'grid_ggd' AOS of the edge_profiles IDS is empty "
            + "and an alternative grid_ggd structure is not provided."
        )

    if not len(edge_profiles_ids.ggd):
        raise RuntimeError("The 'ggd' AOS of the edge_profiles IDS is empty.")

    # Getting equilibrium and magnetic field
    if equilibrium is None:
        equilibrium, psi_interp = load_equilibrium(
            *args, time=time, with_psi_interpolator=True, **kwargs
        )
        psi_interpolator = psi_interpolator or psi_interp

    if b_field is None:
        try:
            b_field = load_magnetic_field(*args, time=time, **kwargs)
        except RuntimeError:
            b_field = equilibrium.b_field

    mask = mask or equilibrium.inside_lcfs

    # Getting core data
    core_grid = load_core_grid(core_profiles_ids.profiles_1d[0].grid)

    composition_core = load_core_species(core_profiles_ids.profiles_1d[0])

    psi_norm = get_psi_norm(
        core_grid["psi"],
        equilibrium.psi_axis,
        equilibrium.psi_lcfs,
        core_grid["rho_tor_norm"],
        psi_interpolator,
    )

    # Getting edge data
    grid_ggd = grid_ggd or edge_profiles_ids.grid_ggd[0]
    grid, subsets, subset_id = load_grid(grid_ggd, with_subsets=True)

    grid_subset_name, grid_subset_index = get_subset_name_index(subset_id, grid_subset_id)

    if np.all(subsets[grid_subset_name] != np.arange(grid.num_cell, dtype=int)):
        # To reduce memory usage, create the sub-grid only if needed.
        grid = grid.subset(subsets[grid_subset_name], name=grid_subset_name)

    composition_edge = load_edge_species(
        edge_profiles_ids.ggd[0], grid_subset_index=grid_subset_index
    )

    # Creating plasma
    tcore = core_profiles_ids.time[0]
    tedge = edge_profiles_ids.time[0]
    name = (
        f"IMAS core + edge plasma: core/edge time {tcore}/{tedge}, "
        f"uri {entry_core.uri} / {entry_edge.uri}."
    )
    plasma = Plasma(parent=parent, name=name)

    # Create plasma geometry
    radius_outer = grid.mesh_extent["rmax"]
    radius_inner = grid.mesh_extent["rmin"]
    height = grid.mesh_extent["zmax"] - grid.mesh_extent["zmin"]
    zmin = grid.mesh_extent["zmin"]
    plasma.geometry = Subtract(Cylinder(radius_outer, height), Cylinder(radius_inner, height))
    plasma.geometry_transform = translate(0, 0, zmin)

    plasma.b_field = VectorAxisymmetricMapper(b_field)

    # Add electron species
    electrons_core = get_core_interpolators(
        psi_norm, composition_core["electron"], equilibrium, return3d=False
    )
    if electrons_core["density_thermal"] is not None:
        electrons_core["density"] = electrons_core["density_thermal"]

    electrons_edge = get_edge_interpolators(
        grid, composition_edge["electron"], b_field, return3d=False
    )

    electrons = blend_core_edge_interpolators(electrons_core, electrons_edge, mask, return3d=True)

    if electrons["density"] is None:
        print("Unable to create Core Plasma: electron density is not available.")
    if electrons["temperature"] is None:
        print("Unable to create Core Plasma: electron temperature is not available.")

    plasma.electron_distribution = Maxwellian(
        electrons["density"], electrons["temperature"], electrons["velocity"], electron_mass
    )

    warn_unsupported_species(composition_core, "molecule")
    warn_unsupported_species(composition_edge, "molecule")
    warn_unsupported_species(composition_core, "molecular_bundle")
    warn_unsupported_species(composition_edge, "molecular_bundle")
    warn_unsupported_species(composition_core, "ion_bundle")
    warn_unsupported_species(composition_edge, "ion_bundle")

    # -----------------------------------
    # === Add ion & neutral species ===
    # -----------------------------------
    # List core species
    core_species = {}
    for species_id, profiles in composition_core["ion"].items():
        d = {first: second for first, second in species_id}
        sp_key = (d["element"], int(round(d["z"])))
        if sp_key in core_species:
            print(
                f"Warning! Skipping {d['name']} core species. "
                + f"Species with the same (element, charge): {sp_key} is already added."
            )
            continue
        if profiles["density_thermal"] is not None:
            profiles["density"] = profiles["density_thermal"]
        core_species[sp_key] = profiles

    # List edge species
    edge_species = {}
    for species_id, profiles in composition_edge["ion"].items():
        d = {first: second for first, second in species_id}
        sp_key = (d["element"], int(round(d["z"])))
        if sp_key in edge_species:
            print(
                f"Warning! Skipping {d['name']} edge species. "
                + f"Species with the same (element, charge): {sp_key} is already added."
            )
            continue
        edge_species[sp_key] = profiles

    species = {}
    for core_key, core_profiles in core_species.items():
        if core_key in edge_species:
            edge_profiles = edge_species[core_key]
            core_interp = get_core_interpolators(
                psi_norm, core_profiles, equilibrium, return3d=False
            )
            edge_interp = get_edge_interpolators(grid, edge_profiles, b_field, return3d=False)
            species[core_key] = blend_core_edge_interpolators(
                core_interp, edge_interp, mask, return3d=True
            )
        else:
            species[core_key] = get_core_interpolators(
                psi_norm, core_profiles, equilibrium, return3d=True
            )
    for edge_key, edge_profiles in edge_species.items():
        if edge_key not in core_species:
            species[edge_key] = get_edge_interpolators(grid, edge_profiles, b_field, return3d=True)

    for (species_type, charge), interp in species.items():
        if "velocity" not in interp or interp["velocity"] is None:
            interp["velocity"] = ConstantVector3D(Vector3D(0, 0, 0))

        if interp["density"] is None:
            print(f"Warning! Skipping {d['name']} species: density is not available.")
        if interp["temperature"] is None:
            print(f"Warning! Skipping {d['name']} species: temperature is not available.")

        distribution = Maxwellian(
            interp["density"],
            interp["temperature"],
            interp["velocity"],
            species_type.atomic_weight * atomic_mass,
        )

        plasma.composition.add(Species(species_type, charge, distribution))

    return plasma


def blend_core_edge_interpolators(
    core_interpolators: dict[str, Function3D | VectorFunction3D | None],
    edge_interpolators: dict[str, Function3D | VectorFunction3D | None],
    mask: Function2D | Function3D,
    return3d: bool = False,
) -> dict[str, Function3D | VectorFunction3D | None]:
    """Blend together interpolators for the core and edge using the modulating mask function.

    Parameters
    ----------
    core_interpolators
        Dictionary with 2D or 3D core profiles interpolators.
    edge_interpolators
        Dictionary with 2D or 3D edge profiles interpolators.
    mask
        Mask function used for blending: ``(1 - mask) * f_edge + mask * f_core``.
    return3d
        If True, return the 3D functions for 2D interpolators assuming
        rotational symmetry, by default False.

    Returns
    -------
    dict
        Dictionary with blended interpolators.
    """
    interpolators = {}

    for core_key, core_func in core_interpolators.items():
        edge_func = edge_interpolators[core_key] if core_key in edge_interpolators else None
        interpolators[core_key] = blend_core_edge_functions(core_func, edge_func, mask, return3d)

    for edge_key, edge_func in edge_interpolators.items():
        if edge_key not in core_interpolators:
            interpolators[edge_key] = blend_core_edge_functions(None, edge_func, mask, return3d)

    return interpolators


def blend_core_edge_functions(
    core_func: Function2D | Function3D | VectorFunction2D | VectorFunction3D | None,
    edge_func: Function2D | Function3D | VectorFunction2D | VectorFunction3D | None,
    mask: Function2D | Function3D,
    return3d: bool,
) -> Function3D | VectorFunction3D | None:
    """Blend together the core and edge interpolating functions using the modulating mask function.

    Parameters
    ----------
    core_func
        A 2D or 3D core interpolator.
    edge_func
        A 2D or 3D edge interpolator.
    mask
        The 2D or 3D mask function used for blending: ``(1 - mask) * f_edge + mask * f_core``.
    return3d
        If True, return the 3D functions for 2D interpolators assuming
        rotational symmetry, by default False.

    Returns
    -------
    Function3D | VectorFunction3D | None
        Blended interpolator.

    Raises
    ------
    ValueError
        If both core and edge functions are None.
    """
    if core_func is None and edge_func is None:
        return None

    if core_func is not None and not isinstance(
        core_func, Function2D | Function3D | VectorFunction2D | VectorFunction3D
    ):
        raise ValueError("The core_func must be a 2D or 3D function.")

    if edge_func is not None and not isinstance(
        edge_func, Function2D | Function3D | VectorFunction2D | VectorFunction3D
    ):
        raise ValueError("The edge_func must be a 2D or 3D function.")

    if not isinstance(mask, Function2D | Function3D):
        raise ValueError("The mask must be a 2D or 3D function.")

    if core_func is None:
        if isinstance(edge_func, Function2D) and return3d:
            return AxisymmetricMapper(edge_func)

        if isinstance(edge_func, VectorFunction2D) and return3d:
            return VectorAxisymmetricMapper(edge_func)

        return edge_func

    if edge_func is None:
        if isinstance(core_func, Function2D) and return3d:
            return AxisymmetricMapper(core_func)

        if isinstance(core_func, VectorFunction2D) and return3d:
            return VectorAxisymmetricMapper(core_func)

        return core_func

    if (
        isinstance(core_func, Function2D)
        and isinstance(edge_func, Function2D)
        and isinstance(mask, Function2D)
    ):
        blended_func = Blend2D(edge_func, core_func, mask)
        return AxisymmetricMapper(blended_func) if return3d else blended_func

    if (
        isinstance(core_func, VectorFunction2D)
        and isinstance(edge_func, VectorFunction2D)
        and isinstance(mask, Function2D)
    ):
        blended_func = VectorBlend2D(edge_func, core_func, mask)
        return VectorAxisymmetricMapper(blended_func) if return3d else blended_func

    # unable to return 2D, convert to 3D

    if isinstance(core_func, Function2D):
        core_func = AxisymmetricMapper(core_func)

    if isinstance(core_func, VectorFunction2D):
        core_func = VectorAxisymmetricMapper(core_func)

    if isinstance(edge_func, Function2D):
        edge_func = AxisymmetricMapper(edge_func)

    if isinstance(edge_func, VectorFunction2D):
        edge_func = VectorAxisymmetricMapper(edge_func)

    if isinstance(mask, Function2D):
        mask = AxisymmetricMapper(mask)

    if isinstance(core_func, Function3D) and isinstance(edge_func, Function3D):
        return Blend3D(edge_func, core_func, mask)

    if isinstance(core_func, VectorFunction3D) and isinstance(edge_func, VectorFunction3D):
        return VectorBlend3D(edge_func, core_func, mask)

    raise ValueError("Cannot blend scalar and vector functions.")
