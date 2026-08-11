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
"""Module for loading radiation emissivity from IMAS-like IDS objects and creating emitter objects."""

from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from raysect.core.math import translate
from raysect.core.math.function.float import (
    Function2D,
    Function3D,
    Interpolator1DArray,
)
from raysect.core.scenegraph._nodebase import _NodeBase
from raysect.primitive import Cylinder, Subtract

from cherab.core.math import AxisymmetricMapper
from cherab.tools.emitters import RadiationFunction
from cherab.tools.equilibrium import EFITEquilibrium
from imas.ids_struct_array import IDSStructArray
from imas.ids_structure import IDSStructure

from .._dbentry import _open_dbentry_for_reading
from ..ggd import UnstructGrid2DExtended
from ..ggd.base_mesh import InterpolatorCacheMode
from ..ids.common import get_ids_time_slice, load_ids_path_reference, resolve_ids_path_reference
from ..ids.common.ggd import load_grid
from ..ids.common.grid_radial import GridData, get_psi_norm, load_core_grid
from ..ids.radiation import load_core_emissivity, load_ggd_emissivity
from ..math import FourierBezierConstructor
from ..math.blend import blend_core_edge_functions
from ..plasma.equilibrium import load_equilibrium
from ..plasma.utility import get_entry_reference, get_subset_name_index

__all__ = ["load_radiation_emitter"]


def _load_emissivity_values(
    processes: IDSStructArray,
    process_indices: Collection[int] | None,
    grid_subset_id: int,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    values_core = None
    values_ggd = None

    for process in processes:
        # Validate process
        if process_indices is not None and process.identifier.index.value not in process_indices:
            continue

        # Values (profiles_1d)
        _values = load_core_emissivity(process).sum()

        if _values is not None:
            if values_core is None:
                values_core = _values
            else:
                values_core += _values

        # Values (GGD)
        _values = load_ggd_emissivity(
            process,
            grid_subset_index=grid_subset_id,
            field="values",
        ).sum()

        if _values is not None:
            if values_ggd is None:
                values_ggd = _values
            else:
                values_ggd += _values

    return values_core, values_ggd


def _create_rad_func_core(
    grid: GridData,
    data: NDArray[np.float64],
    equilibrium: EFITEquilibrium | None,
    psi_interpolator: Callable[[float], float] | None,
    db_args: tuple | None,
    db_kwargs: dict[str, Any] | None,
    time: float,
    occurrence: int,
) -> tuple[AxisymmetricMapper, EFITEquilibrium]:
    if equilibrium is None:
        equilibrium, psi_interp = load_equilibrium(
            *(db_args or ()),
            time=time,
            occurrence=occurrence,
            with_psi_interpolator=True,
            **(db_kwargs or {}),
        )
        psi_interpolator = psi_interpolator or psi_interp
    else:
        if not isinstance(equilibrium, EFITEquilibrium):
            raise ValueError("Argument equilibrium must be a EFITEquilibrium instance.")

    # Create core grid
    psi_norm = get_psi_norm(
        grid.psi,
        equilibrium.psi_axis,
        equilibrium.psi_lcfs,
        grid.rho_tor_norm,
        psi_interpolator,
    )
    psi_norm, index = np.unique(psi_norm, return_index=True)
    extrapolation_range = max(0.0, psi_norm[0], 1.0 - psi_norm[-1])
    rad_func = equilibrium.map3d(
        Interpolator1DArray(psi_norm, data[index], "cubic", "nearest", extrapolation_range)
    )

    return rad_func, equilibrium


def _create_rad_func_ggd(
    grid_ggd: IDSStructure,
    data: NDArray[np.float64],
    grid_subset_id: int,
    ids_root: IDSStructure,
    db_args: tuple[Any, ...] | None,
    db_kwargs: dict[str, Any] | None,
    **interp_kwargs,
) -> tuple[AxisymmetricMapper, dict[str, float]]:
    grid_ggd = _resolve_grid_ggd_reference(grid_ggd, ids_root, db_args, db_kwargs)
    grid, subsets, subset_id = load_grid(grid_ggd, with_subsets=True)
    grid_subset_name, grid_subset_index = get_subset_name_index(subset_id, grid_subset_id)

    subset_indices, subset_mask = subsets[grid_subset_name]
    if not np.array_equal(subset_indices, np.arange(grid.num_cell, dtype=int)):
        grid = grid.subset(subset_indices, name=grid_subset_name, valid_data_mask=subset_mask)

    rad_func = AxisymmetricMapper(grid.interpolator(data, **interp_kwargs))

    return rad_func, grid.mesh_extent


def _resolve_grid_ggd_reference(
    grid_ggd: IDSStructure,
    ids_root: IDSStructure,
    db_args: tuple[Any, ...] | None,
    db_kwargs: dict[str, Any] | None,
) -> IDSStructure:
    if len(grid_ggd.space):
        return grid_ggd

    if not len(grid_ggd.path):
        return grid_ggd

    path = str(grid_ggd.path).strip()
    if not path:
        return grid_ggd

    if "#" in path:
        if db_args is None:
            raise RuntimeError(
                "Unable to resolve external grid_ggd.path without DBEntry arguments."
            )
        with _open_dbentry_for_reading(*db_args, **(db_kwargs or {})) as entry:
            resolved = load_ids_path_reference(entry, path)
    else:
        ids_name = ids_root.metadata.name
        if path.startswith("/"):
            resolved = resolve_ids_path_reference(ids_root, f"#{ids_name}{path}")
        else:
            resolved = resolve_ids_path_reference(ids_root, path)

    if isinstance(resolved, IDSStructArray):
        if not len(resolved):
            raise RuntimeError(f"Resolved grid reference '{path}' points to an empty array.")
        resolved = resolved[0]

    if not isinstance(resolved, IDSStructure) or not hasattr(resolved, "space"):
        raise RuntimeError(
            f"Resolved grid reference '{path}' does not point to a grid_ggd structure."
        )

    return resolved


def load_radiation_emitter(
    *args,
    time: float = 0,
    occurrence: int = 0,
    args2: tuple | None = None,
    kwargs2: dict[str, Any] | None = None,
    time2: float | None = None,
    occurrence2: int = 0,
    process_index: int | Collection[int] | None = None,
    grid_ggd: IDSStructure | None = None,
    grid_subset_id: int = 5,
    equilibrium: EFITEquilibrium | None = None,
    psi_interpolator: Callable[[float], float] | None = None,
    mask: Function2D | Function3D | None = None,
    num_toroidal: int = 64,
    source: Literal["auto", "values", "coefficients"] = "auto",
    time_threshold: float = np.inf,
    step: float = 0.01,
    parent: _NodeBase | None = None,
    interpolator_cache: InterpolatorCacheMode = "memory",
    interpolator_cache_dir: str | Path | None = None,
    **kwargs,
) -> Subtract | Cylinder:
    """Load radiation emissivity and create a single radiation emitter primitive.

    There are two sources of emissivity data in the IMAS radiation IDS:
        1. core-region emissivity and/or edge-region emissivity (GGD-based) (``values``)
        2. emissivity coefficients (JOREK GGD-based)  (``coefficients``)

    In the case of (1), one tries to load both core and edge (GGD-based) emissivity values from one
    IMAS query. If both are available, they are blended using a mask function.
    If the second IMAS query is provided, (``args2``, ``kwargs2``, etc.), it is used to load the
    missing emissivity values if one of them is not available in the first query.

    In the case of (2), one tries to load emissivity coefficients from the GGD structure and
    reconstructs the emissivity based on the Fourier-Bezier method, which is tied to the JOREK
    specifications.

    If ``source="auto"``, the function first tries to load emissivity values (1), and if they are
    not available, it falls back to emissivity coefficients (2). If neither is available, a
    ``RuntimeError`` is raised.

    For GGD-based emissivity, the grid interpolator handles cache lookup and persistence internally.

    Parameters
    ----------
    *args
        IMAS URI, netCDF path, or legacy positional arguments for `imas.DBEntry`.
        For a URI or path, read mode is selected automatically; do not pass ``"r"``.
    time
        Time slice to load from the IDS, by default 0.0.
    occurrence
        Occurrence of the radiation IDS, by default 0.
    args2
        URI, netCDF path, or legacy positional DBEntry arguments for the second emissivity.
        Read mode is selected automatically. If None, the second emissivity is not loaded,
        by default None.
    kwargs2
        Additional DBEntry options for the second emissivity. If None, no options are used,
        by default None.
    time2
        Time slice to load for the second emissivity. By default, uses the same time as the first
        emissivity.
    occurrence2
        Occurrence of the radiation IDS to load for the second emissivity, by default 0.
    process_index
        Radiation process identifier index (or indices) to load.
        By default, all available processes are summed together.
        Reference: https://imas-data-dictionary.readthedocs.io/en/latest/generated/identifier/radiation_identifier.html
        The emissivity value array is assumed to follow the same x-axis as the grid subset.
    grid_ggd
        Specific grid GGD structure alternative to the one in the IDS.
    grid_subset_id
        ID of the grid subset to use, by default 5 (= "cells") subset.
        Reference: https://imas-data-dictionary.readthedocs.io/en/latest/generated/identifier/ggd_subset_identifier.html
    equilibrium
        Alternative `~cherab.tools.equilibrium.efit.EFITEquilibrium` used to map core profiles.
        By default None: the equilibrium is read from the same IMAS query as the core profiles.
        Ignored if the core radiation is not available.
    psi_interpolator
        Alternative ``psi_norm(rho_tor_norm)`` interpolator.
        Used only if ``psi`` is missing in the core grid, by default None.
        Obtained from the ``equilibrium`` IDS in the same IMAS query as the core profiles.
    mask
        Mask function used for blending: ``(1 - mask) * f_gdd + mask * f_core``.
        By default, uses `~cherab.tools.equilibrium.efit.EFITEquilibrium`'s `inside_lcfs`.
    num_toroidal
        Number of toroidal subdivisions for 3D grid extension, by default 64.
        This is used only when the grid is loaded by `.load_unstruct_grid_2d_extended`.
    source
        Source for emissivity data: ``"auto"`` (tries values then coefficients), ``"values"``
        (blended emissivity from core profiles + (edge) GGD values), or ``"coefficients"``
        (reconstruct from Fourier-Bezier coefficients), by default ``"auto"``.
    step
        Step size for the radiation function interpolator, by default 0.01 m.
    parent
        Parent node in the Raysect scenegraph, by default None.
    time_threshold
        Maximum allowed time difference when loading from IDS, by default ``inf``.
    interpolator_cache
        Interpolator cache strategy, by default ``"memory"``.
        Each strategy is described in the `.InterpolatorCacheMode` type alias.
    interpolator_cache_dir
        Directory used when ``interpolator_cache="disk"``, by default None
        (uses the system cache directory, e.g., ``~/.cache/cherab/imas/interpolators``).
    **kwargs
        Additional `imas.DBEntry` options, such as ``dd_version`` or ``xml_path``.

    Returns
    -------
    `~raysect.primitive.csg.Subtract` or `~raysect.primitive.Cylinder`
        Cylindrical emitter primitive with
        `~cherab.tools.emitters.radiation_function.RadiationFunction` material.

    Raises
    ------
    ValueError
        If ``source`` is not one of ``"auto"``, ``"values"``, or ``"coefficients"``.
    RuntimeError
        If no emissivity data is available in either core or GGD radiation data.
    """
    if source not in {"auto", "values", "coefficients"}:
        raise ValueError(
            f"Invalid source '{source}'. Expected one of: 'auto', 'values', 'coefficients'."
        )

    if process_index is None:
        process_indices = None
    elif isinstance(process_index, int):
        process_indices = {process_index}
    else:
        process_indices = set(process_index)

    # Common variables
    ids = None
    ids2 = None
    entry_reference: str | None = None
    entry_reference2: str | None = None
    emitter = None
    grid = None
    primitive_name: str | None = None
    radius_outer: float = 0.0
    radius_inner: float = 0.0
    height: float = 0.0
    zmin: float = 0.0

    try:
        with _open_dbentry_for_reading(*args, **kwargs) as entry:
            ids = get_ids_time_slice(
                entry,
                "radiation",
                time=time,
                occurrence=occurrence,
                time_threshold=time_threshold,
            )
            entry_reference = get_entry_reference(entry)
    except RuntimeError as err:
        raise RuntimeError("Unable to load radiation IDS.") from err

    if args2 is not None and source != "coefficients":
        try:
            with _open_dbentry_for_reading(*args2, **(kwargs2 or {})) as entry:
                ids2 = get_ids_time_slice(
                    entry,
                    "radiation",
                    time=time2 or time,
                    occurrence=occurrence2,
                    time_threshold=time_threshold,
                )
                entry_reference2 = get_entry_reference(entry)
        except RuntimeError as err:
            raise RuntimeError("Unable to load second radiation IDS.") from err

    # ------------------------------
    # === Load emissivity values ===
    # ------------------------------
    # temporary variables
    eq_args = args
    eq_kwargs = kwargs
    eq_time = time
    eq_occurrence = occurrence
    rad_func = None
    rad_func_core = None
    rad_func_ggd = None
    ggd_args: tuple[Any, ...] | None = args
    ggd_kwargs: dict[str, Any] | None = kwargs
    ggd_ids = ids

    if source in {"auto", "values"}:
        # ------------------------------
        # === Load emissivity values ===
        # ------------------------------
        values_core, values_ggd = _load_emissivity_values(
            ids.process, process_indices, grid_subset_id
        )

        # Load emissivity values from the second IDS if available and needed
        if values_core is None and values_ggd is None:
            pass

        elif values_ggd is None or values_core is None:
            if ids2 is not None:
                values_core2, values_ggd2 = _load_emissivity_values(
                    ids2.process, process_indices, grid_subset_id
                )

                if values_core is not None and values_core2 is not None:
                    raise RuntimeError(
                        "Duplicate core emissivity values are available in both radiation IDSs."
                    )
                if values_ggd is not None and values_ggd2 is not None:
                    raise RuntimeError(
                        "Duplicate GGD emissivity values are available in both radiation IDSs."
                    )

                if values_core is None:
                    values_core = values_core2
                    eq_args = args2
                    eq_kwargs = kwargs2
                    eq_time = time2 or time
                    eq_occurrence = occurrence2
                if values_ggd is None:
                    values_ggd = values_ggd2
                    ggd_args = args2
                    ggd_kwargs = kwargs2
                    if ids2 is not None:
                        ggd_ids = ids2
                entry_reference = f"{entry_reference} + {entry_reference2}"

        if values_core is None and values_ggd is None and source == "values":
            raise RuntimeError(
                "No emissivity values are available in either core or GGD radiation data."
            )

        # ----------------------------------
        # === Create radiation functions ===
        # ----------------------------------
        if values_core is not None:
            # TODO: Should load grid data at the same time as emissivity?
            if len(ids.process) and len(ids.process[0].profiles_1d):
                grid_struct = ids.process[0].profiles_1d[0].grid
            elif ids2 is not None and len(ids2.process) and len(ids2.process[0].profiles_1d):
                grid_struct = ids2.process[0].profiles_1d[0].grid
            else:
                raise RuntimeError("No core grid is available in either radiation IDS.")

            grid_data = load_core_grid(grid_struct)
            rad_func_core, equilibrium = _create_rad_func_core(
                grid_data,
                values_core,
                equilibrium,
                psi_interpolator,
                eq_args,
                eq_kwargs,
                eq_time,
                eq_occurrence,
            )
            mask = mask or equilibrium.inside_lcfs
            radius_inner, radius_outer = equilibrium.r_range
            zmin, zmax = equilibrium.z_range
            height = zmax - zmin

        if values_ggd is not None:
            grid_ggd = grid_ggd or ggd_ids.grid_ggd[0]
            rad_func_ggd, extent = _create_rad_func_ggd(
                grid_ggd,
                values_ggd,
                grid_subset_id,
                ids_root=ggd_ids,
                db_args=ggd_args,
                db_kwargs=ggd_kwargs,
                interpolator_cache=interpolator_cache,
                interpolator_cache_dir=interpolator_cache_dir,
            )
            radius_outer = extent["rmax"]
            radius_inner = extent["rmin"]
            height = extent["zmax"] - extent["zmin"]
            zmin = extent["zmin"]

        if (
            rad_func_core is not None
            and rad_func_ggd is not None
            and isinstance(mask, Function2D | Function3D)
        ):
            rad_func = blend_core_edge_functions(
                rad_func_core,
                rad_func_ggd,
                mask,
                return3d=True,
            )
        elif rad_func_core is not None:
            rad_func = rad_func_core
        elif rad_func_ggd is not None:
            rad_func = rad_func_ggd
        else:
            pass

        if isinstance(rad_func, Function3D):
            emitter = RadiationFunction(rad_func, step=step)
            primitive_name = f"RadiationEmitter_{ids.time[0]}s, entry {entry_reference}"

    # ------------------------------------
    # === Load emissivity coefficients ===
    # ------------------------------------

    if emitter is None and source in {"auto", "coefficients"}:
        grid_ggd = _resolve_grid_ggd_reference(ids.grid_ggd[0], ids, args, kwargs)

        # Load GGD Grid
        grid = load_grid(
            grid_ggd,
            with_subsets=False,
            num_toroidal=num_toroidal,
        )

        if not isinstance(grid, UnstructGrid2DExtended):
            raise RuntimeError(
                "Coefficient-based emissivity reconstruction requires a 2D-extended grid."
            )
        coeff = None

        # Load emissivity coefficients
        for process in ids.process:
            # Validate process
            if (
                process_indices is not None
                and process.identifier.index.value not in process_indices
            ):
                continue

            _coeff = load_ggd_emissivity(
                process,
                1,  # NOTE: JOREK-specific: emissivity coefficients are always associated with nodes
                field="coefficients",
            ).sum()

            if _coeff is not None:
                if coeff is None:
                    coeff = _coeff
                else:
                    coeff += _coeff

        if coeff is None:
            if source == "coefficients":
                raise RuntimeError("No emissivity coefficients are available in radiation data.")
            raise RuntimeError(
                "Unable to load emissivity from radiation IDS:"
                " no values or coefficients are available."
            )

        constructor = FourierBezierConstructor(grid_ggd, coefficients=coeff)

        d_phi = 360.0 / grid.num_toroidal
        phis = np.arange(d_phi * 0.5, 360.0, d_phi, dtype=np.float64)

        emissivity = constructor.average_gaussian_faces_per_toroidal(phis).ravel()
        primitive_name = f"RadiationEmitter_{ids.time[0]}s, entry {entry_reference}"

        rad_func = grid.interpolator(
            emissivity,
            interpolator_cache=interpolator_cache,
            interpolator_cache_dir=interpolator_cache_dir,
        )
        # Create RadiationFunction material
        emitter = RadiationFunction(rad_func, step=step)

        # Determine primitive dimensions
        radius_outer = grid.mesh_extent["rmax"]
        radius_inner = grid.mesh_extent["rmin"]
        height = grid.mesh_extent["zmax"] - grid.mesh_extent["zmin"]
        zmin = grid.mesh_extent["zmin"]

    # -------------------------------
    # === Create Primitive object ===
    # -------------------------------
    if emitter is None:
        raise RuntimeError("Emitter material cannot be constructed.")

    if radius_inner > 0:
        primitive = Subtract(
            Cylinder(radius_outer, height), Cylinder(radius_inner, height), parent=parent
        )
    else:
        primitive = Cylinder(radius_outer, height, parent=parent)

    primitive.transform = translate(0, 0, zmin)
    primitive.material = emitter
    primitive.name = primitive_name or "RadiationEmitter"

    return primitive
