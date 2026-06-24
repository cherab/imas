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

from pathlib import Path
from typing import Literal

import numpy as np
from raysect.core.math import translate
from raysect.core.math.function.float import Function2D
from raysect.core.scenegraph._nodebase import _NodeBase
from raysect.primitive import Cylinder, Subtract

from cherab.core.math import AxisymmetricMapper
from cherab.tools.emitters import RadiationFunction
from imas import DBEntry
from imas.ids_structure import IDSStructure

from ..ggd.base_mesh import InterpolatorCacheMode
from ..ids.common import get_ids_time_slice
from ..ids.common.ggd import load_grid
from ..ids.radiation import load_radiation_coefficients, load_radiation_emissivity
from ..math import FourierBezierConstructor
from ..plasma.utility import get_subset_name_index

__all__ = ["load_radiation_emitter"]


def load_radiation_emitter(
    *args,
    time: float = 0,
    occurrence: int = 0,
    process_index: int | None = None,
    ion_index: int = 0,
    emissivity_index: int = 0,
    grid_ggd: IDSStructure | None = None,
    grid_subset_id: int | str = 5,
    num_toroidal: int | None = None,
    phis: np.ndarray | None = None,
    source: Literal["auto", "values", "coefficients"] = "auto",
    step: float = 0.01,
    parent: _NodeBase | None = None,
    time_threshold: float = np.inf,
    interpolator_cache: InterpolatorCacheMode = "memory",
    interpolator_cache_dir: str | Path | None = None,
    **kwargs,
) -> Subtract | Cylinder:
    """Load radiation emissivity and create a single radiation emitter primitive.

    The grid interpolator handles cache lookup and persistence internally.

    Parameters
    ----------
    *args
        Positional arguments passed to `imas.DBEntry`.
    time
        Time slice to load from the IDS, by default 0.
    occurrence
        Occurrence of the radiation IDS to load, by default 0.
    process_index
        Index of the radiation process to load, by default None (loads the first process).
    ion_index
        Index of the ion species to load, by default 0.
    emissivity_index
        Index of the emissivity data to load, by default 0.
    grid_ggd
        Alternative grid GGD structure to use if the radiation IDS grid is empty, by default None.
    grid_subset_id
        ID or name of the grid subset to use, by default 5 (``"Cells"``).
    num_toroidal
        Number of toroidal subdivisions for 3D grid extension, by default None.
        This is used only when the grid is loaded by `.load_unstruct_grid_2d_extended`.
    phis
        Array of toroidal angles in degrees for emissivity reconstruction, by default None.
        This is used only when the grid is loaded by `.load_unstruct_grid_2d_extended`.
    source
        Source for emissivity data: ``"auto"`` (tries values then coefficients), ``"values"``
        (emissivity values), or ``"coefficients"`` (reconstruct from Fourier-Bezier coefficients),
        by default ``"auto"``.
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
        Additional keyword arguments passed to `imas.DBEntry`.

    Returns
    -------
    `~raysect.primitive.csg.Subtract` or `~raysect.primitive.Cylinder`
        Cylindrical emitter primitive with
        `~cherab.tools.emitters.radiation_function.RadiationFunction` material.

    Raises
    ------
    RuntimeError
        If the radiation IDS or its emissivity data cannot be loaded.
    """
    with DBEntry(*args, **kwargs) as entry:
        radiation_ids = get_ids_time_slice(
            entry,
            "radiation",
            time=time,
            occurrence=occurrence,
            time_threshold=time_threshold,
        )

    if not len(radiation_ids.grid_ggd) and grid_ggd is None:
        raise RuntimeError(
            "The 'grid_ggd' AOS of the radiation IDS is empty"
            " and an alternative grid_ggd structure is not provided."
        )

    grid_ggd_struct = grid_ggd or radiation_ids.grid_ggd[0]

    try:
        grid, subsets, subset_id = load_grid(
            grid_ggd_struct,
            with_subsets=True,
            num_toroidal=num_toroidal,
        )
        try:
            grid_subset_name, grid_subset_index = get_subset_name_index(subset_id, grid_subset_id)
            if not np.array_equal(subsets[grid_subset_name], np.arange(grid.num_cell, dtype=int)):
                grid = grid.subset(subsets[grid_subset_name], name=grid_subset_name)
            subset_enabled = True
        except ValueError:
            subset_enabled = False
            grid_subset_index = None
    except NotImplementedError:
        subset_enabled = False
        grid = load_grid(grid_ggd_struct, with_subsets=False, num_toroidal=num_toroidal)
        grid_subset_index = None

    emissivity = None
    values_error: Exception | None = None

    if source in {"auto", "values"}:
        try:
            if grid_subset_index is None and subset_enabled:
                raise RuntimeError("Unable to determine grid subset index for emissivity.values.")
            emissivity = load_radiation_emissivity(
                radiation_ids,
                process_index=process_index,
                grid_subset_index=5 if grid_subset_index is None else grid_subset_index,
            )
        except Exception as err:
            values_error = err
            if source == "values":
                raise

    if emissivity is None and source in {"auto", "coefficients"}:
        coeff = load_radiation_coefficients(
            radiation_ids,
            process_index=0 if process_index is None else process_index,
            ion_index=ion_index,
            emissivity_index=emissivity_index,
            grid_subset_index=grid_subset_index,
        )

        constructor = FourierBezierConstructor(grid_ggd_struct, coefficients=coeff)

        if phis is None:
            if not hasattr(grid, "num_toroidal"):
                raise RuntimeError(
                    "Coefficient-based emissivity reconstruction requires a 2D-extended grid with a num_toroidal attribute."
                )
            d_phi = 360.0 / grid.num_toroidal
            phis_array = np.arange(d_phi * 0.5, 360.0, d_phi, dtype=np.float64)
        else:
            phis_array = np.asarray(phis, dtype=np.float64)

        emissivity = constructor.average_gaussian_faces_per_toroidal(phis_array).ravel()

    if emissivity is None:
        if values_error is not None:
            raise RuntimeError(
                "Unable to load emissivity from radiation IDS using either values or coefficients."
            ) from values_error
        raise RuntimeError("Unable to load emissivity from radiation IDS.")

    rad_func = grid.interpolator(
        emissivity,
        interpolator_cache=interpolator_cache,
        interpolator_cache_dir=interpolator_cache_dir,
    )

    if isinstance(rad_func, Function2D):
        rad_func = AxisymmetricMapper(rad_func)

    emitter = RadiationFunction(rad_func, step=step)

    radius_outer = grid.mesh_extent["rmax"]
    radius_inner = grid.mesh_extent["rmin"]
    height = grid.mesh_extent["zmax"] - grid.mesh_extent["zmin"]
    zmin = grid.mesh_extent["zmin"]

    if radius_inner > 0:
        primitive = Subtract(
            Cylinder(radius_outer, height), Cylinder(radius_inner, height), parent=parent
        )
    else:
        primitive = Cylinder(radius_outer, height, parent=parent)

    primitive.transform = translate(0, 0, zmin)
    primitive.material = emitter
    primitive.name = f"RadiationEmitter_{radiation_ids.time[0]}s, uri {entry.uri}"

    return primitive
