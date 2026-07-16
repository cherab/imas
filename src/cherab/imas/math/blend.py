# Copyright 2023 Euratom
# Copyright 2023 United Kingdom Atomic Energy Authority
# Copyright 2023 Centro de Investigaciones Energéticas, Medioambientales y Tecnologicas
#
# Licensed under the EUPL, Version 1.1 or - as soon they will be approved by the
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
"""Utilities for blending core and edge profile functions."""

from raysect.core.math.function.float import Blend2D, Blend3D, Function2D, Function3D
from raysect.core.math.function.vector3d import Blend2D as VectorBlend2D
from raysect.core.math.function.vector3d import Blend3D as VectorBlend3D
from raysect.core.math.function.vector3d import Function2D as VectorFunction2D
from raysect.core.math.function.vector3d import Function3D as VectorFunction3D

from cherab.core.math import AxisymmetricMapper, VectorAxisymmetricMapper

__all__ = ["blend_core_edge_functions"]


def blend_core_edge_functions(
    core_func: Function2D | Function3D | VectorFunction2D | VectorFunction3D | None,
    edge_func: Function2D | Function3D | VectorFunction2D | VectorFunction3D | None,
    mask: Function2D | Function3D,
    return3d: bool,
) -> Function2D | Function3D | VectorFunction2D | VectorFunction3D | None:
    """Blend core and edge functions using ``(1 - mask) * edge + mask * core``.

    Parameters
    ----------
    core_func
        A 2D or 3D core scalar/vector function.
    edge_func
        A 2D or 3D edge scalar/vector function.
    mask
        A 2D or 3D scalar mask function.
    return3d
        If True, map 2D outputs to 3D assuming axisymmetry.

    Returns
    -------
    Function2D | Function3D | VectorFunction2D | VectorFunction3D | None
        The blended function, or None if both inputs are None.

    Raises
    ------
    TypeError
        If function/mask types are unsupported.
    RuntimeError
        If scalar and vector functions are mixed.
    """
    if core_func is None and edge_func is None:
        return None

    if core_func is not None and not isinstance(
        core_func, Function2D | Function3D | VectorFunction2D | VectorFunction3D
    ):
        raise TypeError("The core_func must be a 2D or 3D function.")

    if edge_func is not None and not isinstance(
        edge_func, Function2D | Function3D | VectorFunction2D | VectorFunction3D
    ):
        raise TypeError("The edge_func must be a 2D or 3D function.")

    if not isinstance(mask, Function2D | Function3D):
        raise TypeError("The mask must be a 2D or 3D function.")

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

    raise RuntimeError("Cannot blend scalar and vector functions.")
