"""Module defining unit vector functions."""

from collections.abc import Callable

from raysect.core.math import Vector3D
from raysect.core.math.function.vector3d import Function1D as VectorFunction1D
from raysect.core.math.function.vector3d import Function2D as VectorFunction2D
from raysect.core.math.function.vector3d import Function3D as VectorFunction3D

__all__ = ["UnitVector1D", "UnitVector2D", "UnitVector3D"]

class UnitVector1D(VectorFunction1D):
    """Evaluates a unit vector for the given VectorFunction1D instance."""

    _vector: VectorFunction1D
    def __init__(
        self, vector: VectorFunction1D | tuple[float, float, float] | Callable[[float], Vector3D]
    ) -> None: ...

class UnitVector2D(VectorFunction2D):
    """Evaluates a unit vector for the given VectorFunction2D instance."""

    _vector: VectorFunction2D
    def __init__(
        self,
        vector: VectorFunction2D | tuple[float, float, float] | Callable[[float, float], Vector3D],
    ) -> None: ...

class UnitVector3D(VectorFunction3D):
    """Evaluates a unit vector for the given VectorFunction3D instance."""

    _vector: VectorFunction3D
    def __init__(
        self,
        vector: VectorFunction3D
        | tuple[float, float, float]
        | Callable[[float, float, float], Vector3D],
    ) -> None: ...
