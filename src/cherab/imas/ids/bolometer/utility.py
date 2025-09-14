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
"""Module for bolometer utility functions."""

from enum import Enum

__all__ = ["GeometryType"]


class GeometryType(Enum):
    """Enum for geometry type.

    The geometry type of the bolometer foil or slit.

    Attributes
    ----------
    OUTLINE : int
        The geometry is defined by an outline.
    CIRCULAR : int
        The geometry is circular.
    RECTANGLE : int
        The geometry is rectangular.
    """

    OUTLINE = 1
    CIRCULAR = 2
    RECTANGLE = 3

    @classmethod
    def from_value(cls, value: int):
        """Get the geometry type from a value.

        Parameters
        ----------
        value : int
            The integer value to convert to a geometry type.
            If the value is not a valid geometry type, the default is `RECTANGLE`.
        """
        if value in cls._value2member_map_:
            return cls(value)
        return cls.RECTANGLE
