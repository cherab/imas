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
"""Subpackage for creating plasma/equilibrium/magnetic field objects from IMAS."""

from .blend import load_plasma
from .core import load_core_plasma
from .edge import load_edge_plasma
from .equilibrium import load_equilibrium, load_magnetic_field

__all__ = [
    "load_plasma",
    "load_core_plasma",
    "load_edge_plasma",
    "load_equilibrium",
    "load_magnetic_field",
]
