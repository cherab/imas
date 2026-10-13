(intro/get-started)=

# 🚀 Get Started

This page gives a quick overview of how to get started with `cherab-imas`, including installation instructions and a simple example script.

## Installation

[![PyPI - Version][pypi-badge]][pypi]
[![Conda][conda-badge]][conda]

[pypi]: https://pypi.org/project/cherab-imas/
[pypi-badge]: https://img.shields.io/pypi/v/cherab-imas?label=PyPI&logo=pypi&logoColor=gold&style=flat-square
[conda]: https://prefix.dev/channels/conda-forge/packages/cherab-imas
[conda-badge]: https://img.shields.io/conda/vn/conda-forge/cherab-imas?logo=conda-forge&style=flat-square

`cherab-imas` can be installed by many package managers.
Explore the various methods below to install `cherab-imas` using your preferred package manager.

::::{md-tab-set}
:::{md-tab-item} pip

```bash
pip install cherab-imas
```

:::
:::{md-tab-item} conda

```bash
conda install -c conda-forge cherab-imas
```

:::
:::{md-tab-item} uv

```bash
uv add cherab-imas
```

:::
:::{md-tab-item} pixi

```bash
pixi add cherab-imas
```

:::
::::

## Writing your first script

Here is a simple example of how to use `cherab-imas` to create a plasma object from an IMAS database.

```python
from raysect.optical import World
from cherab.imas.plasma import load_plasma

# Create a world
world = World()

# Load plasma from IMAS database
plasma = load_plasma(
    "imas:hdf5?path=testdb",  # IMAS URI
    time=0.0,
    parent=world,
)
```

This script creates a `World` object and loads a plasma from an IMAS database located at `testdb`, reading the `core_profiles/equilibrium/edge_profiles` IDSs.
The plasma is created close to time `0.0` and is added to the world as its parent.

All `cherab-imas` loader APIs open URI and netCDF sources in IMAS read mode automatically. Pass
the source directly, without the `"r"` required by `imas.DBEntry` itself:

```python
# IMAS URI
plasma = load_plasma("imas:hdf5?path=testdb")

# A netCDF data entry works the same way
plasma = load_plasma("/path/to/data.nc")

# Other DBEntry options remain available as keyword arguments
plasma = load_plasma("/path/to/data.nc", dd_version="4.1.0")

# Legacy form is also supported
from imas.ids_defs import HDF5_BACKEND

plasma = load_plasma(HDF5_BACKEND, "testdb", 12345, 0)
```

Existing calls that explicitly pass `"r"` continue to work for compatibility, but are deprecated.
Modes that create or replace data, such as `"w"`, `"a"`, and `"x"`, are rejected by loader APIs.

You can find more examples and detailed documentation in the [Examples](examples) and [API Reference](api) sections.

## Citations

If you use `cherab-imas` in your research, please cite the <doi:10.5281/zenodo.1206141>.
