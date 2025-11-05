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
# You can put same parameters defined in `imas.DBEntry`
plasma = load_plasma(
    "imas:hdf5?path=testdb",  # IMAS URI
    "r",  # read mode
    time=0.0,
    parent=world,
)
```

This script creates a `World` object and loads a plasma from an IMAS database located at `testdb`. The plasma is created close to time `0.0` and is added to the world as its parent.

You can find more examples and detailed documentation in the [Examples](examples) and [API Reference](api) sections.

## Citations

If you use `cherab-imas` in your research, please cite the <doi:10.5281/zenodo.1206142>.
