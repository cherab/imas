# CHERAB-IMAS

<!-- BEGIN-HEADER -->

|         |                                                                                                                     |
| ------- | ------------------------------------------------------------------------------------------------------------------- |
| CI/CD   | [![CI][ci-badge]][ci] [![PyPI Publish][PyPI-publish-badge]][PyPi-publish] [![codecov][codecov-badge]][codecov]      |
| Docs    | [![Documentation Status][docs-badge]][docs]                                                                         |
| Package | [![PyPI - Version][pypi-badge]][pypi] [![Conda][conda-badge]][conda] [![PyPI - Python Version][python-badge]][pypi] |
| Meta    | [![License - EUPL-1.1][license-badge]][license] [![Pixi Badge][pixi-badge]][pixi-url]                               |

[ci]: https://github.com/cherab/imas/actions/workflows/ci.yaml
[ci-badge]: https://img.shields.io/github/actions/workflow/status/cherab/imas/ci.yaml?style=flat-square&logo=GitHub&label=CI
[codecov]: https://codecov.io/github/cherab/imas
[codecov-badge]: https://img.shields.io/codecov/c/github/cherab/imas?token=05LZGWUUXA&style=flat-square&logo=codecov
[conda]: https://prefix.dev/channels/conda-forge/packages/cherab-imas
[conda-badge]: https://img.shields.io/conda/vn/conda-forge/cherab-imas?logo=conda-forge&style=flat-square
[docs]: https://cherab-imas.readthedocs.io/en/latest/?badge=latest
[docs-badge]: https://readthedocs.org/projects/cherab-imas/badge/?version=latest&style=flat-square
[license]: https://opensource.org/licenses/EUPL-1.1
[license-badge]: https://img.shields.io/badge/license-EUPL_1.1%20-blue?style=flat-square
[pixi-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square
[pixi-url]: https://pixi.sh
[pypi]: https://pypi.org/project/cherab-imas/
[pypi-badge]: https://img.shields.io/pypi/v/cherab-imas?label=PyPI&logo=pypi&logoColor=gold&style=flat-square
[pypi-publish]: https://github.com/cherab/imas/actions/workflows/deploy-pypi.yml
[pypi-publish-badge]: https://img.shields.io/github/actions/workflow/status/cherab/imas/deploy-pypi.yml?style=flat-square&label=PyPI%20Publish&logo=github
[python-badge]: https://img.shields.io/pypi/pyversions/cherab-imas?logo=Python&logoColor=gold&style=flat-square

---

`cherab` add-on module for IMAS (Integrated Modelling & Analysis Suite).

This module enables the creation of `cherab`'s functional objects (e.g. plasma, observers, meshes, etc.) from IMAS IDS (Interface Data Structures).

<!-- END-HEADER -->

## 🔧 Get started for Developers

### Pre-requisites

- [`pixi`](pixi-url), a tool for project and package management.

If you don't have `git` installed, you can install it through `pixi` global installation:

```bash
pixi global install git
```

### Download and Run tasks

You can clone the repository and enter the directory with:

```bash
git clone https://github.com/cherab/imas
cd imas
```

Then, you can run tasks with `pixi` like:

```bash
pixi run <task>
```

For example, to run the tests:

```bash
pixi run test
```

Any other command can be seen with:

```bash
pixi task list
```

## 🌐 Installation (future release)

You can install the package from PyPI:

```bash
pip install cherab-imas
```

Or from Conda:

```bash
mamba install -c conda-forge cherab-imas
```

## 📝 Documentation

The documentation will be available [here][docs].

## 📄 License

This project is licensed under the terms of the [EUPL-1.1][license].
