# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-11-18

### Added

- Add some missing type hints.

### Changed

- Migrate docstring linting/formatting from `numpydoc` and `docformatter` to `ruff`
- Migrate `numpydoc` Python API reference to `napoleon` extension for Sphinx (to enjoy type hinting support)
- Update docstrings to be compatible with the `napoleon` style

### Fixed

- Fix values assignment in `load_equilibrium_data` function (convert to python types)
- Bug fix in `load_unstruct_grid_2d` function (incorrect `cells.append(cell)` line)

### Removed

- Remove `numpydoc` dependency (migrated to `ruff` for docstring linting as well)
- Remove `docformatter` dependency (migrated to `ruff` for docstring formatting as well)

## [0.2.0] - 2025-11-04

### Added

- 3D unstructured mesh support with tetrahedralization functionality
- Documentation sources and Jupyter notebooks for demos
- GitHub Actions workflow for documentation build and deployment
- Dataset utilities and fetchers using `pooch` and `rich` libraries
- Unit tests for loading plasma objects and data fetching functionality
- Test coverage reporting with pytest-cov
- VSCode settings for Python testing configuration
- Support for extended unstructured 2D grids with triangular mesh plotting
- OpenMP support for parallel compilation

### Changed

- 💥 **BREAKING**: Minimum Python version requirement increased to 3.10+
- Build system migrated from setuptools to hatchling with hatch-vcs
- Dependencies updated to use `imas-python[netcdf]` instead of separate packages
- Improved documentation structure and installation instructions
- Enhanced type checking configurations with mypy and basedpyright
- Refactored codebase to use src/ layout
- Updated docstrings formatting and improved code organization

### Fixed

- Grid subset name handling in edge plasma mesh demo
- String concatenation in error and warning messages
- Access to z_min and z_max attributes in get_ion_state function
- Species attribute references updated from `label` to `name` for DD4
- Toroidal magnetic field component renamed from `b_field_tor` to `b_field_phi`
- Bug in UnstructGrid2D pickling with numpy array setflags()
- Incorrect transposition in get_cylindrical_velocity_interpolators()
- Branch name references from 'main' to 'master' in CI workflows
- macOS x86_64 compilation support with appropriate LDFLAGS

### Removed

- Remove support for python 3.9 and earlier
- Obsolete build, clean, and test scripts replaced by pixi tasks
- setup.py file (build configuration moved to pyproject.toml)
- imas-data-dictionaries dependency
- Unused ruff target settings

## [0.1.1] - 2024-09-12

### Added

- Support Cython 3 stable release
- Support Cherab 1.5

### Fixed

- Fix numpy array setflags() bug in UnstructGrid2D pickling. (#4)
- Fix incorrect transposition in get_cylindrical_velocity_interpolators(). (#5)

## [0.1.0] - 2023-09-22

Initial release.
