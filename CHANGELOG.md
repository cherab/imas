# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0]

### Added

- Add synthetic JINTRAC radiation values dataset support and related regression tests
- Extend radiation emitter loading to support dual emissivity sources with improved validation
- Add IDS path utility helpers and `get_entry_reference` for resolving entry references
- Add unit tests for IDS path handling, grid loading via path references, and plasma utility entry-reference workflows
- Add public `CellConnectivity`, `CellData`, and `VertexIndices` types and the `as_cell_data()`
  validation helper for GGD meshes
- Add validity-mask support for retaining the relationship between compacted 2D grids and their
  source GGD face data
- Add an optimized Cython implementation for calculating polygonal cell areas and area-weighted
  centroids, with OpenMP support for large meshes
- Add profile-shape inspection and plain-text/Rich summaries for species compositions
- Add regression tests for 2D grid geometry, subset validity, molecular species, and core/edge
  profile loading
- Add a 2D radiation-emitter example notebook

### Changed

- Select read mode automatically when CHERAB object loaders receive an IMAS URI or netCDF path;
  explicit `"r"` arguments remain temporarily supported but are deprecated
- Reject data-creating DBEntry modes in CHERAB object loader APIs
- Update API docstrings, examples, demos, and notebooks to use mode-free loader calls
- Improve radiation emitter loading checks for duplicate emissivity values and core-profile grid data
- Refactor `load_grid` to support explicit `entry` selection and improve referenced-grid source resolution
- Enhance plasma and emitter loading paths to use entry-reference aware grid resolution
- **Breaking:** Change `load_grid(..., with_subsets=True)` and
  `load_unstruct_grid_2d(..., with_subsets=True)` subset values from index arrays to
  `(indices, valid_data_mask)` tuples
- Extend `UnstructGrid2D` with source-data validity tracking so interpolation and plotting accept
  either compacted cell data or source-sized data
- Calculate cylindrical 2D cell volumes from area-weighted centroids over a full toroidal rotation
- Filter edge and blended-plasma profile arrays consistently when invalid GGD faces are omitted
- Classify neutral and charged molecular species consistently and retain molecular bundles in core
  and edge species compositions
- Promote `rich` from a test-only dependency to a runtime dependency
- Summarize non-empty species groups with compact labels, conventional ionic charge notation, and
  stable symbol-derived coloring in plain and `rich` tree output
- Update the edge-plasma example to use the ITER-SOLPS sample dataset

### Fixed

- Fix grid data loading checks for radiation core profiles
- Improve error handling in radiation emitter loading workflows
- Ignore missing, incomplete, and out-of-range GGD faces while preserving correct subset mappings
- Fix triangle indexing, cylindrical cell-volume calculation, and volume preservation in 2D grid
  subsets
- Fix the Cython compile and link arguments on macOS
- Preserve neutral and charged molecular species when loading core and edge compositions

## [0.5.0] - 2026-06-24

### Added

- Add `UnstructGrid3D` class and integrate it into the mesh loading workflow
- Add line-of-sight (LOS) handling utilities and related test scripts
- Add [binder](https://mybinder.org/) badge and configuration for interactive notebook demos
- Add notebook.link environment configuration for interactive notebook demos for future use in documentation.
- Add Fourier-Bezier reconstruction module and integrate it into math utilities
- Add JINTRAC/JOREK dataset fixtures and interpolator cache tests
- Add radiation data loading modules and emitter initialization
- Add JOREK-based 3D radiation emitter visualization notebook
- Add 3D grid visualization notebook for ITER JOREK data
- Add unit tests for `get_ids_time_slice` function with fallback handling

### Changed

- Expose `load2d` and `load3d` modules as public APIs
- Enhance mesh/grid utilities (`GGDGrid`, `UnstructGrid2D`, base mesh helpers) with improved plotting styles, utility functions, and type hints
- Update bolometer LOS geometry handling to use cylindrical coordinates for detector/aperture positions
- Improve documentation tooling and rendering configuration (MathJax/CSS and nblink environment setup)
- Enhance interpolator caching behavior in grid classes and add subset validation in `load_unstruct_grid_2d_extended`
- Update pixi channels and `doc-serve` task to support normal conda-forge settings and dynamic port arguments
- Refactor dataset handling to make `pooch` a required dependency and simplify cache directory handling
- Implement re-slicing of IDS via IMAS memory backend with fallback handling
- Update `environment.yml` to include additional dependencies for binder
- Update docstrings for interpolator cache parameters in grid classes

### Fixed

- Ensure bolometer dataset cache directories are created before use
- Fix total power calculation and exception handling in the emission notebook workflow
- Fix charge type conversion in `load_core_plasma` function
- Fix grid subset handling in plasma loading functions
- Ensure coefficients array is C-contiguous in `FourierBezierConstructor`

### Removed

- Remove `license-files` entry from `pyproject.toml`

## [0.4.1] - 2026-04-03

### Added

- Add builtin mock bolometer dataset generator `datasets.bolometer_moc`
- Add bolometer observer notebook (`docs/notebooks/observer/bolometer.ipynb`)
- Add observer tests and fixtures for loading and visualizing bolometer data
- Add `plotly` to test dependencies for bolometer visualization testing

### Changed

- Export builtin dataset helpers from `cherab.imas.datasets`
- Update bolometer visualization output to show ray-through ratio as a percentage
- Refine documentation notebooks and formatting related to plasma and bolometer workflows

### Fixed

- Fix dataset path examples in fetcher docstrings
- Fix notebook raw-cell metadata formatting
- Fix Python 3.10 typing compatibility in bolometer enum `from_value()` methods by replacing `Self`-based annotations, preventing version-specific type errors

## [0.4.0] - 2026-03-31

### Added

- Support Python 3.14
- Add `visualize` function for bolometer camera geometry visualization using `plotly`
- Add ion bundle splitting support in `load_plasma` for edge/core plasma loading
- Add `solve_coronal_equilibrium` utility and corresponding fractional abundance notebook
- Add dataset patching utility and update dataset registry/fetchers for JINTRAC workflows
- Add new plasma data models (`SpeciesData`, `ProfileData`, `SpeciesComposition`, `VelocityData`) in common IDs utilities

### Changed

- Change pixi environment to use Python 3.14 as default
- Add `plotly` dependency required by bolometer camera `visualize` support
- Refactor core/edge profile loaders to use dataclass-driven species/profile handling
- Refactor magnetic field and equilibrium loading paths and improve interpolator imports/types
- Improve documentation notebooks and Sphinx configuration for plasma workflows
- Update CI/quality tooling (`pyrefly`, pre-commit hooks, notebook stripping, pixi tasks)
- Temporarily skip `pyrefly` check in CI until external stubs are available

### Deprecated

- Deprecate `b_field_tor` in magnetic field loading API (warning added)

### Fixed

- Fix regression in magnetic field loading
- Fix path handling consistency in `iter_jintrac` / `fix_jintrac` dataset workflows

## [0.3.0] - 2026-01-30

### Added

- Implement Bolometry Observer functionality
- Add new notebook for creating emission profiles
- Add `*.pyi` files for cython sources
- Add `overload` decorator for better type hinting
- Add `ultraplot` package into `pixi` default environment

### Changed

- Add `pyrefly` package for type checking (still experimental)
- Refactor `load_equilibrium` and `load_magnetic_field_data` to use dataclasses
- Use `ultraplot` for plotting in examples and notebooks

### Fixed

- Fix `num_toroidal` parameter handling in `load_unstruct_grid_2d` function
- Fix typo in warning message in `load_profiles.py`

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
