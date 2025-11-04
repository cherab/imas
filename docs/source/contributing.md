# Contributing

We welcome contributions to `cherab-imas`! Whether you're fixing bugs, adding new features, or improving documentation, your help is appreciated.

## How to Contribute

1. **Fork the Repository**: Create your own fork of the repository on GitHub.
2. **Create a Branch**: Create a new branch for your feature or bugfix, like `feature/my-new-feature` or `bugfix/fix-issue-123`.
3. **Make Changes**: Make your changes in the new branch.
4. **Submit a Pull Request**: Once you're happy with your changes, submit a pull request to the main repository.

## Reporting Issues

If you encounter any issues or have suggestions for improvements, please open an issue on the GitHub repository. Provide as much detail as possible to help us understand and address the problem.

## Development Setup

### Prerequisites

Development tasks are managed using `pixi`. If you don't have `pixi` installed, please refer to the https://pixi.sh documentation for installation instructions.
Other tools like `git` can be installed globally via `pixi`:

```bash
pixi global install git
```

### Cloning the Repository

You can clone the repository and enter the directory with:

```bash
git clone https://github.com/cherab/imas.git
cd imas
```

### Running Development Tasks

You can run various development tasks using `pixi`. All available tasks can be listed with:

```bash
pixi task list
```

Here are some common tasks you might find useful:

::::{md-tab-set}
:::{md-tab-item} test
To run the tests, you can do so with:

```bash
pixi run test
```

Then, you can choose the environment you want to test against, e.g., `test` for Python 3.13 (`test-py313` is an alias for this).

Also, it is possible to choose specific test environments first:

```bash
pixi run -e test test
```

:::
:::{md-tab-item} docs
To build the documentation, you can use:

```bash
pixi run doc-build
```

The documentation will be built in the `docs/build/html` directory.
If you want to host the documentation locally, you can do so with:

```bash
pixi run doc-serve
```

The documentation will be served at http://localhost:8000.

:::
:::{md-tab-item} lint/format
To check the code style and formatting for all files, you can run:

```bash
pixi run lint
```

All linters and formatters registered in the [lefthook](https://lefthook.dev) configuration will be run.
If you want to run a specific linter or formatter, you can do so like:

```bash
pixi run ruff-check
```

When you commit your changes, the pre-commit hooks will automatically run once you install pre-commit hooks:

```bash
pixi run hooks
```

We recommend installing the pre-commit hooks to ensure code quality.

If you want to run them manually, then execute:

```bash
pixi run pre-commit
```

Note that you must stage your changes before running the pre-commit hooks.
:::

:::{md-tab-item} ipython
To run the IPython shell, you can do so with:

```bash
pixi run ipython
```

The IPython shell will be started with the package installed.
:::
::::

### Useful Links

The main dependency of `cherab-imas` is `imas-python`, handling the IMAS data structures. Here is the link to its documentation:

- [imas-python Documentation](https://imas-python.readthedocs.io/)

The IMAS data structure is standardized by the IMAS Data Dictionary (DD), where it defines all the available Interface Data Structures (IDSs).
`cherab-imas` interacts with some of these IDSs (e.g. `equilibrium`, `core_profiles`, etc.). Here is the link to the IDS reference:

- [IMAS IDS Reference](https://imas-data-dictionary.readthedocs.io/en/latest/reference_ids.html)
