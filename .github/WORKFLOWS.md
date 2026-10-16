# Workflow filtering

`.github/filters.yml` defines the inputs for the `tests`, `wheels` and `docs` filters used by `dorny/paths-filter`.
Update these rules when adding build inputs, test helpers or documentation dependencies.
Rules apply to added, modified, deleted and renamed files.

| Changed files                                                 | Tests / coverage | Wheels | Docs |
| ------------------------------------------------------------- | :--------------: | :----: | :--: |
| Package source, metadata or dependency configuration          |       Yes        |  Yes   | Yes  |
| `tests/**`                                                    |       Yes        |  Yes   |  No  |
| `docs/generate_versions.py`                                   |       Yes        |  Yes   | Yes  |
| Other `docs/**` files                                         |        No        |   No   | Yes  |
| `demos/**`                                                    |        No        |  Yes   |  No  |
| CI workflow                                                   |       Yes        |   No   |  No  |
| Packaging workflow                                            |        No        |  Yes   |  No  |
| Documentation workflows or notebook environment configuration |        No        |   No   | Yes  |
| Shared filter rules                                           |       Yes        |  Yes   | Yes  |

`README.md` is package metadata and is included in the documentation, so changing it selects all three filters.
The version manifest generator selects tests and wheels because both test suites execute `tests/test_generate_versions.py`.
Lint and source distribution builds run for every PR, including documentation-only changes.

PRs are compared with their base branch.
Pushes to `master` are compared with the branch tip before the push, including all commits in the push.
The CI tests and development documentation use the corresponding filters on these pushes.
Manual documentation builds and manual or release packaging builds bypass filtering.
Release creation and versioned documentation always run for their configured tag or release events.

Documentation previews are built for relevant PR changes, including fork PRs.
Deployment remains restricted to PRs from this repository.
Closing a PR always runs preview cleanup for same-repository PRs, without requiring change detection or a documentation build.

## Required checks

Use these aggregate jobs as required checks in GitHub branch protection or rulesets:

- `CI status`
- `Packaging status`
- `Documentation preview status`

Each aggregate runs even when a dependency fails or is skipped.
It accepts only successful required work or an intentional skip based on a successful filter result.
Change detection failures therefore fail the aggregate instead of being accepted as unnecessary work.
For documentation previews, fork deployment skips and PR-close cleanup are handled separately.
Adding these jobs does not update repository branch protection or ruleset settings automatically.
