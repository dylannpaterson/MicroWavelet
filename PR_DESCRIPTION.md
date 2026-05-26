# Make MicroWavelet PyPI-installable and add release automation

## Summary

This PR prepares MicroWavelet for packaging and automated releases. It adds complete PyPI metadata, an MIT license, citation metadata, contributor/release documentation, Ruff linting, CI test coverage across Python versions and runner architectures, and manual GitHub Actions workflows for Ruff fixes and releases.

## Changes

- Add PyPI package metadata to `pyproject.toml`, including README, license, authors, classifiers, project URLs, dev dependencies, pytest config, and Ruff config.
- Add `LICENSE`, `CITATION.cff`, `docs/CONTRIBUTING.md`, and `docs/RELEASE.md`.
- Update `README.md` with PyPI install instructions, development checks, citation guidance, and release documentation pointers.
- Add GitHub Actions workflows:
  - `Test`: runs pytest first, then Ruff lint and format checks.
  - `Apply Ruff Fixes`: manually applies and commits Ruff automatic fixes.
  - `Release`: manually sets date-based versions, optionally reserves a Zenodo DOI, builds distributions, creates a GitHub release with commit-based notes, and triggers trusted PyPI publishing only when it creates a new tag.
  - `Publish to PyPI`: publishes on pushed `v*` tags, sets the package version from the tag before building, and preserves the upstream trusted-publisher flow for tokenless PyPI publishing.
- Add release helper scripts:
  - `scripts/bump_version.py`
  - `scripts/reserve_zenodo_doi.py`
- Add Dependabot configuration for GitHub Actions and pip updates.
- Apply Ruff formatting and lint fixes across the existing codebase.

## Release Behavior

The release workflow is manual and intended to run from GitHub Actions on `main`. It creates the release commit and tag automatically for normal full releases. Pushing a `v*` tag directly is also supported as a PyPI-only publish path.

Missing external credentials are handled gracefully:

- Missing `ZENODO_TOKEN`: skips Zenodo DOI reservation.
- Existing requested tag: skips GitHub release creation and PyPI publishing.
- PyPI publishing uses Trusted Publishing via `.github/workflows/publish.yml` and the GitHub `pypi` environment; no PyPI API token is required.
- Existing PyPI artifact: skipped via `skip-existing`.

## Maintainer Notes

`docs/RELEASE.md` explains the release process for maintainers who are new to GitHub Releases, PyPI, Zenodo, GitHub Secrets, Trusted Publishing, and linting. It includes setup instructions, workflow usage, local Ruff commands, and common failure cases.

## Verification

- `python3 -m pytest -q`
- `python3 -m ruff check .`
- `python3 -m ruff format --check .`
- `python3 -m build`
- `python3 -m twine check dist/*`
