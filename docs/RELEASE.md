# Release and Workflow Guide

This repository is set up so releases happen in GitHub Actions, not from a laptop.

The maintainer should only need to:

1. Push normal code changes to `main`.
2. Make sure the automated tests pass.
3. Create the required service tokens once.
4. Run the manual `Release` workflow from GitHub.

## Plain-English Glossary

GitHub Actions:
Automated jobs that run on GitHub servers. This repo uses them for testing, linting, version bumps, GitHub releases, Zenodo DOI reservation, and PyPI publishing.

PyPI:
The Python Package Index. This is where `pip install MicroWavelet` downloads the package from.

GitHub Release:
A versioned release page on GitHub. It is attached to a git tag such as `v0.1.1` and contains release notes plus built package files.

Tag:
A named pointer to one commit. Python releases normally use tags like `v0.1.1`. In this repo, the release workflow creates and pushes the tag automatically.

Linting:
Static checks for code style, formatting, imports, and common mistakes. This repo uses Ruff. Ruff can also automatically fix many issues.

DOI:
A permanent citation identifier. Zenodo can reserve a DOI for a software release and this repo updates `README.md` and `CITATION.cff` when the DOI is reserved.

Secret:
A private value stored in GitHub repository settings, such as a PyPI token. The workflows can read secrets, but the values are not committed to the repository.

## Workflows in This Repo

### Test

File: `.github/workflows/test.yml`

Runs automatically on:

- Pushes to `main`
- Pull requests
- Manual runs from the GitHub Actions tab

What it does:

1. Installs the package with development tools.
2. Runs the test suite with `pytest`.
3. Runs Ruff lint validation.
4. Runs Ruff formatting validation.

Important detail:
Tests run before linting. This means a formatting issue will not hide functional test failures.

### Apply Ruff Fixes

File: `.github/workflows/lint-fix.yml`

Runs only when manually started from the GitHub Actions tab.

What it does:

1. Runs Ruff automatic fixes.
2. Runs Ruff formatting.
3. Commits and pushes any resulting code changes back to the branch.

Use this if CI fails only because of Ruff and you want GitHub to apply the mechanical fixes.

### Release

File: `.github/workflows/release.yml`

Runs only when manually started from the GitHub Actions tab.

What it does:

1. Runs tests.
2. Applies Ruff fixes and formatting.
3. Bumps the version in:
   - `pyproject.toml`
   - `microwavelet/__init__.py`
   - `CITATION.cff`
4. Reserves a Zenodo DOI if `ZENODO_TOKEN` is configured.
5. Builds the source distribution and wheel.
6. Commits the release metadata back to `main`.
7. Creates and pushes a git tag such as `v0.1.1`.
8. Creates a GitHub Release with notes generated from commits since the previous tag.
9. Publishes to PyPI if `PYPI_API_TOKEN` is configured.

If a token is missing, that part is skipped. Missing tokens should not break ordinary testing.

Release workflow gotcha:
The release workflow can apply Ruff fixes before committing the release metadata. Treat that as a safety net, not the normal linting process. The latest `Test` workflow should already be green before running a release.

## One-Time Setup

These setup steps must be done by the repository/package owner.

### 1. Create a PyPI Account

Go to <https://pypi.org/account/register/> and create an account.

Turn on two-factor authentication. PyPI strongly expects maintainers to use 2FA.

### 2. Create the PyPI API Token

This workflow currently expects a GitHub secret named `PYPI_API_TOKEN`.

For the very first publish:

1. Log in to PyPI.
2. Go to Account settings.
3. Find API tokens.
4. Create a new token.
5. If the `MicroWavelet` project does not exist on PyPI yet, create an account-wide token because there is no project to scope the token to yet.
6. Copy the token immediately. PyPI only shows it once.

After the first successful publish:

1. Go back to PyPI account settings.
2. Create a new token scoped only to the `MicroWavelet` project.
3. Replace the GitHub `PYPI_API_TOKEN` secret with the project-scoped token.
4. Revoke the old account-wide token.

Why replace it:
A project-scoped token is safer. If it leaks, it can only publish this package.

### 3. Create the Zenodo Token

Zenodo is optional. If this is not configured, releases still work; they just keep the DOI as pending.

To enable DOI reservation:

1. Create or log in to a Zenodo account at <https://zenodo.org/>.
2. Open the account menu.
3. Go to Applications.
4. Create a new personal access token.
5. Give it a name like `MicroWavelet GitHub release workflow`.
6. Select these scopes:
   - `deposit:write`
   - `deposit:actions`
7. Copy the token immediately.

Optional sandbox testing:
Zenodo has a separate sandbox at <https://sandbox.zenodo.org/>. Sandbox accounts and tokens are separate from production Zenodo. Sandbox DOIs are test DOIs and should not be used for real releases.

### 4. Add Tokens as GitHub Secrets

In the GitHub repository:

1. Open the repository on GitHub.
2. Click Settings.
3. In the left sidebar, open Secrets and variables.
4. Click Actions.
5. Click New repository secret.

Add these secrets:

`PYPI_API_TOKEN`
The PyPI token copied from PyPI. Required for PyPI publishing.

`ZENODO_TOKEN`
The Zenodo token copied from Zenodo. Optional; required for DOI reservation.

`ZENODO_DEPOSITION_ID`
Optional. Use this only if you already have an existing Zenodo deposition/concept you want new versions attached to.

Zenodo gotcha:
The current workflow reserves a DOI and updates `README.md` and `CITATION.cff`. It does not upload files to Zenodo or publish the Zenodo record. GitHub Releases and PyPI publishing are automated; final Zenodo record management may still need to be completed in Zenodo.

Important:
Do not paste tokens into code, issues, release notes, terminal output, Slack, email, or documentation. Put them only in GitHub Secrets.

## Normal Development Flow

For direct pushes to `main`:

1. Make the code change.
2. Run tests and Ruff locally if possible:

   ```bash
   python -m pip install -e ".[dev]"
   python -m pytest
   python -m ruff check .
   python -m ruff format --check .
   ```

3. Push to `main`.
4. Check the `Test` workflow in GitHub Actions.

If Ruff fails locally and you want Ruff to fix things:

```bash
python -m ruff check --fix .
python -m ruff format .
```

Then review the changed files, commit, and push.

If Ruff fails in GitHub and the fix is mechanical, run the manual `Apply Ruff Fixes` workflow.

## Releasing a New Version

Do not create the tag locally for normal releases. The release workflow creates the release commit and tag.

### Before Running the Release

1. Go to the repository on GitHub.
2. Open the Actions tab.
3. Click the `Test` workflow.
4. Confirm the latest run on `main` passed.
5. Confirm `PYPI_API_TOKEN` is configured if you want PyPI publishing.
6. Confirm `ZENODO_TOKEN` is configured if you want DOI reservation.

### Run the Release Workflow

1. Go to the repository on GitHub.
2. Open Actions.
3. Select `Release`.
4. Click Run workflow.
5. Choose the branch: `main`.
6. Choose the version bump:
   - `patch`: bug fixes, docs, small improvements, for example `0.1.0` to `0.1.1`
   - `minor`: new features, for example `0.1.0` to `0.2.0`
   - `major`: breaking changes, for example `1.2.3` to `2.0.0`
7. Leave `exact_version` blank unless you need a specific version.
8. Leave `zenodo_sandbox` unchecked for a real release.
9. Click Run workflow.

### What Happens Next

If the workflow succeeds, it will:

1. Push a release commit to `main`.
2. Push a tag like `v0.1.1`.
3. Create a GitHub Release.
4. Upload built package files to the GitHub Release.
5. Publish the package to PyPI if the PyPI token exists.
6. Update DOI metadata if the Zenodo token exists.

### After the Release

Check these pages:

1. GitHub Actions: confirm the `Release` workflow is green.
2. GitHub Releases: confirm a new release exists.
3. PyPI: confirm the new version appears at <https://pypi.org/project/MicroWavelet/>.
4. README: confirm the DOI badge changed if Zenodo was configured.
5. `CITATION.cff`: confirm the version and DOI metadata changed if Zenodo was configured.
6. Zenodo: if DOI reservation was configured, confirm whether the draft record needs any manual metadata, file upload, or publication steps in Zenodo.

Test install from PyPI after PyPI has updated:

```bash
python -m pip install --upgrade MicroWavelet
```

## Version Rules and Gotchas

PyPI does not allow replacing an already uploaded version.

If version `0.1.1` was published and you find a mistake, publish `0.1.2`. Do not try to overwrite `0.1.1`.

The publish step uses `skip-existing`, so rerunning a release will not overwrite files already on PyPI. That protects the workflow from failing just because an artifact already exists, but it also means a bad uploaded version cannot be corrected in place.

If a release is seriously broken, yank it on PyPI instead of deleting it:

1. Go to the PyPI project.
2. Open Manage project.
3. Open Releases.
4. Choose the broken version.
5. Yank the release and provide a reason.
6. Publish a fixed version.

## Tags

For normal releases, do not run git tag commands. The workflow handles tags.

The workflow creates an annotated tag automatically:

```bash
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

Those commands are shown only so you know what the workflow is doing. Running them manually before the workflow can cause the release workflow to fail or create confusing release notes.

## If Something Fails

### Tests Fail

The package was not released. Fix the test failure and rerun the workflow.

### Ruff Fails in the Test Workflow

Run locally:

```bash
python -m ruff check --fix .
python -m ruff format .
```

Commit and push the changes.

Or run the manual `Apply Ruff Fixes` workflow from GitHub Actions.

### PyPI Publish Is Skipped

This means `PYPI_API_TOKEN` is not set. Add it as a GitHub repository secret and rerun the release workflow, or run a new patch release.

### PyPI Says the Version Already Exists

That version number is already taken. Bump to a new patch version and release again.

### Zenodo DOI Reservation Is Skipped

This means `ZENODO_TOKEN` is not set. Add it as a GitHub repository secret if DOI reservation is needed.

### Zenodo Fails

Check that the token has `deposit:write` and `deposit:actions`.

If using `zenodo_sandbox`, make sure the token came from <https://sandbox.zenodo.org/>, not production Zenodo.

If using production Zenodo, make sure the token came from <https://zenodo.org/>, not the sandbox.

### The Workflow Created a Release Commit but Failed Later

Do not delete random tags or releases without checking what happened.

First inspect:

1. Whether the tag was pushed.
2. Whether the GitHub Release exists.
3. Whether the version exists on PyPI.

If PyPI already has the version, use a new version number for the next attempt.

## Security Notes

Never commit API tokens.

Use repository secrets for `PYPI_API_TOKEN`, `ZENODO_TOKEN`, and `ZENODO_DEPOSITION_ID`.

Do not print secrets in workflow logs.

Prefer project-scoped PyPI tokens after the first release.

Revoke and recreate any token that may have been copied to the wrong place.

## Useful Local Checks

These are optional but recommended before pushing:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

To let Ruff fix simple issues:

```bash
python -m ruff check --fix .
python -m ruff format .
```

To build the package locally only as a sanity check:

```bash
python -m build
python -m twine check dist/*
```

Local builds are not required for release. The GitHub `Release` workflow does the real build and publish.
