# Release and Workflow Guide

This repository is set up so releases happen in GitHub Actions, not from a laptop.

The maintainer should normally only need to:

1. Push normal code changes to `main`.
2. Make sure the automated tests pass.
3. Configure PyPI Trusted Publishing once.
4. Optionally configure a Zenodo token once.
5. Run the manual `Release` workflow from GitHub.

## Plain-English Glossary

GitHub Actions:
Automated jobs that run on GitHub servers. This repo uses them for testing, linting, version bumps, GitHub releases, Zenodo DOI reservation, and PyPI publishing.

PyPI:
The Python Package Index. This is where `pip install MicroWavelet` downloads the package from.

Trusted Publishing:
PyPI's tokenless publishing setup. Instead of storing a long-lived PyPI API token in GitHub, PyPI trusts one specific GitHub Actions workflow, repository, and environment. This repo uses `.github/workflows/publish.yml` with the GitHub environment named `pypi`.

GitHub Release:
A versioned release page on GitHub. It is attached to a git tag such as `v26.5.26` and contains release notes plus built package files.

Tag:
A named pointer to one commit. Python releases normally use tags like `v26.5.26`. In this repo, the release workflow creates and pushes the tag automatically.

Linting:
Static checks for code style, formatting, imports, and common mistakes. This repo uses Ruff. Ruff can also automatically fix many issues.

DOI:
A permanent citation identifier. Zenodo can reserve a DOI for a software release and this repo updates `README.md` and `CITATION.cff` when the DOI is reserved.

Secret:
A private value stored in GitHub repository settings. Zenodo uses a secret in this repo; PyPI publishing does not need one because it uses Trusted Publishing.

## Workflows in This Repo

### Test

File: `.github/workflows/test.yml`

Runs automatically on pushes to `main`, pull requests, and manual runs from the GitHub Actions tab.

What it does:

1. Installs the package with development tools.
2. Runs the test suite with `pytest`.
3. Runs Ruff lint validation.
4. Runs Ruff formatting validation.

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
3. Sets the date-based version in `pyproject.toml`, `microwavelet/__init__.py`, and `CITATION.cff`.
4. Reserves a Zenodo DOI if `ZENODO_TOKEN` is configured.
5. Builds the source distribution and wheel.
6. Commits the release metadata back to `main`.
7. Creates and pushes a git tag such as `v26.5.26`.
8. Creates a GitHub Release with notes generated from commits since the previous tag.
9. Triggers the `Publish to PyPI` workflow only if it created a new tag.

Release workflow gotcha:
The release workflow can apply Ruff fixes before committing the release metadata. Treat that as a safety net, not the normal linting process. The latest `Test` workflow should already be green before running a release.

### Publish to PyPI

File: `.github/workflows/publish.yml`

Runs when a `v*` tag is pushed, when manually started from the GitHub Actions tab, or when the `Release` workflow explicitly triggers it after creating a new tag.

What it does:

1. Checks out the released code.
2. Sets the package version from the tag, such as `v26.5.26` -> `26.5.26`.
3. Builds the source distribution and wheel.
4. Publishes to PyPI with Trusted Publishing.

This workflow must match the PyPI Trusted Publisher configuration:

- Repository owner: the GitHub owner or organization that owns the accepted upstream repo.
- Repository name: `MicroWavelet`
- Workflow filename: `publish.yml`
- Environment name: `pypi`

## One-Time Setup

These setup steps must be done by the repository/package owner.

### 1. Create or Confirm the PyPI Account

Go to <https://pypi.org/account/register/> and create an account if needed.

Turn on two-factor authentication. PyPI strongly expects maintainers to use 2FA.

### 2. Configure PyPI Trusted Publishing

This repo does not need a `PYPI_API_TOKEN`.

For an existing PyPI project:

1. Log in to PyPI.
2. Open the `MicroWavelet` project.
3. Go to Manage project.
4. Open Publishing.
5. Add a new GitHub trusted publisher.
6. Use these values:
   - Owner: the GitHub owner or organization for the accepted upstream repository.
   - Repository name: `MicroWavelet`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

For the first publish if the PyPI project does not exist yet:

1. Log in to PyPI.
2. Go to your account publishing settings.
3. Add a pending GitHub trusted publisher.
4. Use these values:
   - PyPI project name: `MicroWavelet`
   - Owner: the GitHub owner or organization for the accepted upstream repository.
   - Repository name: `MicroWavelet`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

Pending publisher gotcha:
The PyPI project name must match the package name in `pyproject.toml`. PyPI normalizes names, but do not intentionally change the spelling during setup.

### 3. Create the GitHub `pypi` Environment

The publish workflow uses this line:

```yaml
environment: pypi
```

In the GitHub repository:

1. Open Settings.
2. Open Environments.
3. Create a new environment named `pypi`.
4. Optional but recommended: require manual approval for this environment.

If approval is enabled, the PyPI publish job will pause until an approved maintainer allows it to continue.

### 4. Create the Zenodo Token

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

### 5. Add Zenodo Secrets in GitHub

In the GitHub repository:

1. Open Settings.
2. Open Secrets and variables.
3. Click Actions.
4. Click New repository secret.

Add these secrets only if Zenodo DOI reservation is wanted:

`ZENODO_TOKEN`
The Zenodo token copied from Zenodo.

`ZENODO_DEPOSITION_ID`
Optional. Use this only if you already have an existing Zenodo deposition/concept you want new versions attached to.

Zenodo gotcha:
The current workflow reserves a DOI and updates `README.md` and `CITATION.cff`. It does not upload files to Zenodo or publish the Zenodo record. GitHub Releases and PyPI publishing are automated; final Zenodo record management may still need to be completed in Zenodo.

Do not paste tokens into code, issues, release notes, terminal output, chat, email, or documentation. Put Zenodo tokens only in GitHub Secrets.

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
5. Confirm PyPI Trusted Publishing is configured for `publish.yml` and environment `pypi`.
6. Confirm `ZENODO_TOKEN` is configured if DOI reservation is wanted.

### Run the Release Workflow

1. Go to the repository on GitHub.
2. Open Actions.
3. Select `Release`.
4. Click Run workflow.
5. Choose the branch: `main`.
6. Leave `exact_version` blank for the normal date-based version.
7. Fill in `exact_version` only if you need a specific date version, such as `26.5.26` or `26.5.26.1`.
8. Leave `zenodo_sandbox` unchecked for a real release.
9. Click Run workflow.

Date version behavior:
The normal version for May 26, 2026 is `26.5.26`. If `26.5.26` is already the current version and another release is made the same day, the workflow uses `26.5.26.1`, then `26.5.26.2`, and so on.

Existing tag behavior:
If `exact_version` points to a tag that already exists, the workflow stops before creating a GitHub Release or publishing to PyPI. This avoids accidentally republishing an old tag.

### What Happens Next

If the workflow succeeds, it will:

1. Push a release commit to `main`.
2. Push a tag like `v26.5.26`.
3. Create a GitHub Release.
4. Upload built package files to the GitHub Release.
5. Trigger the `Publish to PyPI` workflow if the tag was newly created.
6. Publish the package to PyPI through Trusted Publishing.
7. Update DOI metadata if the Zenodo token exists.

### After the Release

Check these pages:

1. GitHub Actions: confirm the `Release` workflow is green.
2. GitHub Actions: confirm the `Publish to PyPI` workflow is green.
3. GitHub Releases: confirm a new release exists.
4. PyPI: confirm the new version appears at <https://pypi.org/project/MicroWavelet/>.
5. README: confirm the DOI badge changed if Zenodo was configured.
6. `CITATION.cff`: confirm the version and DOI metadata changed if Zenodo was configured.
7. Zenodo: if DOI reservation was configured, confirm whether the draft record needs any manual metadata, file upload, or publication steps in Zenodo.

Test install from PyPI after PyPI has updated:

```bash
python -m pip install --upgrade MicroWavelet
```

## Version Rules and Gotchas

PyPI does not allow replacing an already uploaded version.

If version `26.5.26` was published and you find a mistake, publish `26.5.26.1`. Do not try to overwrite `26.5.26`.

The publish step uses `skip-existing`, so rerunning a publish will not overwrite files already on PyPI. That protects the workflow from failing just because an artifact already exists, but it also means a bad uploaded version cannot be corrected in place.

If a release is seriously broken, yank it on PyPI instead of deleting it:

1. Go to the PyPI project.
2. Open Manage project.
3. Open Releases.
4. Choose the broken version.
5. Yank the release and provide a reason.
6. Publish a fixed version.

## Tags

For normal full releases, do not run git tag commands. The `Release` workflow handles version metadata, Zenodo DOI reservation, GitHub release notes, the tag, and PyPI publishing.

The workflow creates an annotated tag automatically, equivalent to:

```bash
git tag -a v26.5.26 -m "Release v26.5.26"
git push origin v26.5.26
```

Those commands are shown so you know what the workflow is doing. Running them manually before the workflow can cause the release workflow to skip the GitHub release and PyPI publish for that version.

For PyPI-only publishing from an existing commit, pushing a `v*` date-version tag triggers `Publish to PyPI`:

```bash
git tag -a v26.5.26 -m "Release v26.5.26"
git push origin v26.5.26
```

That will run only the `Publish to PyPI` workflow. It will not reserve a Zenodo DOI, update committed citation metadata, or create commit-based GitHub release notes.

The publish workflow sets the package version from the tag before building, so the version in `pyproject.toml` does not need to be edited manually for tag-only publishing.

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

### Publish to PyPI Is Waiting

If the GitHub `pypi` environment requires approval, approve the pending environment deployment in GitHub Actions.

### Publish to PyPI Did Not Start After the Button Release

The `Release` workflow explicitly starts `Publish to PyPI` after it creates a new tag. If the requested exact tag already existed, it intentionally skips PyPI publishing.

If someone pushes a `v*` tag manually, that tag push should also start `Publish to PyPI`.

### Publish to PyPI Fails with a Trusted Publisher Error

Check that the PyPI Trusted Publisher exactly matches:

- Owner
- Repository name: `MicroWavelet`
- Workflow name: `publish.yml`
- Environment name: `pypi`

Also check that `.github/workflows/publish.yml` has:

```yaml
permissions:
  id-token: write
```

### PyPI Says the Version Already Exists

That version number is already taken. Release again with the next date serial, such as `26.5.26.1`.

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

PyPI publishing should use Trusted Publishing, not a long-lived PyPI token.

Use repository secrets only for `ZENODO_TOKEN` and optional `ZENODO_DEPOSITION_ID`.

Do not print secrets in workflow logs.

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

Local builds are not required for release. The GitHub `Release` and `Publish to PyPI` workflows do the real build and publish.
