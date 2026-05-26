# Contributing

Contributions should come from a fork and a pull request.

Before opening a pull request, run the same checks used by CI:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check --fix .
python -m ruff format .
```

Pull requests should include tests for behavior changes. Keep changes focused, and note any scientific or numerical assumptions that affect the detector output.

Maintainers should release from `main` with the manual GitHub Actions release workflow. The workflow sets the date-based version, reserves a Zenodo DOI when credentials are configured, tags the release, creates GitHub release notes from commits since the previous tag, and triggers PyPI publishing through the trusted-publisher workflow.

Detailed maintainer instructions are in `RELEASE.md`.
