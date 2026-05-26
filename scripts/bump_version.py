"""Bump the project version in pyproject.toml, __init__.py, and CITATION.cff."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "microwavelet" / "__init__.py"
CITATION = ROOT / "CITATION.cff"

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[a-zA-Z0-9._+-]*)?$")


def current_version() -> str:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def bump(version: str, part: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if not match:
        raise SystemExit(f"Cannot automatically bump non-semver version: {version}")

    major, minor, patch = (int(value) for value in match.groups())
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise SystemExit(f"Unknown bump part: {part}")


def replace(path: pathlib.Path, pattern: str, replacement: str) -> None:
    text = path.read_text()
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Could not update version in {path}")
    path.write_text(updated)


def update_citation(version: str) -> None:
    today = dt.date.today().isoformat()
    replace(CITATION, r'^version: ".*"$', f'version: "{version}"')
    replace(CITATION, r'^date-released: ".*"$', f'date-released: "{today}"')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["major", "minor", "patch"], default="patch")
    parser.add_argument("--version", help="Exact version to set instead of bumping.")
    args = parser.parse_args()

    next_version = args.version or bump(current_version(), args.part)
    if not VERSION_RE.match(next_version):
        raise SystemExit(f"Invalid version: {next_version}")

    replace(PYPROJECT, r'^version = ".*"$', f'version = "{next_version}"')
    replace(INIT, r'^__version__ = ".*"$', f'__version__ = "{next_version}"')
    update_citation(next_version)
    print(next_version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
