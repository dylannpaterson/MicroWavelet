"""Set the date-based project version in pyproject.toml, __init__.py, and CITATION.cff."""

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

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:\.\d+)?(?:[a-zA-Z0-9._+-]*)?$")


def current_version() -> str:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def date_version(today: dt.date | None = None) -> str:
    today = today or dt.date.today()
    return f"{today.year % 100}.{today.month}.{today.day}"


def next_date_version(version: str, today: dt.date | None = None) -> str:
    base = date_version(today)
    if version == base:
        return f"{base}.1"

    serial_match = re.match(rf"^{re.escape(base)}\.(\d+)$", version)
    if serial_match:
        return f"{base}.{int(serial_match.group(1)) + 1}"

    return base


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
    parser.add_argument(
        "--version",
        help="Exact date-based version to set instead of using today's date.",
    )
    args = parser.parse_args()

    next_version = args.version or next_date_version(current_version())
    if not VERSION_RE.match(next_version):
        raise SystemExit(f"Invalid version: {next_version}")

    replace(PYPROJECT, r'^version = ".*"$', f'version = "{next_version}"')
    replace(INIT, r'^__version__ = ".*"$', f'__version__ = "{next_version}"')
    update_citation(next_version)
    print(next_version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
