"""Reserve a Zenodo DOI and update citation metadata.

The script is intentionally dependency-free so the release workflow can run it
before publishing. It skips cleanly when ZENODO_TOKEN is not configured.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
CITATION = ROOT / "CITATION.cff"
README = ROOT / "README.md"


def request(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
    separator = "&" if "?" in url else "?"
    url = f"{url}{separator}{urllib.parse.urlencode({'access_token': token})}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        raise SystemExit(f"Zenodo API request failed: {error.code} {body}") from error


def reserve_doi(token: str, deposition_id: str | None, sandbox: bool, version: str) -> str:
    base = "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"
    api = f"{base}/api/deposit/depositions"

    if deposition_id:
        created = request("POST", f"{api}/{deposition_id}/actions/newversion", token)
        latest_draft = created["links"]["latest_draft"]
        deposition = request("GET", latest_draft, token)
        deposition_url = f"{api}/{deposition['id']}"
    else:
        deposition = request("POST", api, token, {})
        deposition_url = f"{api}/{deposition['id']}"

    metadata = {
        "metadata": {
            "title": "MicroWavelet",
            "upload_type": "software",
            "description": (
                "CWT-based anomaly detector for multi-filter microlensing light curves."
            ),
            "creators": [{"name": "Paterson, Dylan"}],
            "access_right": "open",
            "license": "MIT",
            "version": version,
            "prereserve_doi": True,
        }
    }
    updated = request("PUT", deposition_url, token, metadata)
    prereserved = updated.get("metadata", {}).get("prereserve_doi", {})
    doi = prereserved.get("doi") or updated.get("doi")
    if not doi:
        raise SystemExit("Zenodo did not return a DOI for the draft deposition.")
    return doi


def replace(path: pathlib.Path, pattern: str, replacement: str) -> None:
    text = path.read_text()
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL | re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Could not update {path}")
    path.write_text(updated)


def update_files(doi: str) -> None:
    if re.search(r"^doi:", CITATION.read_text(), flags=re.MULTILINE):
        replace(CITATION, r'^doi: ".*"$', f'doi: "{doi}"')
    else:
        text = CITATION.read_text()
        text = text.replace("repository-code:", f'doi: "{doi}"\nrepository-code:', 1)
        CITATION.write_text(text)

    identifier_block = f'identifiers:\n  - type: doi\n    value: "{doi}"'
    replace(CITATION, r"identifiers:\s*(?:\[\]|(?:\n  .*)*)", identifier_block)

    badge = (
        "<!-- zenodo-doi-badge -->\n"
        f"[![DOI](https://zenodo.org/badge/DOI/{doi}.svg)](https://doi.org/{doi})\n"
        "<!-- /zenodo-doi-badge -->"
    )
    replace(README, r"<!-- zenodo-doi-badge -->.*?<!-- /zenodo-doi-badge -->", badge)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--sandbox", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ZENODO_TOKEN is not configured; skipping Zenodo DOI reservation.")
        return 0

    doi = reserve_doi(
        token=token,
        deposition_id=os.environ.get("ZENODO_DEPOSITION_ID"),
        sandbox=args.sandbox,
        version=args.version,
    )
    update_files(doi)
    print(doi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
