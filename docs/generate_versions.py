"""Generate the sphinx-immaterial version manifest for GitHub Pages."""

from __future__ import annotations

import argparse
import json
import re
from importlib.metadata import version as distribution_version
from pathlib import Path
from textwrap import dedent
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MANIFEST_URL = "https://cherab.github.io/imas/versions.json"


def write_redirect(output_dir: Path, latest: str) -> None:
    """Redirect ``/latest/`` to the latest release directory."""
    target = f"../{latest}/"
    latest_dir = output_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "index.html").write_text(
        dedent(
            f"""\
            <!doctype html>
            <html lang="en">
              <head>
                <meta charset="utf-8">
                <meta http-equiv="refresh" content="0; url={target}">
                <link rel="canonical" href="{target}">
                <title>Redirecting to the latest documentation</title>
              </head>
              <body>
                <p>Redirecting to the <a href="{target}">latest documentation</a>.</p>
              </body>
            </html>
            """
        ),
        encoding="utf-8",
    )


def fetch_versions(url: str) -> list[dict[str, object]]:
    """Fetch the current manifest, falling back to an empty list.

    Returns
    -------
    list of dict
        Valid manifest entries, or an empty list when the manifest is unavailable.
    """
    try:
        request = Request(url, headers={"Cache-Control": "no-cache"})
        with urlopen(request, timeout=10) as response:
            manifest = json.load(response)
    except (HTTPError, URLError, json.JSONDecodeError, TypeError) as error:
        print(f"Could not read {url}: {error}; starting with an empty manifest")
        return []
    if not isinstance(manifest, list):
        print(f"Could not read {url}: manifest root is not a list; starting empty")
        return []
    return [
        item for item in manifest if isinstance(item, dict) and isinstance(item.get("version"), str)
    ]


def version_key(version: str) -> tuple[tuple[int, ...], bool, str]:
    """Return a sortable key for a release version.

    Returns
    -------
    tuple
        Numeric version parts, final-release flag, and suffix.
    """
    match = re.fullmatch(r"(\d+(?:\.\d+)*)(.*)", version)
    if match is None:
        return ((), False, version)
    suffix = match.group(2)
    return (tuple(map(int, match.group(1).split("."))), not suffix, suffix)


def generate(kind: str, output: Path, source_url: str, release: str | None = None) -> bool:
    """Update the fetched manifest for a development or release build.

    Returns
    -------
    bool
        Whether a new manifest was written.
    """
    current = (
        "dev" if kind == "dev" else release or distribution_version("cherab-imas").split("+")[0]
    )
    existing = fetch_versions(source_url)
    deployed = {str(item["version"]) for item in existing}
    if kind == "dev" and current in deployed:
        output.unlink(missing_ok=True)
        print(f"{source_url} already contains {current!r}; skipping")
        return False

    deployed.add(current)
    releases = sorted(
        (version for version in deployed if version_key(version)[0]),
        key=version_key,
        reverse=True,
    )
    latest = releases[0] if releases else None
    versions = [*(["dev"] if "dev" in deployed else []), *releases]
    manifest = [
        {
            "version": version,
            "title": f"{version} (latest)" if version == latest else version,
            "aliases": ["latest"] if version == latest else [],
        }
        for version in versions
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if latest:
        write_redirect(output.parent, latest)
    print(f"Wrote {output.absolute()}")
    return True


def main() -> None:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["dev", "release"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-url", default=DEFAULT_MANIFEST_URL)
    parser.add_argument("--release")
    args = parser.parse_args()
    generate(args.kind, args.output, args.source_url, args.release)


if __name__ == "__main__":
    main()
