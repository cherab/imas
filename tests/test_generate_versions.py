import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "docs" / "generate_versions.py"


def run_generator(
    tmp_path: Path,
    kind: str,
    existing: list[dict[str, object]] | None,
    *extra_args: str,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    source = tmp_path / "published-versions.json"
    if existing is not None:
        source.write_text(json.dumps(existing), encoding="utf-8")
    output = tmp_path / "output" / "versions.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            kind,
            str(output),
            "--source-url",
            source.as_uri(),
            *extra_args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result, output


def test_dev_is_added_to_existing_manifest(tmp_path: Path) -> None:
    _, output = run_generator(
        tmp_path,
        "dev",
        [
            {"version": "0.3.0", "title": "0.3.0 (latest)", "aliases": ["latest"]},
            {"version": "0.2.0", "title": "0.2.0", "aliases": []},
        ],
    )

    assert json.loads(output.read_text()) == [
        {"version": "dev", "title": "dev", "aliases": []},
        {"version": "0.3.0", "title": "0.3.0 (latest)", "aliases": ["latest"]},
        {"version": "0.2.0", "title": "0.2.0", "aliases": []},
    ]


def test_existing_dev_manifest_is_not_regenerated(tmp_path: Path) -> None:
    result, output = run_generator(
        tmp_path,
        "dev",
        [{"version": "dev", "title": "dev", "aliases": []}],
    )

    assert not output.exists()
    assert "already contains 'dev'; skipping" in result.stdout


def test_release_updates_latest_version(tmp_path: Path) -> None:
    _, output = run_generator(
        tmp_path,
        "release",
        [
            {"version": "dev", "title": "dev", "aliases": []},
            {"version": "0.2.0", "title": "0.2.0 (latest)", "aliases": ["latest"]},
        ],
        "--release",
        "0.3.0",
    )

    assert json.loads(output.read_text()) == [
        {"version": "dev", "title": "dev", "aliases": []},
        {"version": "0.3.0", "title": "0.3.0 (latest)", "aliases": ["latest"]},
        {"version": "0.2.0", "title": "0.2.0", "aliases": []},
    ]
    redirect = output.with_name("latest") / "index.html"
    assert "url=../0.3.0/" in redirect.read_text()


def test_missing_manifest_initializes_dev(tmp_path: Path) -> None:
    result, output = run_generator(tmp_path, "dev", None)

    assert json.loads(output.read_text()) == [{"version": "dev", "title": "dev", "aliases": []}]
    assert "starting with an empty manifest" in result.stdout
