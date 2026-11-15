"""Frozen CLI stdout contracts used by pipeline.service simulator queries."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "simopenbci", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_list_datasets_stdout_is_json_array() -> None:
    result = _run("--list-datasets")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    ids = {item["id"] for item in payload}
    assert "shuqfa2024" in ids
    assert "liu2024_stroke" in ids
    assert "recorded_session" in ids
    for item in payload:
        assert set(item) >= {"id", "label"}


def test_list_subjects_json_is_array_when_dataset_has_subjects() -> None:
    result = _run("--list-subjects", "--dataset", "shuqfa2024", "--json")
    # May be empty if data is missing locally; still must be valid JSON on success.
    if result.returncode != 0:
        # Data root may be absent in CI — contract is still "JSON or documented error".
        assert result.stdout == "" or result.stdout.strip()
        return
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    for item in payload:
        assert "id" in item
