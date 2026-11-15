"""One-shot ``simopenbci`` dumps (samples / trial markers) as JSON."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _run_dump(
    project: Path,
    *,
    dataset: str,
    subject: str | None,
    session_dir: str | None,
    extra: list[str],
    failure: str,
) -> Any:
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "simopenbci",
        "--no-dialog",
        "--dataset",
        dataset,
        *extra,
    ]
    if session_dir:
        command.extend(["--file", session_dir])
    if subject:
        command.extend(["--subject", subject])
    completed = subprocess.run(
        command,
        cwd=project,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or failure).strip()
        raise RuntimeError(detail.splitlines()[-1])
    return json.loads(completed.stdout or "{}")


def dump_stream(
    project: Path,
    *,
    dataset: str,
    subject: str | None,
    session_dir: str | None,
    seconds: float,
) -> dict[str, Any]:
    data = _run_dump(
        project,
        dataset=dataset,
        subject=subject,
        session_dir=session_dir,
        extra=["--dump-seconds", str(seconds)],
        failure="dump-seconds failed",
    )
    if not isinstance(data, dict) or not data.get("samples"):
        raise RuntimeError("dump-seconds returned no samples")
    return data


def dump_markers(
    project: Path,
    *,
    dataset: str,
    subject: str | None,
    session_dir: str | None,
) -> dict[str, Any]:
    data = _run_dump(
        project,
        dataset=dataset,
        subject=subject,
        session_dir=session_dir,
        extra=["--dump-trials", "--trial-seconds", "4"],
        failure="dump-trials failed",
    )
    if not isinstance(data, dict):
        raise RuntimeError("dump-trials returned invalid JSON")
    return data
