"""Ask the simulator CLI for datasets, subjects, windows, and trials."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def query_simulator_datasets(simulator_project: Path) -> list[dict[str, str]]:
    """Ask the simulator which datasets it can play."""
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(simulator_project),
            "simopenbci",
            "--list-datasets",
        ],
        cwd=simulator_project,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "simulator exited").strip()
        last = detail.splitlines()[-1] if detail else "Could not list datasets"
        raise RuntimeError(last)
    data = json.loads(completed.stdout or "[]")
    if not isinstance(data, list):
        raise RuntimeError("dataset list must be a JSON array")
    datasets: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        datasets.append(
            {"id": str(item["id"]), "label": str(item.get("label") or item["id"])}
        )
    return datasets


def query_simulator_subjects(
    simulator_project: Path, dataset: str
) -> list[dict[str, str]]:
    """Ask the simulator which participants a dataset can play."""
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(simulator_project),
            "simopenbci",
            "--dataset",
            dataset,
            "--list-subjects",
            "--json",
        ],
        cwd=simulator_project,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "simulator exited").strip()
        last = detail.splitlines()[-1] if detail else "Could not list subjects"
        raise RuntimeError(last)
    data = json.loads(completed.stdout or "[]")
    if not isinstance(data, list):
        raise RuntimeError("subject list must be a JSON array")
    subjects: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        subjects.append(
            {"id": str(item["id"]), "label": str(item.get("label") or item["id"])}
        )
    return subjects


def query_simulator_window(
    simulator_project: Path,
    *,
    dataset: str,
    subject: str | None,
    session_dir: str | None,
    seconds: float,
) -> dict[str, Any]:
    """Read an opening window from the dataset. The simulator only reads the file."""
    if not dataset:
        raise ValueError("dataset is required")
    command = [
        "uv",
        "run",
        "--project",
        str(simulator_project),
        "simopenbci",
        "--no-dialog",
        "--dataset",
        dataset,
        "--dump-seconds",
        str(seconds),
    ]
    if session_dir:
        command.extend(["--file", session_dir])
    if subject:
        command.extend(["--subject", subject])
    completed = subprocess.run(
        command,
        cwd=simulator_project,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "simulator exited").strip()
        last = detail.splitlines()[-1] if detail else "Could not read the participant"
        raise RuntimeError(last)
    data = json.loads(completed.stdout or "{}")
    if not isinstance(data, dict) or not data.get("samples"):
        raise RuntimeError("participant window must include samples")
    return data


def query_simulator_trials(
    simulator_project: Path,
    *,
    dataset: str,
    subject: str | None,
    session_dir: str | None,
    trial_seconds: float,
) -> dict[str, Any]:
    """Read every marker window from the dataset. The caller keeps the imagery ones."""
    if not dataset:
        raise ValueError("dataset is required")
    command = [
        "uv",
        "run",
        "--project",
        str(simulator_project),
        "simopenbci",
        "--no-dialog",
        "--dataset",
        dataset,
        "--dump-trials",
        "--trial-seconds",
        str(trial_seconds),
    ]
    if session_dir:
        command.extend(["--file", session_dir])
    if subject:
        command.extend(["--subject", subject])
    completed = subprocess.run(
        command,
        cwd=simulator_project,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "simulator exited").strip()
        last = detail.splitlines()[-1] if detail else "Could not read the trials"
        raise RuntimeError(last)
    data = json.loads(completed.stdout or "{}")
    if not isinstance(data, dict) or not data.get("trials"):
        raise RuntimeError("participant trials must include trials")
    return data
