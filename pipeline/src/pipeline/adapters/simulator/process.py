"""Start and configure the simulator subprocess."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any


def start_simulator_process(command: list[str]) -> subprocess.Popen[Any]:
    cwd = None
    if "--project" in command:
        cwd = command[command.index("--project") + 1]
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process.output_lines = []  # type: ignore[attr-defined]

    def _drain() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            lines: list[str] = process.output_lines  # type: ignore[attr-defined]
            lines.append(line.rstrip())
            del lines[:-30]

    threading.Thread(target=_drain, name="sim-output", daemon=True).start()
    return process


def build_simulator_command(
    *,
    simulator_project: Path,
    lsl_name: str,
    dataset: str | None = None,
    session_dir: str | None = None,
    file: str | None = None,
    subject: str | None = None,
    data_root: str | None = None,
    run: str | None = None,
    max_samples: int | None = None,
) -> list[str]:
    """argv for ``uv run simopenbci`` against a recorded session or another dataset."""
    if dataset == "recorded_session" and not (session_dir or file):
        raise ValueError("recorded_session requires session_dir")
    command = [
        "uv",
        "run",
        "--project",
        str(simulator_project),
        "simopenbci",
        "--no-dialog",
        "--lsl-name",
        lsl_name,
    ]
    if dataset:
        command.extend(["--dataset", dataset])
    target = session_dir or file
    if target:
        command.extend(["--file", target])
    if subject:
        command.extend(["--subject", subject])
    if data_root:
        command.extend(["--data-root", data_root])
    if run:
        command.extend(["--run", run])
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    return command
