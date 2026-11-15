"""Shared paths and named network ports for the pipeline package."""

from __future__ import annotations

import os
from pathlib import Path

CONTROL_WS_PORT = 8090
OBSERVE_WS_PORT = 8091
SIMULATOR_LSL_NAME = "OpenBCI_EEG"


def default_recordings_root() -> Path:
    return Path(__file__).resolve().parents[2] / "recordings"


def default_simulator_project() -> Path:
    env = os.environ.get("SIMULATOR_PROJECT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "simulator"


def default_registry_root() -> Path:
    return default_recordings_root().parent / "online-models"
