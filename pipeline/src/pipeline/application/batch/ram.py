"""Resident-memory helpers for the batch worker budget."""

from __future__ import annotations

import ctypes
import gc
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pipeline.classify import SubjectTrials

WORKER_BYTES = 512 * 1024 * 1024


def release_ram() -> None:
    """Return free allocator pages so the next worker budget sees the real resident size."""
    gc.collect()
    if not sys.platform.startswith("linux"):
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        return


def available_ram() -> int | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def fit_workers(
    requested: int,
    groups: Sequence[SubjectTrials],
    available: int | None,
) -> int:
    """How many fresh workers fit beside one shared copy of the trials.

    Each worker is a new interpreter. It attaches the shared block instead of
    copying this process, and keeps a private working set for one model.
    ``available`` is the free RAM in bytes, or ``None`` when it is unknown.
    """
    count = max(1, int(requested))
    if available is None:
        return count
    trial_bytes = sum(
        int(np.asarray(trial).nbytes) for group in groups for trial in group.trials
    )
    room = int(available * 0.5)
    if trial_bytes > room:
        return 1
    copies = room // WORKER_BYTES
    return max(1, min(count, copies))
