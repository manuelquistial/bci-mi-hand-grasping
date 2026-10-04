"""Playback configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlaybackConfig:
    """Options for a single playback run."""

    max_samples: int | None = None
    stream_name: str | None = None
    notify_finish: bool = True
    progress_every: int = 500
    # Give Lab Streaming Layer time to discover the outlet before/after data.
    advertise_seconds: float = 0.0
    hold_seconds: float = 0.0
