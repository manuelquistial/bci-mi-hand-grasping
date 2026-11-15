"""Session file names, records, and time-to-sample mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

FORMAT = "pipeline-session-v1"
SESSION_NAME = "session.json"
SIGNAL_NAME = "signal.f32"
TIME_NAME = "time.f64"
MARKERS_NAME = "markers.jsonl"
EMG_NAME = "emg.f32"
EMG_TIME_NAME = "emg_time.f64"


@dataclass(frozen=True)
class SessionMarker:
    lsl_time: float
    kind: str
    code: str
    label: str


@dataclass(frozen=True)
class Recording:
    root: Path
    session_id: str
    sample_rate_hz: float
    units: str
    channel_labels: tuple[str, ...]
    stream_name: str
    samples: np.ndarray
    timestamps: np.ndarray
    markers: tuple[SessionMarker, ...]
    device: dict[str, object]
    emg_samples: np.ndarray
    emg_timestamps: np.ndarray
    emg_sample_rate_hz: float
    emg_channel_labels: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.samples.shape[1]) if self.samples.ndim == 2 else 0


def sample_index_for_time(timestamps: Sequence[float] | np.ndarray, lsl_time: float) -> int:
    """Map an LSL time onto the nearest sample at or after that instant.

    An empty recording maps to index 0. A time past the last sample maps to
    the last sample so playback can still emit the marker.
    """
    times = np.asarray(timestamps, dtype=np.float64)
    if times.size == 0:
        return 0
    index = int(np.searchsorted(times, float(lsl_time), side="left"))
    if index >= times.size:
        return int(times.size - 1)
    return index
