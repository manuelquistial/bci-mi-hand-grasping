"""Recent EEG window the game draws behind a running session."""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Sequence

import numpy as np

from pipeline.buffer import RingBuffer


def parse_marker_text(raw: str) -> tuple[str, str]:
    """``kind:code:label`` becomes ``(kind, label)``."""
    text = (raw or "").strip()
    parts = text.split(":")
    if len(parts) >= 3:
        kind = parts[0] or "event"
        label = ":".join(parts[2:]) or parts[1] or text
        return kind, label
    if len(parts) == 2:
        return parts[0] or "event", parts[1] or text
    return "event", text or "?"


class LiveSignal:
    """Thread-safe view of the last few seconds of a stream."""

    WINDOW_SECONDS = 5.0
    MAX_POINTS = 400

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffer: RingBuffer | None = None
        self._times: deque[float] = deque()
        self._markers: deque[tuple[float, str, str]] = deque(maxlen=500)
        self._channels: list[str] = []
        self._rate = 0.0

    def configure(
        self,
        n_channels: int,
        sample_rate_hz: float,
        channels: Sequence[str],
    ) -> None:
        rate = float(sample_rate_hz) if sample_rate_hz and sample_rate_hz > 0 else 250.0
        capacity = max(32, int(rate * self.WINDOW_SECONDS))
        labels = [str(name) for name in channels if str(name)][:n_channels]
        while len(labels) < n_channels:
            labels.append(f"Ch{len(labels) + 1}")
        with self._lock:
            self._buffer = RingBuffer(n_channels, capacity)
            self._times = deque(maxlen=capacity)
            self._markers = deque(maxlen=500)
            self._channels = labels
            self._rate = rate

    def append(
        self,
        samples: np.ndarray,
        timestamps: Sequence[float] | None = None,
    ) -> None:
        with self._lock:
            if self._buffer is None:
                return
            array = np.asarray(samples, dtype=np.float64)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            count = int(array.shape[0])
            self._buffer.append(array)
            self._times.extend(self._timestamps_for(count, timestamps))

    def add_marker(self, timestamp: float, kind: str, label: str) -> None:
        with self._lock:
            if self._buffer is None:
                return
            self._markers.append((float(timestamp), kind, label))

    def _timestamps_for(
        self,
        count: int,
        timestamps: Sequence[float] | None,
    ) -> list[float]:
        if timestamps is not None and len(timestamps) == count:
            return [float(value) for value in timestamps]
        step = 1.0 / self._rate if self._rate else 0.004
        start = (self._times[-1] + step) if self._times else 0.0
        return [start + index * step for index in range(count)]

    def close(self) -> None:
        with self._lock:
            self._buffer = None
            self._times = deque()
            self._markers = deque(maxlen=500)
            self._channels = []
            self._rate = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active = self._buffer is not None
            channels = list(self._channels)
            rate = self._rate
            data = None if self._buffer is None or self._buffer.size == 0 else self._buffer.latest()
            times = list(self._times)
            stored_markers = list(self._markers)
        if data is None:
            return {
                "active": active,
                "channels": channels,
                "sample_rate_hz": rate,
                "samples": [],
                "markers": [],
            }
        markers = _markers_in_view(stored_markers, times)
        n = data.shape[1]
        count = min(n, self.MAX_POINTS)
        if count < n:
            index = np.unique(np.linspace(0, n - 1, count).astype(int))
            data = data[:, index]
        samples = [[round(float(value), 3) for value in row] for row in data]
        return {
            "active": True,
            "channels": channels,
            "sample_rate_hz": rate,
            "samples": samples,
            "markers": markers,
        }


def _markers_in_view(
    markers: Sequence[tuple[float, str, str]],
    times: Sequence[float],
) -> list[dict[str, Any]]:
    """Marker onsets that fall inside the visible samples, as 0..1 across the plot."""
    if len(times) < 1:
        return []
    start = float(times[0])
    end = float(times[-1])
    span = end - start
    visible: list[dict[str, Any]] = []
    for timestamp, kind, label in markers:
        if timestamp < start or timestamp > end:
            continue
        position = 0.0 if span <= 0 else (float(timestamp) - start) / span
        visible.append(
            {
                "kind": kind,
                "label": label,
                "position": round(position, 4),
            }
        )
    return visible
