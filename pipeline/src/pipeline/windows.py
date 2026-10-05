"""Cut the five activity windows out of a live EEG stream."""

from __future__ import annotations

import threading
from typing import Any, Sequence

import numpy as np

from pipeline.epochs import activity_windows

_KEEP_SECONDS = 8.0


class ActivityWindower:
    """Keeps recent samples and slices each open activity into 2 s windows."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._times: list[float] = []
        self._samples: list[np.ndarray] = []
        self._activities: list[dict[str, Any]] = []

    def reset(self) -> None:
        with self._lock:
            self._times = []
            self._samples = []
            self._activities = []

    def begin(self, onset: float, label: str, code: str) -> dict[str, Any]:
        windows = [
            {"start": start, "end": end, "count": 0, "samples": None}
            for start, end in activity_windows(onset)
        ]
        activity = {
            "label": label,
            "code": code,
            "started_at": float(onset),
            "ended_at": None,
            "windows": windows,
        }
        with self._lock:
            self._activities.append(activity)
            self._fill()
        return _view(activity)

    def close_open(self, ended_at: float) -> dict[str, Any] | None:
        with self._lock:
            activity = next(
                (item for item in reversed(self._activities) if item["ended_at"] is None),
                None,
            )
            if activity is None:
                return None
            activity["ended_at"] = float(ended_at)
            return _view(activity)

    def push(self, samples: np.ndarray, timestamps: Sequence[float]) -> None:
        rows = np.asarray(samples, dtype=np.float64)
        if rows.size == 0:
            return
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        times = [float(value) for value in timestamps]
        if len(times) != rows.shape[0]:
            return
        with self._lock:
            self._samples.extend(rows)
            self._times.extend(times)
            self._fill()
            self._trim()

    def latest_view(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._activities:
                return None
            return _view(self._activities[-1])

    def _fill(self) -> None:
        if not self._times:
            return
        newest = self._times[-1]
        for activity in self._activities:
            for window in activity["windows"]:
                if window["samples"] is not None or newest < window["end"]:
                    continue
                chosen = [
                    row
                    for row, stamp in zip(self._samples, self._times)
                    if window["start"] <= stamp < window["end"]
                ]
                window["samples"] = (
                    np.vstack(chosen) if chosen else np.zeros((0, self._samples[0].shape[0]))
                )
                window["count"] = int(window["samples"].shape[0])

    def _trim(self) -> None:
        if not self._times:
            return
        cutoff = self._times[-1] - _KEEP_SECONDS
        pending = [
            window["start"]
            for activity in self._activities
            for window in activity["windows"]
            if window["samples"] is None
        ]
        if pending:
            cutoff = min(cutoff, min(pending))
        drop = 0
        while drop < len(self._times) and self._times[drop] < cutoff:
            drop += 1
        if drop:
            del self._times[:drop]
            del self._samples[:drop]


def _view(activity: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": activity["label"],
        "code": activity["code"],
        "started_at": activity["started_at"],
        "ended_at": activity["ended_at"],
        "windows": [
            {"start": window["start"], "end": window["end"], "samples": window["count"]}
            for window in activity["windows"]
        ],
    }
