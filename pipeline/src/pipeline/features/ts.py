"""Downsampled time series of an aligned trial.

A sliding window adapts to the epoch length so each trial becomes a short
multichannel series instead of the raw sample vector.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_TARGET = 32


def downsample(trial: np.ndarray, points: int = _TARGET) -> np.ndarray:
    """Mean of a 50% overlapping window. ``trial`` is ``(channels, samples)``."""
    array = np.asarray(trial, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("A time-series trial needs at least two samples")
    count = int(array.shape[1])
    hop = max(1, count // max(1, int(points)))
    window = min(count, hop * 2)
    frames = [
        array[:, start : start + window].mean(axis=1)
        for start in range(0, count - window + 1, hop)
    ]
    return np.stack(frames, axis=1)


def time_series(trials: Sequence[np.ndarray], points: int = _TARGET) -> np.ndarray:
    """Mean downsampled trial, shaped ``(samples, channels)``."""
    if not trials:
        raise ValueError("Time series needs at least one trial")
    down = [downsample(trial, points) for trial in trials]
    length = min(int(item.shape[1]) for item in down)
    stacked = np.stack([item[:, :length] for item in down], axis=0)
    return stacked.mean(axis=0).T
