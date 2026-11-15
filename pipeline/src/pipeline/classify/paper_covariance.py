"""Trial, window, and band covariances for the paper models."""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

import numpy as np
from scipy.signal import butter, sosfiltfilt

from pipeline.classify.paper_linalg import _spd_batch


_SOS: dict[tuple[float, float, float], np.ndarray] = {}


_SOS_LOCK = Lock()


def _sos(rate: float, low: float, high: float) -> np.ndarray:
    key = (float(rate), float(low), float(high))
    with _SOS_LOCK:
        cached = _SOS.get(key)
        if cached is None:
            cached = butter(4, [low, high], btype="bandpass", fs=rate, output="sos")
            _SOS[key] = cached
        return cached


def _window_covs(
    trials: Sequence[np.ndarray],
    rate: float,
    window: tuple[float, float],
    band: tuple[float, float],
) -> np.ndarray:
    stacked = np.stack([np.asarray(trial, dtype=np.float64) for trial in trials])
    filtered = sosfiltfilt(_sos(rate, band[0], band[1]), stacked, axis=2)
    start = int(round(window[0] * rate))
    stop = int(round(window[1] * rate))
    piece = filtered[:, :, start:stop]
    if piece.shape[2] < 16:
        raise ValueError("The time window is shorter than 16 samples")
    return _covariances_batch(piece)


def _covariances(trials: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.stack([np.asarray(trial, dtype=np.float64) for trial in trials])
    return _covariances_batch(stacked)


def _covariances_batch(trials: np.ndarray) -> np.ndarray:
    """Centered covariances for ``(trials, channels, samples)``."""
    centered = trials - trials.mean(axis=2, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", centered, centered, optimize=True)
    covs /= max(1, trials.shape[2] - 1)
    return _spd_batch(covs)
