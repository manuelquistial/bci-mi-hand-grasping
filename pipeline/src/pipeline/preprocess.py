"""Offline EEG before epoching: notch, then bandpass.

The default 8–30 Hz band is the classifier band for left/right hand imagery.
It covers the usual sensorimotor rhythms (Gao 2025; Tobón-Henao 2022).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

PREVIEW_SECONDS = 8.0
DISPLAY_POINTS = 500


@dataclass(frozen=True)
class FilterSpec:
    notch_hz: float = 50.0
    notch_q: float = 30.0
    band_low_hz: float = 8.0
    band_high_hz: float = 30.0
    order: int = 4


def apply_notch_then_bandpass(
    samples: np.ndarray,
    sample_rate_hz: float,
    spec: FilterSpec | None = None,
) -> np.ndarray:
    """Filter continuous EEG. Notch first, then the bandpass. Shape is (samples, channels)."""
    spec = spec or FilterSpec()
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] < 16 or array.shape[1] < 1:
        raise ValueError("Need a continuous EEG segment with at least 16 samples")
    rate = float(sample_rate_hz)
    if rate <= 0:
        raise ValueError("Sample rate must be positive")
    nyquist = rate / 2.0
    if not (0 < spec.notch_hz < nyquist):
        raise ValueError("Notch must sit between 0 Hz and half the sample rate")
    if spec.notch_q <= 0:
        raise ValueError("Notch Q must be positive")
    if spec.order < 1:
        raise ValueError("Filter order must be at least 1")
    if not (0 < spec.band_low_hz < spec.band_high_hz < nyquist):
        raise ValueError(
            "Bandpass low must be below bandpass high, and high must be below half the sample rate"
        )
    notch_b, notch_a = iirnotch(spec.notch_hz, spec.notch_q, fs=rate)
    notched = filtfilt(notch_b, notch_a, array, axis=0)
    band_b, band_a = butter(
        spec.order,
        [spec.band_low_hz, spec.band_high_hz],
        btype="bandpass",
        fs=rate,
    )
    return filtfilt(band_b, band_a, notched, axis=0)


def decimate_time(samples: np.ndarray, max_points: int = DISPLAY_POINTS) -> np.ndarray:
    """Keep at most ``max_points`` rows, using the same step on every channel."""
    array = np.asarray(samples, dtype=np.float64)
    count = int(array.shape[0])
    if count <= max_points:
        return array
    step = int(np.ceil(count / max_points))
    return array[::step]


def traces(samples: np.ndarray) -> list[list[float]]:
    """One list per channel, rounded for a plot."""
    rounded = np.round(np.asarray(samples, dtype=np.float64), 3)
    return rounded.T.tolist()
