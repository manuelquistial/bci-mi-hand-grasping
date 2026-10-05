"""EMG conditioning before any later feature step.

Order: bandpass 10–250 Hz, notch at line frequency, full-wave rectification,
then a short-window envelope. The default envelope is the moving RMS. A moving
mean of the rectified signal is the linear envelope. The file on disk stays raw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


@dataclass(frozen=True)
class EmgSpec:
    band_low_hz: float = 10.0
    band_high_hz: float = 250.0
    notch_hz: float = 50.0
    notch_q: float = 30.0
    order: int = 4
    window_seconds: float = 0.1
    envelope: str = "rms"


def apply_emg(
    samples: np.ndarray,
    sample_rate_hz: float,
    spec: EmgSpec | None = None,
) -> np.ndarray:
    """Condition continuous EMG and return the envelope. Shape is (samples, channels)."""
    spec = spec or EmgSpec()
    conditioned = condition_emg(samples, sample_rate_hz, spec)
    return emg_envelope(conditioned, sample_rate_hz, spec)


def condition_emg(
    samples: np.ndarray,
    sample_rate_hz: float,
    spec: EmgSpec | None = None,
) -> np.ndarray:
    """Bandpass, then the line notch. The result is still bipolar."""
    spec = spec or EmgSpec()
    array = _as_segment(samples)
    rate = _rate(sample_rate_hz)
    _check_spec(spec, rate)
    band_b, band_a = butter(
        spec.order,
        [spec.band_low_hz, spec.band_high_hz],
        btype="bandpass",
        fs=rate,
    )
    banded = filtfilt(band_b, band_a, array, axis=0)
    notch_b, notch_a = iirnotch(spec.notch_hz, spec.notch_q, fs=rate)
    return filtfilt(notch_b, notch_a, banded, axis=0)


def emg_envelope(
    samples: np.ndarray,
    sample_rate_hz: float,
    spec: EmgSpec | None = None,
) -> np.ndarray:
    """Full-wave rectification, then RMS or a moving mean over a short window."""
    spec = spec or EmgSpec()
    array = _as_segment(samples)
    rate = _rate(sample_rate_hz)
    rectified = np.abs(array)
    window = _window_samples(spec.window_seconds, rate, array.shape[0])
    if spec.envelope == "rms":
        return _moving(rectified**2, window, root=True)
    if spec.envelope == "mean":
        return _moving(rectified, window, root=False)
    raise ValueError("Envelope must be 'rms' or 'mean'")


def _as_segment(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] < 64 or array.shape[1] < 1:
        raise ValueError("Need a continuous EMG segment with at least 64 samples")
    return array


def _rate(sample_rate_hz: float) -> float:
    rate = float(sample_rate_hz)
    if rate <= 0:
        raise ValueError("Sample rate must be positive")
    return rate


def _check_spec(spec: EmgSpec, rate: float) -> None:
    nyquist = rate / 2.0
    if spec.order < 1:
        raise ValueError("Filter order must be at least 1")
    if spec.notch_q <= 0:
        raise ValueError("Notch Q must be positive")
    if not (0 < spec.band_low_hz < spec.band_high_hz < nyquist):
        raise ValueError(
            "EMG bandpass must sit inside 0 Hz and half the sample rate. "
            "10–250 Hz needs a sample rate above 500 Hz."
        )
    if not (0 < spec.notch_hz < nyquist):
        raise ValueError("Notch must sit between 0 Hz and half the sample rate")
    if spec.envelope not in {"rms", "mean"}:
        raise ValueError("Envelope must be 'rms' or 'mean'")
    if spec.window_seconds <= 0:
        raise ValueError("Envelope window must be positive")


def _window_samples(window_seconds: float, rate: float, n_samples: int) -> int:
    window = int(round(window_seconds * rate))
    if window % 2 == 0:
        window += 1
    window = max(window, 1)
    if window >= n_samples:
        raise ValueError("Envelope window must be shorter than the EMG segment")
    return window


def _moving(values: np.ndarray, window: int, *, root: bool) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float64) / window
    out = np.empty_like(values)
    for index in range(values.shape[1]):
        averaged = np.convolve(values[:, index], kernel, mode="same")
        out[:, index] = np.sqrt(np.maximum(averaged, 0.0)) if root else averaged
    return out
