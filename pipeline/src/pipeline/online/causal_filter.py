"""Causal notch + bandpass for real-time EEG (no future samples).

Offline training uses ``filtfilt``. Online uses the same IIR coefficients with
``lfilter`` and persistent ``zi`` state so each sample only sees the past.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

from pipeline.preprocess import FilterSpec


class CausalNotchBandpass:
    """Stateful causal filter matching :class:`~pipeline.preprocess.FilterSpec`."""

    def __init__(self, sample_rate_hz: float, spec: FilterSpec | None = None) -> None:
        self.spec = spec or FilterSpec()
        self.sample_rate_hz = float(sample_rate_hz)
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be positive")
        nyquist = self.sample_rate_hz / 2.0
        if not (0 < self.spec.notch_hz < nyquist):
            raise ValueError("Notch must sit between 0 Hz and half the sample rate")
        if not (0 < self.spec.band_low_hz < self.spec.band_high_hz < nyquist):
            raise ValueError("Bandpass edges are invalid for this sample rate")
        self._notch_b, self._notch_a = iirnotch(
            self.spec.notch_hz, self.spec.notch_q, fs=self.sample_rate_hz
        )
        self._band_b, self._band_a = butter(
            self.spec.order,
            [self.spec.band_low_hz, self.spec.band_high_hz],
            btype="bandpass",
            fs=self.sample_rate_hz,
        )
        self._n_channels: int | None = None
        self._notch_zi: np.ndarray | None = None
        self._band_zi: np.ndarray | None = None

    def reset(self, n_channels: int | None = None) -> None:
        channels = n_channels if n_channels is not None else self._n_channels
        self._n_channels = channels
        self._notch_zi = None
        self._band_zi = None

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Filter ``(samples, channels)`` in time order. Updates internal state."""
        array = np.asarray(samples, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[0] < 1:
            raise ValueError("Need a continuous EEG segment shaped (samples, channels)")
        if self._n_channels is None:
            self._n_channels = int(array.shape[1])
        if array.shape[1] != self._n_channels:
            raise ValueError("Channel count changed mid-stream")
        if self._notch_zi is None or self._band_zi is None:
            self._notch_zi = self._init_zi(self._notch_b, self._notch_a, array[0])
            self._band_zi = self._init_zi(self._band_b, self._band_a, array[0])
        notched, self._notch_zi = lfilter(
            self._notch_b, self._notch_a, array, axis=0, zi=self._notch_zi
        )
        banded, self._band_zi = lfilter(
            self._band_b, self._band_a, notched, axis=0, zi=self._band_zi
        )
        return banded

    @staticmethod
    def _init_zi(b: np.ndarray, a: np.ndarray, first_row: np.ndarray) -> np.ndarray:
        base = lfilter_zi(b, a)
        return np.outer(base, first_row)
