"""Notch, then bandpass, on continuous EEG before epoching."""

from __future__ import annotations

import numpy as np

from pipeline.preprocess import FilterSpec, apply_notch_then_bandpass


def _tone(rate: float, seconds: float, hz: float, amplitude: float = 1.0) -> np.ndarray:
    times = np.arange(0, seconds, 1.0 / rate)
    return amplitude * np.sin(2 * np.pi * hz * times)


def _power(samples: np.ndarray, rate: float, hz: float) -> float:
    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(samples.shape[0], 1.0 / rate)
    nearest = int(np.argmin(np.abs(freqs - hz)))
    return float(np.abs(spectrum[nearest]) ** 2)


def test_default_band_keeps_sensorimotor_rhythms() -> None:
    rate = 250.0
    raw = _tone(rate, 8, 12) + _tone(rate, 8, 50, 0.8) + _tone(rate, 8, 40, 0.8)
    samples = np.column_stack([raw, raw])
    filtered = apply_notch_then_bandpass(samples, rate, FilterSpec())
    kept = _power(filtered[:, 0], rate, 12)
    line = _power(filtered[:, 0], rate, 50)
    above = _power(filtered[:, 0], rate, 40)
    assert kept > line * 50
    assert kept > above * 50


def test_notch_removes_line_noise_inside_a_wide_band() -> None:
    rate = 250.0
    raw = _tone(rate, 8, 12) + _tone(rate, 8, 50)
    samples = raw.reshape(-1, 1)
    filtered = apply_notch_then_bandpass(
        samples,
        rate,
        FilterSpec(band_low_hz=1, band_high_hz=80),
    )
    assert _power(filtered[:, 0], rate, 12) > _power(filtered[:, 0], rate, 50) * 50
