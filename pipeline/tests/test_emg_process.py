"""EMG: bandpass 10–250 Hz, line notch, rectify, then a short-window envelope."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.emg_process import EmgSpec, apply_emg, condition_emg


def _tone(rate: float, seconds: float, hz: float, amplitude: float = 1.0) -> np.ndarray:
    times = np.arange(0, seconds, 1.0 / rate)
    return amplitude * np.sin(2 * np.pi * hz * times)


def _power(samples: np.ndarray, rate: float, hz: float) -> float:
    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(samples.shape[0], 1.0 / rate)
    nearest = int(np.argmin(np.abs(freqs - hz)))
    return float(np.abs(spectrum[nearest]) ** 2)


def _middle(samples: np.ndarray) -> np.ndarray:
    count = samples.shape[0]
    return samples[count // 4 : 3 * count // 4]


def test_bandpass_then_notch_keeps_emg_and_drops_line_noise() -> None:
    rate = 1000.0
    raw = (
        _tone(rate, 4, 80)
        + _tone(rate, 4, 50)
        + _tone(rate, 4, 5, 0.8)
        + _tone(rate, 4, 400, 0.8)
    )
    conditioned = condition_emg(np.column_stack([raw, raw]), rate, EmgSpec())
    channel = conditioned[:, 0]
    kept = _power(channel, rate, 80)
    assert kept > _power(channel, rate, 50) * 20
    assert kept > _power(channel, rate, 5) * 20
    assert kept > _power(channel, rate, 400) * 20


def test_envelope_is_the_rectified_short_window_rms() -> None:
    rate = 1000.0
    tone = _tone(rate, 4, 80)
    envelope = apply_emg(tone, rate, EmgSpec())
    body = _middle(envelope[:, 0])
    assert np.all(envelope >= -1e-9)
    # A unit sine has RMS 1/sqrt(2) after the band keeps it.
    assert 0.5 < float(np.mean(body)) < 0.9


def test_mean_envelope_is_also_non_negative() -> None:
    rate = 1000.0
    envelope = apply_emg(_tone(rate, 4, 80), rate, EmgSpec(envelope="mean"))
    assert np.all(envelope >= -1e-9)
    assert float(np.mean(_middle(envelope[:, 0]))) > 0.4


def test_line_only_input_collapses() -> None:
    rate = 1000.0
    line = apply_emg(_tone(rate, 4, 50), rate)
    muscle = apply_emg(_tone(rate, 4, 80), rate)
    assert float(np.mean(_middle(muscle[:, 0]))) > float(np.mean(_middle(line[:, 0]))) * 5


def test_band_requires_a_sample_rate_above_500_hz() -> None:
    with pytest.raises(ValueError, match="500"):
        condition_emg(np.zeros((200, 1)), 250.0)
