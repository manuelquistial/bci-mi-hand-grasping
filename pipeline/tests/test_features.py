"""Descriptors that follow Euclidean alignment. CSP and FBCSP stay separate."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.features import extract_features
from pipeline.features.csp import csp_features
from pipeline.features.fbcsp import fbcsp_features
from pipeline.features.psd import band_powers
from pipeline.features.tdp import time_domain_parameters
from pipeline.features.ts import time_series


def _classes() -> tuple[list[np.ndarray], list[str]]:
    rng = np.random.default_rng(0)
    rate = 160.0
    times = np.arange(0, 4, 1.0 / rate)
    tone = np.sin(2 * np.pi * 10 * times)
    left = []
    right = []
    for _ in range(6):
        noise = 0.05 * rng.normal(size=(2, times.size))
        left.append(np.vstack([3 * tone, 0.2 * tone]) + noise)
        right.append(np.vstack([0.2 * tone, 3 * tone]) + noise)
    trials = left + right
    labels = ["left"] * len(left) + ["right"] * len(right)
    return trials, labels


def test_time_series_shortens_the_epoch() -> None:
    trial = np.ones((3, 640))
    waveform = time_series([trial, trial * 2])
    assert waveform.shape[1] == 3
    assert waveform.shape[0] < 640
    assert waveform.shape[0] >= 16


def test_tdp_drops_the_channel_that_does_not_separate_classes() -> None:
    rng = np.random.default_rng(1)
    left = []
    right = []
    for _ in range(6):
        shared = rng.normal(size=200)
        left.append(np.vstack([4 * rng.normal(size=200), shared]))
        right.append(np.vstack([0.25 * rng.normal(size=200), shared]))
    names, values = time_domain_parameters(
        left + right,
        ["left"] * 6 + ["right"] * 6,
        ["C3", "C4"],
    )
    assert values.shape == (12, 3)
    assert all(name.startswith("C3 ") for name in names)
    assert not any(name.startswith("C4 ") for name in names)


def test_mu_power_exceeds_beta_for_a_10_hz_sine() -> None:
    rate = 160.0
    times = np.arange(0, 4, 1.0 / rate)
    tone = np.sin(2 * np.pi * 10 * times)
    names, values = band_powers([np.vstack([tone, tone])], rate, ["C3", "C4"])
    mu = names.index("C3 mu")
    beta = names.index("C3 beta")
    theta = names.index("C3 theta")
    assert values[0, mu] > values[0, beta]
    assert values[0, mu] > values[0, theta]


def test_csp_log_variance_separates_left_from_right() -> None:
    trials, labels = _classes()
    names, values = csp_features(trials, labels)
    assert names[0].startswith("left")
    assert names[-1].startswith("right")
    left = values[:6, 0]
    right = values[6:, 0]
    assert abs(float(left.mean() - right.mean())) > 0.5


def test_fbcsp_keeps_four_features_from_the_mu_band() -> None:
    trials, labels = _classes()
    names, values = fbcsp_features(trials, labels, 160.0)
    assert values.shape == (12, 4)
    assert len(names) == 4
    assert any(name.startswith("8-12 Hz") for name in names)
    csp_names, _csp_values = csp_features(trials, labels)
    assert names != csp_names


def test_unknown_feature_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown feature method"):
        extract_features("wavelet", [np.ones((2, 32))], ["left"], 160.0, ["C3", "C4"])
