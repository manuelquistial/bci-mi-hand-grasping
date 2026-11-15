"""Unit tests for MI control decoder strategies A–D."""

from __future__ import annotations

import math

import numpy as np

from pipeline.online.control_decoder import (
    ControlDecoder,
    ControlDecoderConfig,
    signed_decision_margin,
)
from pipeline.online.model import train_frozen_model


def _toy_svm():
    rng = np.random.default_rng(0)
    trials = []
    labels = []
    for label, freq, ch in (("left", 10.0, 0), ("right", 20.0, 1)) * 10:
        n = 400
        t = np.arange(n) / 250.0
        data = rng.normal(0.0, 0.3, size=(8, n))
        data[ch] += 2.5 * np.sin(2 * np.pi * freq * t)
        trials.append(data)
        labels.append(label)
    return train_frozen_model(
        trials,
        labels,
        sample_rate_hz=250.0,
        classifier_names=("linear_svm",),
        apply_filter=False,
        apply_training_ea=False,
    )


def test_signed_margin_matches_classes_order() -> None:
    model = _toy_svm()
    head = model.head()
    assert list(head.classifier.classes_) == ["left", "right"]
    rng = np.random.default_rng(1)
    rightish = rng.normal(0.0, 0.3, size=(8, 400))
    rightish[1] += 3.0 * np.sin(2 * np.pi * 20.0 * np.arange(400) / 250.0)
    row = model.feature_vector(rightish)
    margin = signed_decision_margin(head.classifier, row)
    assert margin is not None
    label, _ = model.predict(rightish)
    if label == "right":
        assert margin > 0
    else:
        assert margin < 0


def test_baseline_uses_right_minus_left() -> None:
    dec = ControlDecoder(ControlDecoderConfig(strategy="baseline_proba"))
    out = dec.decode(label="right", score=1.0, p_left=0.2, p_right=0.8)
    assert out.valid
    assert abs(out.control - 0.6) < 1e-9


def test_discrete_maps_sign_to_pm1() -> None:
    dec = ControlDecoder(ControlDecoderConfig(strategy="discrete"))
    right = dec.decode(label="right", score=2.0, decision_margin=1.5)
    left = dec.decode(label="left", score=2.0, decision_margin=-1.5)
    assert right.control == 1.0
    assert left.control == -1.0


def test_temporal_discrete_k_resets_between_trials() -> None:
    dec = ControlDecoder(ControlDecoderConfig(strategy="temporal_discrete", window_k=3))
    dec.decode(label="right", score=1.0, decision_margin=1.0)
    dec.decode(label="right", score=1.0, decision_margin=1.0)
    mid = dec.decode(label="left", score=1.0, decision_margin=-0.1)
    # mean of [1,1,-0.1] > 0 → still right
    assert mid.control == 1.0
    dec.reset_trial()
    after = dec.decode(label="left", score=1.0, decision_margin=-0.1)
    assert after.control == -1.0
    assert after.buffer_size == 1


def test_temporal_proportional_bounded() -> None:
    dec = ControlDecoder(
        ControlDecoderConfig(
            strategy="temporal_proportional",
            window_k=2,
            proportional_scale=2.0,
        )
    )
    out = dec.decode(label="right", score=1.0, decision_margin=10.0)
    assert out.valid
    assert -1.0 <= out.control <= 1.0
    assert abs(out.control - math.tanh(2.0 * 10.0)) < 1e-9


def test_rejected_and_non_finite() -> None:
    dec = ControlDecoder(ControlDecoderConfig(strategy="discrete"))
    bad = dec.decode(label="left", score=1.0, rejected=True, reason="emg")
    assert not bad.valid
    nan = dec.decode(label="left", score=float("nan"), decision_margin=float("nan"))
    assert not nan.valid


def test_deterministic_identical_inputs() -> None:
    cfg = ControlDecoderConfig(strategy="temporal_discrete", window_k=3)
    a = ControlDecoder(cfg)
    b = ControlDecoder(cfg)
    seq = [0.5, -0.2, 0.1]
    out_a = [a.decode(label="right", score=1.0, decision_margin=m).control for m in seq]
    out_b = [b.decode(label="right", score=1.0, decision_margin=m).control for m in seq]
    assert out_a == out_b
