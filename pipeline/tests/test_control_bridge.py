"""Control WebSocket bridge maps frozen decisions to game samples."""

from __future__ import annotations

import numpy as np

from pipeline.online.control_bridge import decision_to_control_sample
from pipeline.online.model import train_frozen_model
from pipeline.online.types import WindowDecision


def _toy_model():
    rng = np.random.default_rng(0)
    trials = []
    labels = []
    for label, freq, ch in (("left", 10.0, 0), ("right", 20.0, 1)) * 8:
        n = 500
        t = np.arange(n) / 250.0
        data = rng.normal(0.0, 0.4, size=(8, n))
        data[ch] += 2.0 * np.sin(2 * np.pi * freq * t)
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


def test_decision_to_control_sample_uses_proba_pair() -> None:
    model = _toy_model()
    rng = np.random.default_rng(1)
    trial = rng.normal(0.0, 0.4, size=(8, 400))
    trial[0] += 2.0 * np.sin(2 * np.pi * 10.0 * np.arange(400) / 250.0)
    label, score = model.predict(trial)
    decision = WindowDecision(
        window_index=0,
        start=0.0,
        end=2.0,
        label=label,  # type: ignore[arg-type]
        score=score,
        emg_rejected=False,
        reason="classified",
        classifier="linear_svm",
        predictions=(),
    )
    sample = decision_to_control_sample(
        decision,
        model=model,
        trial_channels_by_time=trial,
        timestamp=12.5,
    )
    assert sample is not None
    assert 0.0 <= sample["left"] <= 1.0
    assert 0.0 <= sample["right"] <= 1.0
    assert abs(sample["left"] + sample["right"] - 1.0) < 1e-6
    assert sample["timestamp"] == 12.5


def test_decision_to_control_sample_skips_rejected() -> None:
    decision = WindowDecision(
        window_index=0,
        start=0.0,
        end=2.0,
        label=None,
        score=None,
        emg_rejected=True,
        reason="emg",
        classifier="linear_svm",
        predictions=(),
    )
    assert decision_to_control_sample(decision) is None
