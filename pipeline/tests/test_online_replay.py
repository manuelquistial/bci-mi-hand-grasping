"""File replay online session (no LSL)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.online.model import train_frozen_model
from pipeline.online.replay_session import ReplayOnlineConfig, ReplayOnlineSession


def test_replay_session_classifies_from_dump(tmp_path: Path) -> None:
    rate = 250.0
    channels = 8
    rng = np.random.default_rng(0)
    # 3 s rest + 4.2 s left imagery
    rest_n = int(rate * 3)
    img_n = int(rate * 4.2)
    rest = rng.normal(0.0, 0.3, size=(rest_n, channels))
    imagery = rng.normal(0.0, 0.3, size=(img_n, channels))
    t = np.arange(img_n) / rate
    imagery[:, 0] += 2.0 * np.sin(2 * np.pi * 10 * t)
    samples = np.vstack([rest, imagery])

    trials = []
    labels = []
    for label, freq, ch in (("left", 10.0, 0), ("right", 20.0, 1)) * 8:
        n = int(rate * 4)
        tt = np.arange(n) / rate
        data = rng.normal(0.0, 0.4, size=(channels, n))
        data[ch] += 2.0 * np.sin(2 * np.pi * freq * tt)
        trials.append(data)
        labels.append(label)
    model = train_frozen_model(
        trials,
        labels,
        sample_rate_hz=rate,
        classifier_names=("linear_svm", "logistic_regression"),
        apply_filter=False,
    )

    def stream_loader(**kwargs):
        del kwargs
        return {"sample_rate_hz": rate, "samples": samples.tolist()}

    def marker_loader(**kwargs):
        del kwargs
        return {
            "trials": [
                {"onset": 0.0, "kind": "rest", "label": "rest", "samples": [[0.0] * channels]},
                {
                    "onset": 3.0,
                    "kind": "trial_start",
                    "label": "left",
                    "samples": imagery[: int(rate * 4)].tolist(),
                },
            ]
        }

    session = ReplayOnlineSession(
        model,
        ReplayOnlineConfig(
            dataset="shuqfa2024",
            subject="1",
            max_trials=1,
            log_path=tmp_path / "replay.jsonl",
        ),
        stream_loader=stream_loader,
        marker_loader=marker_loader,
    )
    summary = session.run()
    assert summary["n_trials"] == 1
    assert "classifiers" in summary
    assert (tmp_path / "replay.jsonl").exists()
