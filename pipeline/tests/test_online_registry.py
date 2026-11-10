"""Offline export registry for online reuse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline.online.model import train_frozen_model
from pipeline.online.registry import (
    OfflineModeSpec,
    OnlineModelRegistry,
    export_offline_modes,
    load_offline_modes,
)


def _bundle() -> dict:
    rng = np.random.default_rng(0)
    rate = 250.0
    channels = 8
    samples = int(rate * 4)
    t = np.arange(samples) / rate
    trials = []
    for label, freq, ch in (("left", 10.0, 0), ("right", 20.0, 1)) * 6:
        data = rng.normal(0.0, 0.4, size=(samples, channels))
        data[:, ch] += 2.0 * np.sin(2 * np.pi * freq * t)
        trials.append({"kind": "trial_start", "label": label, "samples": data.tolist()})
    return {
        "sample_rate_hz": rate,
        "channels": [f"Ch{i}" for i in range(channels)],
        "trials": trials,
    }


def test_export_and_load_from_registry(tmp_path: Path) -> None:
    bundle = _bundle()

    def loader(**kwargs):
        del kwargs
        return bundle

    registry = OnlineModelRegistry(tmp_path / "online-models")
    modes = [
        OfflineModeSpec(
            name="demo-svm-lr",
            dataset="shuqfa2024",
            subject="S1",
            classifiers=("linear_svm", "logistic_regression"),
            control_classifier="linear_svm",
        )
    ]
    entries = export_offline_modes(modes, registry=registry, trial_loader=loader)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "demo-svm-lr"
    assert entry.classifiers == ["linear_svm", "logistic_regression"]
    assert (registry.root / entry.id / "model.joblib").exists()
    assert (registry.root / entry.id / "manifest.json").exists()
    assert registry.index_path.exists()

    listed = registry.list_entries()
    assert listed[0].id == entry.id
    loaded = registry.load_model(entry.id)
    assert loaded.classifier_names == ("linear_svm", "logistic_regression")
    by_name = registry.load_model("demo-svm-lr")
    assert by_name.channel_count == loaded.channel_count
    latest = registry.latest(dataset="shuqfa2024", subject="S1")
    assert latest.id == entry.id


def test_load_offline_modes_config(tmp_path: Path) -> None:
    path = tmp_path / "modes.json"
    path.write_text(
        json.dumps(
            {
                "modes": [
                    {
                        "name": "a",
                        "dataset": "shuqfa2024",
                        "subject": "S1",
                        "classifiers": ["linear_svm"],
                    },
                    {
                        "name": "b",
                        "dataset": "liu2024_stroke",
                        "classifiers": "logistic_regression",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    modes = load_offline_modes(path)
    assert [mode.name for mode in modes] == ["a", "b"]
    assert modes[0].resolved_classifiers() == ("linear_svm",)
    assert modes[1].resolved_classifiers() == ("logistic_regression",)


def test_register_single_trained_model(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    trials = []
    labels = []
    for label, freq, ch in (("left", 10.0, 0), ("right", 20.0, 1)) * 8:
        n = 500
        t = np.arange(n) / 250.0
        data = rng.normal(0.0, 0.4, size=(8, n))
        data[ch] += 2.0 * np.sin(2 * np.pi * freq * t)
        trials.append(data)
        labels.append(label)
    model = train_frozen_model(
        trials,
        labels,
        sample_rate_hz=250.0,
        classifier_names=("linear_svm", "logistic_regression"),
        apply_filter=False,
    )
    registry = OnlineModelRegistry(tmp_path / "reg")
    entry = registry.register(
        model,
        OfflineModeSpec(
            name="pack",
            dataset="shuqfa2024",
            subject="S9",
            classifiers=("linear_svm", "logistic_regression"),
        ),
    )
    assert registry.get("pack").id == entry.id
