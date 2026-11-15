"""Characterization: start_online(registry=…) binds control+observe hubs."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pipeline.models import SampleChunk, StreamSummary
from pipeline.online.model import train_frozen_model
from pipeline.online.registry import OfflineModeSpec, OnlineModelRegistry
from pipeline.service import PipelineService

STATUS_ONLINE_KEYS = frozenset(
    {
        "source",
        "pid",
        "running",
        "samples_seen",
        "emg_samples_seen",
        "error",
        "activity",
        "registry",
        "control_ws_url",
        "observe_ws_url",
        "control_samples",
        "last_control_sample",
        "last_pipeline_stage",
        "last_stream_frame",
    }
)


class _FakeReader:
    def __init__(self, chunks: list[SampleChunk], summary: StreamSummary) -> None:
        self.summary = summary
        self._chunks = list(chunks)
        self.closed = False

    def pull_chunk(self, timeout: float = 0.0, max_samples: int = 1024) -> SampleChunk:
        if self._chunks:
            return self._chunks.pop(0)
        time.sleep(min(timeout, 0.01))
        return SampleChunk(samples=[], timestamps=[])

    def close(self) -> None:
        self.closed = True


class _Proc:
    def __init__(self) -> None:
        self.pid = 77
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = -9


def _summary() -> StreamSummary:
    return StreamSummary(
        name="OpenBCI_EEG",
        stream_type="EEG",
        channel_count=8,
        nominal_srate=250.0,
        channel_format="float32",
        source_id="cyton",
        hostname="localhost",
        uid="uid",
        channel_labels=tuple(f"Ch{i + 1}" for i in range(8)),
    )


def _chunk() -> SampleChunk:
    rows = np.arange(16, dtype=np.float64).reshape(2, 8).tolist()
    return SampleChunk(samples=rows, timestamps=[1.0, 1.004])


def _train_and_register(root: Path) -> str:
    rng = np.random.default_rng(3)
    trials, labels = [], []
    for label, freq, ch in (("left", 10.0, 0), ("right", 20.0, 1)) * 6:
        n = 400
        t = np.arange(n) / 250.0
        data = rng.normal(0.0, 0.3, size=(8, n))
        data[ch] += 2.0 * np.sin(2 * np.pi * freq * t)
        trials.append(data)
        labels.append(label)
    model = train_frozen_model(
        trials,
        labels,
        sample_rate_hz=250.0,
        classifier_names=("linear_svm",),
        apply_filter=False,
        apply_training_ea=False,
    )
    model = replace(
        model,
        meta={"held_out_online_subject": True, "n_train_subjects": 2},
    )
    registry = OnlineModelRegistry(root)
    entry = registry.register(
        model,
        OfflineModeSpec(
            name="contract-loso",
            dataset="shuqfa2024",
            online_subject="S001",
            classifiers=("linear_svm",),
            control_classifier="linear_svm",
            hold_out_online_subject=True,
        ),
        run_id="contract-loso-pack",
        replace=True,
    )
    return entry.id


def test_start_online_with_registry_binds_both_hubs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_root = tmp_path / "online-models"
    pack_id = _train_and_register(registry_root)
    monkeypatch.setattr(
        "pipeline.online.registry.default_registry_root",
        lambda: registry_root,
    )

    control_port = 18290
    service = PipelineService(
        tmp_path / "recordings",
        eeg_opener=lambda **_: _FakeReader([_chunk(), _chunk()], _summary()),
        process_starter=lambda command: _Proc(),  # type: ignore[return-value, arg-type]
    )
    started = service.start_online(
        source="simulator",
        lsl_name="OpenBCI_EEG",
        dataset="shuqfa2024",
        subject="S001",
        registry=pack_id,
        control_ws_port=control_port,
        wait_seconds=0.1,
    )
    try:
        assert started["control_ws_url"] == f"ws://127.0.0.1:{control_port}"
        assert started["observe_ws_url"] == f"ws://127.0.0.1:{control_port + 1}"
        assert started["registry"] == pack_id

        deadline = time.monotonic() + 3
        online = None
        while time.monotonic() < deadline:
            online = service.status().get("online")
            if online and online.get("control_ws_url"):
                break
            time.sleep(0.02)
        assert online is not None
        assert set(online) == STATUS_ONLINE_KEYS
        assert online["control_ws_url"] == f"ws://127.0.0.1:{control_port}"
        assert online["observe_ws_url"] == f"ws://127.0.0.1:{control_port + 1}"
        assert online["registry"] == pack_id
    finally:
        stopped = service.stop_online()
        assert stopped["stopped"] is True


def test_list_online_models_loso_only_via_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_root = tmp_path / "online-models"
    pack_id = _train_and_register(registry_root)
    # Also register a non-LOSO-looking entry by clearing held_out flags in offline.
    registry = OnlineModelRegistry(registry_root)
    entry = registry.get(pack_id)
    pooled = entry
    pooled.offline["held_out_online_subject"] = False
    pooled.offline["hold_out_online_subject"] = False
    pooled.id = "pooled-pack"
    pooled.name = "pooled"
    folder = registry_root / "pooled-pack"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(
        json.dumps(pooled.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    # model file not needed for list
    monkeypatch.setattr(
        "pipeline.online.registry.default_registry_root",
        lambda: registry_root,
    )
    service = PipelineService(tmp_path / "recordings")
    all_models = service.list_online_models(loso_only=False)["models"]
    loso = service.list_online_models(loso_only=True)["models"]
    assert any(item["id"] == pack_id for item in all_models)
    assert any(item["id"] == "pooled-pack" for item in all_models)
    assert any(item["id"] == pack_id for item in loso)
    assert all(item["held_out"] for item in loso)
    assert not any(item["id"] == "pooled-pack" for item in loso)
