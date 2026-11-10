"""Simulator → windows → online pipeline (no game)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.models import SampleChunk, StreamSummary
from pipeline.online.dataset_model import train_frozen_from_dataset
from pipeline.online.model import train_frozen_model
from pipeline.online.simulator_session import (
    SimulatorOnlineConfig,
    SimulatorOnlineSession,
)
from pipeline.preprocess import FilterSpec


class _Proc:
    def __init__(self) -> None:
        self.pid = 7
        self.returncode: int | None = None
        self.output_lines: list[str] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = -9


class _FakeReader:
    def __init__(self, chunks: list[SampleChunk], summary: StreamSummary) -> None:
        self._chunks = list(chunks)
        self.summary = summary
        self.closed = False

    def pull_chunk(self, timeout: float = 0.0, max_samples: int = 1024) -> SampleChunk:
        del timeout, max_samples
        if self._chunks:
            return self._chunks.pop(0)
        return SampleChunk(samples=[], timestamps=[])

    def close(self) -> None:
        self.closed = True


def _summary(channels: int = 8, rate: float = 250.0) -> StreamSummary:
    return StreamSummary(
        name="OpenBCI_EEG",
        stream_type="EEG",
        channel_count=channels,
        nominal_srate=rate,
        channel_format="float32",
        source_id="test",
        hostname="localhost",
        uid="uid",
        channel_labels=tuple(f"Ch{i}" for i in range(channels)),
    )


def _hand_bundle(channels: int = 8, rate: float = 250.0) -> dict:
    rng = np.random.default_rng(0)
    samples = int(rate * 4)
    t = np.arange(samples) / rate
    trials = []
    for label, freq, ch in (("left", 10.0, 0), ("right", 20.0, 1)) * 6:
        data = rng.normal(0.0, 0.4, size=(samples, channels))
        data[:, ch] += 2.0 * np.sin(2 * np.pi * freq * t)
        trials.append(
            {
                "kind": "trial_start",
                "label": label,
                "samples": data.tolist(),
            }
        )
    return {
        "sample_rate_hz": rate,
        "channels": [f"Ch{i}" for i in range(channels)],
        "trials": trials,
    }


def test_train_frozen_from_dataset_loader() -> None:
    bundle = _hand_bundle()

    def loader(**kwargs):
        del kwargs
        return bundle

    model = train_frozen_from_dataset(
        dataset="shuqfa2024",
        subject="S1",
        classifier_name="logistic_regression",
        trial_loader=loader,
        filter=FilterSpec(),
    )
    assert model.channel_count == 8
    assert model.offline_accuracy is not None
    assert model.meta["dataset"] == "shuqfa2024"


def test_simulator_session_classifies_streamed_windows(tmp_path: Path) -> None:
    rate = 250.0
    channels = 8
    rng = np.random.default_rng(1)
    # Build a continuous stream: 3 s rest + 4 s left imagery.
    rest_n = int(rate * 3)
    img_n = int(rate * 4.2)
    rest = rng.normal(0.0, 0.3, size=(rest_n, channels))
    imagery = rng.normal(0.0, 0.3, size=(img_n, channels))
    t = np.arange(img_n) / rate
    imagery[:, 0] += 2.0 * np.sin(2 * np.pi * 10 * t)

    eeg_chunks: list[SampleChunk] = []
    # Push rest then imagery in two chunks with absolute timestamps.
    eeg_chunks.append(
        SampleChunk(
            samples=rest.tolist(),
            timestamps=[i / rate for i in range(rest_n)],
        )
    )
    eeg_chunks.append(
        SampleChunk(
            samples=imagery.tolist(),
            timestamps=[3.0 + i / rate for i in range(img_n)],
        )
    )
    # Empty chunks so the session can idle-finish after the proc exits.
    for _ in range(3):
        eeg_chunks.append(SampleChunk(samples=[], timestamps=[]))

    marker_chunks = [
        SampleChunk(samples=[["rest:T0:rest"]], timestamps=[0.0]),
        SampleChunk(samples=[["trial_start:T1:left"]], timestamps=[3.0]),
        SampleChunk(samples=[], timestamps=[]),
    ]

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
        classifier_name="linear_svm",
        apply_filter=True,
    )

    proc = _Proc()
    eeg = _FakeReader(eeg_chunks, _summary(channels, rate))
    markers = _FakeReader(marker_chunks, _summary(1, rate))

    def starter(command: list[str]) -> _Proc:
        del command
        return proc

    session = SimulatorOnlineSession(
        model,
        SimulatorOnlineConfig(
            dataset="shuqfa2024",
            max_trials=1,
            log_path=tmp_path / "online.jsonl",
            idle_timeout_seconds=0.2,
            wait_seconds=0.1,
        ),
        process_starter=starter,
        eeg_opener=lambda **_: eeg,
        marker_opener=lambda **_: markers,
    )

    # Exit the simulator after the first poll cycle by marking the process done
    # once EEG has been drained far enough — hook via wrapping pull.
    original_pull = session._pull

    def pull_and_exit(timeout: float = 0.1) -> int:
        added = original_pull(timeout)
        if not eeg._chunks:
            proc.returncode = 0
        return added

    session._pull = pull_and_exit  # type: ignore[method-assign]
    summary = session.run()
    assert summary["n_trials"] == 1
    assert eeg.closed and markers.closed
    text = (tmp_path / "online.jsonl").read_text(encoding="utf-8")
    assert '"event": "window"' in text or '"event":"window"' in text.replace(" ", "")
    # Prefer the spaced JSON form written by json.dumps
    assert "window" in text
    assert "trial_end" in text
