"""The simulator reads a pipeline session folder without writing it."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from dataset_loaders.recorded_session import RecordedSessionLoader, sample_index_for_time
from simopenbci.cli import run
from simopenbci.sinks.recording import RecordingSink


def _write_session(root: Path) -> None:
    root.mkdir()
    labels = [f"Ch{i + 1}" for i in range(8)]
    samples = [float(i) for i in range(16)]
    times = [10.0, 10.004]
    (root / "signal.f32").write_bytes(struct.pack("<16f", *samples))
    (root / "time.f64").write_bytes(struct.pack("<2d", *times))
    (root / "markers.jsonl").write_text(
        json.dumps(
            {"lsl_time": 10.006, "kind": "trial_start", "code": "T2", "label": "right"}
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "session.json").write_text(
        json.dumps(
            {
                "format": "pipeline-session-v1",
                "session_id": root.name,
                "stream_name": "OpenBCI_EEG",
                "sample_rate_hz": 250.0,
                "units": "uV",
                "channel_labels": labels,
                "n_samples": 2,
                "device": {
                    "board": "OpenBCI Cyton",
                    "eeg_channels": 8,
                    "ground": "BIAS",
                    "reference": "SRB2",
                    "reference_recorded": False,
                    "ground_recorded": False,
                },
            }
        ),
        encoding="utf-8",
    )


def test_loader_reads_samples_and_maps_marker(tmp_path: Path) -> None:
    session = tmp_path / "sess"
    _write_session(session)
    loader = RecordedSessionLoader()
    assert RecordedSessionLoader.detect_from_path(session)
    samples = list(loader.iter_samples(session))
    assert len(samples) == 2
    assert samples[1][0] == pytest.approx(8.0)
    assert loader.metadata.n_channels == 8
    assert loader.metadata.sampling_rate == 250.0
    events = loader.load_events(session)
    assert events[0].label == "right"
    assert events[0].kind == "trial_start"
    assert events[0].sample_index == sample_index_for_time([10.0, 10.004], 10.006)
    assert events[0].sample_index == 1


def test_cli_plays_recorded_session(tmp_path: Path) -> None:
    session = tmp_path / "sess"
    _write_session(session)
    sinks: list[RecordingSink] = []

    def factory() -> RecordingSink:
        sink = RecordingSink()
        sinks.append(sink)
        return sink

    code = run(
        [
            "--dataset",
            "recorded_session",
            "--file",
            str(session),
            "--max-samples",
            "2",
            "--no-dialog",
            "--no-markers",
        ],
        sink_factory=factory,
    )
    assert code == 0
    assert len(sinks[0].samples) == 2
    assert len(sinks[0].samples[0]) == 8


def test_dump_trials_prints_complete_marker_windows(tmp_path: Path, capsys) -> None:
    session = tmp_path / "sess"
    session.mkdir()
    labels = [f"Ch{index + 1}" for index in range(8)]
    count = 8
    samples = [float(index % 8) for index in range(count * 8)]
    times = [10.0 + index / 250.0 for index in range(count)]
    (session / "signal.f32").write_bytes(struct.pack(f"<{count * 8}f", *samples))
    (session / "time.f64").write_bytes(struct.pack(f"<{count}d", *times))
    (session / "markers.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"lsl_time": 10.0, "kind": "trial_start", "code": "T1", "label": "left"}),
                json.dumps({"lsl_time": 10.024, "kind": "rest", "code": "T0", "label": "rest"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (session / "session.json").write_text(
        json.dumps(
            {
                "format": "pipeline-session-v1",
                "session_id": "sess",
                "stream_name": "OpenBCI_EEG",
                "sample_rate_hz": 250.0,
                "units": "uV",
                "channel_labels": labels,
                "n_samples": count,
                "device": {"eeg_channels": 8, "ground": "BIAS", "reference": "SRB2"},
            }
        ),
        encoding="utf-8",
    )
    code = run(
        [
            "--dataset",
            "recorded_session",
            "--file",
            str(session),
            "--dump-trials",
            "--trial-seconds",
            "0.02",
            "--no-dialog",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sample_rate_hz"] == 250
    assert len(payload["trials"]) == 1
    assert payload["trials"][0]["label"] == "left"
    assert len(payload["trials"][0]["samples"]) == 5
