"""Tests for the Cyton profile and the session folder format."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline.cyton import validate_cyton_stream
from pipeline.models import StreamSummary
from pipeline.session import SessionWriter, load_recording, sample_index_for_time


def _summary(*, channels: int = 8, srate: float = 250.0) -> StreamSummary:
    return StreamSummary(
        name="OpenBCI_EEG",
        stream_type="EEG",
        channel_count=channels,
        nominal_srate=srate,
        channel_format="float32",
        source_id="cyton-1",
        hostname="localhost",
        uid="uid",
        channel_labels=tuple(f"Ch{i + 1}" for i in range(channels)),
    )


def test_cyton_rejects_extra_channels() -> None:
    with pytest.raises(ValueError, match="8 channels"):
        validate_cyton_stream(_summary(channels=16))


def test_cyton_rejects_wrong_rate() -> None:
    with pytest.raises(ValueError, match="250"):
        validate_cyton_stream(_summary(srate=125.0))


def test_session_roundtrip_keeps_microvolts_and_markers(tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / "sess1"
    writer = SessionWriter(
        root,
        session_id="sess1",
        channel_labels=tuple(f"Ch{i + 1}" for i in range(8)),
        sample_rate_hz=250.0,
        stream_name="OpenBCI_EEG",
    )
    samples = np.arange(16, dtype=np.float64).reshape(2, 8)
    writer.append_samples(samples, [10.0, 10.004])
    writer.append_marker(10.004, "trial_start", "T1", "left")
    writer.append_marker(10.006, "trial_end", "T1", "left")
    writer.close()

    meta = json.loads((root / "session.json").read_text(encoding="utf-8"))
    assert meta["format"] == "pipeline-session-v1"
    assert meta["device"]["ground"] == "BIAS"
    assert meta["device"]["reference"] == "SRB2"
    assert meta["device"]["reference_recorded"] is False
    assert meta["units"] == "uV"

    recording = load_recording(root)
    assert recording.n_samples == 2
    assert recording.n_channels == 8
    np.testing.assert_allclose(recording.samples[1], np.arange(8, 16), atol=1e-5)
    assert [marker.label for marker in recording.markers] == ["left", "left"]
    assert sample_index_for_time(recording.timestamps, 10.006) == 1
    assert sample_index_for_time(recording.timestamps, 99.0) == 1


def test_sample_index_uses_first_sample_at_or_after_marker() -> None:
    times = [10.0, 10.004, 10.008, 12.0]
    assert sample_index_for_time(times, 10.006) == 2
    assert sample_index_for_time(times, 10.004) == 1
    assert sample_index_for_time(times, 9.0) == 0
    assert sample_index_for_time([], 1.0) == 0
