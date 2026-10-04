"""Unit tests that avoid requiring a live LSL source."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline.discover import summarize_stream
from pipeline.models import StreamSummary
from pipeline.reader import LslStreamReader


def _fake_info(
    *,
    name: str = "OpenBCI",
    stream_type: str = "EEG",
    channel_count: int = 8,
    srate: float = 250.0,
) -> MagicMock:
    info = MagicMock()
    info.name.return_value = name
    info.type.return_value = stream_type
    info.channel_count.return_value = channel_count
    info.nominal_srate.return_value = srate
    info.channel_format.return_value = "float32"
    info.source_id.return_value = "sim-1"
    info.hostname.return_value = "localhost"
    info.uid.return_value = "uid-1"

    channel = MagicMock()
    channel.name.return_value = "channel"
    channel.child_value.side_effect = ["Cz", ""]
    channel.next_sibling.return_value = SimpleNamespace(name=lambda: "")

    channels_root = MagicMock()
    channels_root.child.return_value = channel
    desc = MagicMock()
    desc.child.return_value = channels_root
    info.desc.return_value = desc
    return info


def test_summarize_stream_basic_fields() -> None:
    summary = summarize_stream(_fake_info())
    assert isinstance(summary, StreamSummary)
    assert summary.name == "OpenBCI"
    assert summary.stream_type == "EEG"
    assert summary.channel_count == 8
    assert summary.nominal_srate == 250.0
    assert "OpenBCI" in summary.display_name


def test_list_streams_filters_by_type() -> None:
    from pipeline import discover

    infos = [_fake_info(name="A", stream_type="EEG")]
    with patch.dict("sys.modules", {"pylsl": MagicMock()}):
        import sys

        sys.modules["pylsl"].resolve_byprop.return_value = infos
        streams = discover.discover_streams(stream_type="EEG", wait_seconds=0.1)
        sys.modules["pylsl"].resolve_byprop.assert_called_once()
        assert len(streams) == 1
        assert streams[0].name == "A"


def test_reader_open_raises_when_no_streams() -> None:
    with patch.dict("sys.modules", {"pylsl": MagicMock()}):
        import sys

        sys.modules["pylsl"].resolve_byprop.return_value = []
        with pytest.raises(TimeoutError, match="No LSL streams found"):
            LslStreamReader.open(stream_type="EEG", wait_seconds=0.1)


def test_reader_pull_sample_and_chunk() -> None:
    inlet = MagicMock()
    inlet.pull_sample.return_value = ([0.1, 0.2], 12.5)
    inlet.pull_chunk.return_value = ([[0.1, 0.2], [0.3, 0.4]], [12.5, 12.504])

    summary = StreamSummary(
        name="OpenBCI",
        stream_type="EEG",
        channel_count=2,
        nominal_srate=250.0,
        channel_format="float32",
        source_id="sim-1",
        hostname="localhost",
        uid="uid-1",
    )
    reader = LslStreamReader(inlet, summary)

    sample = reader.pull_sample(timeout=0.5)
    assert sample == ([0.1, 0.2], 12.5)

    chunk = reader.pull_chunk(timeout=0.0, max_samples=16)
    assert len(chunk.samples) == 2
    assert chunk.timestamps[1] == 12.504


def test_cli_list_prints_streams(capsys: pytest.CaptureFixture[str]) -> None:
    from pipeline.cli import main

    fake = [
        StreamSummary(
            name="OpenBCI",
            stream_type="EEG",
            channel_count=8,
            nominal_srate=250.0,
            channel_format="float32",
            source_id="sim-1",
            hostname="localhost",
            uid="uid-1",
            channel_labels=("Cz", "C3"),
        )
    ]
    with patch("pipeline.reader.list_streams", return_value=fake):
        code = main(["list", "--wait", "0.1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "OpenBCI" in out
    assert "Found 1 stream" in out


def test_cli_list_empty_returns_1(capsys: pytest.CaptureFixture[str]) -> None:
    from pipeline.cli import main

    with patch("pipeline.reader.list_streams", return_value=[]):
        code = main(["list"])
    assert code == 1
    assert "No LSL streams found" in capsys.readouterr().out
