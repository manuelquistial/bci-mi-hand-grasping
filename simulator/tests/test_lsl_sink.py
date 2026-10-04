"""LSL sink tests — skipped when pylsl/liblsl is unavailable."""
from __future__ import annotations

import pytest

from dataset_loaders.base import DatasetMetadata

pylsl = pytest.importorskip("pylsl")


def _lsl_usable() -> bool:
    try:
        info = pylsl.StreamInfo("pytest_probe", "EEG", 1, 100, "float32", "probe")
        outlet = pylsl.StreamOutlet(info)
        del outlet
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _lsl_usable(), reason="liblsl not available")


def test_lsl_sink_opens_with_channel_names():
    from simopenbci.sinks.lsl import LslSampleSink

    meta = DatasetMetadata(
        name="TestStream",
        n_channels=2,
        sampling_rate=250.0,
        channel_names=["C3", "C4"],
    )
    sink = LslSampleSink(source_id="pytest-simopenbci")
    sink.open(meta)
    sink.push([0.1, 0.2])
    assert sink._info is not None
    assert sink._info.name() == "TestStream"
    assert sink._info.channel_count() == 2
    sink.close()
