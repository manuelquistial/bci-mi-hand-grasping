"""Forearm EMG montage: 4 electrodes per arm, 2 anterior and 2 posterior."""

from __future__ import annotations

import pytest

from pipeline.emg import EMG_CHANNELS, validate_emg_stream
from pipeline.models import StreamSummary


def _summary(*, channels: int = 8, stream_type: str = "EMG", srate: float = 1000.0) -> StreamSummary:
    return StreamSummary(
        name="Forearm_EMG",
        stream_type=stream_type,
        channel_count=channels,
        nominal_srate=srate,
        channel_format="float32",
        source_id="emg",
        hostname="localhost",
        uid="uid",
    )


def test_montage_is_four_per_forearm() -> None:
    assert len(EMG_CHANNELS) == 8
    for arm in ("left", "right"):
        arm_channels = [item for item in EMG_CHANNELS if item["arm"] == arm]
        assert len(arm_channels) == 4
        assert sum(item["side"] == "anterior" for item in arm_channels) == 2
        assert sum(item["side"] == "posterior" for item in arm_channels) == 2


def test_emg_rejects_the_wrong_width() -> None:
    with pytest.raises(ValueError, match="8 channels"):
        validate_emg_stream(_summary(channels=4))
