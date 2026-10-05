"""Forearm EMG inlet: 8 electrodes on two arms.

Each forearm carries 4 electrodes, 2 on the anterior side and 2 on the posterior
side. The LSL stream is type EMG. These columns are muscles, not the Cyton EEG.
"""

from __future__ import annotations

from pipeline.models import StreamSummary

EMG_CHANNEL_COUNT = 8

# Column order of emg.f32. The stream is expected in this order.
EMG_CHANNELS: tuple[dict[str, str], ...] = (
    {"label": "left_forearm_anterior_1", "arm": "left", "side": "anterior"},
    {"label": "left_forearm_anterior_2", "arm": "left", "side": "anterior"},
    {"label": "left_forearm_posterior_1", "arm": "left", "side": "posterior"},
    {"label": "left_forearm_posterior_2", "arm": "left", "side": "posterior"},
    {"label": "right_forearm_anterior_1", "arm": "right", "side": "anterior"},
    {"label": "right_forearm_anterior_2", "arm": "right", "side": "anterior"},
    {"label": "right_forearm_posterior_1", "arm": "right", "side": "posterior"},
    {"label": "right_forearm_posterior_2", "arm": "right", "side": "posterior"},
)


def channel_labels() -> tuple[str, ...]:
    return tuple(item["label"] for item in EMG_CHANNELS)


def montage() -> list[dict[str, str]]:
    return [dict(item) for item in EMG_CHANNELS]


def validate_emg_stream(summary: StreamSummary) -> None:
    """Reject an inlet that is not the 8-electrode forearm EMG."""
    if summary.stream_type and summary.stream_type.upper() != "EMG":
        raise ValueError(
            f"EMG stream type must be EMG, got {summary.stream_type!r}."
        )
    if summary.channel_count != EMG_CHANNEL_COUNT:
        raise ValueError(
            f"EMG stream must have {EMG_CHANNEL_COUNT} channels, "
            f"got {summary.channel_count}. "
            "4 on the left forearm and 4 on the right; "
            "2 anterior and 2 posterior on each."
        )
    if float(summary.nominal_srate) <= 0:
        raise ValueError("EMG stream must publish a positive sample rate.")
