"""OpenBCI Cyton acquisition profile.

The board exposes 8 EEG channels. The ground pin (BIAS) and the reference
pin (SRB2) are electrode connections, not samples in the stream.
"""

from __future__ import annotations

from pipeline.models import StreamSummary

BOARD = "OpenBCI Cyton"
EEG_CHANNELS = 8
SAMPLE_RATE_HZ = 250.0
UNITS = "uV"
GAIN = 24
RESOLUTION_BITS = 24
REFERENCE = "SRB2"
GROUND = "BIAS"


def device_metadata() -> dict[str, object]:
    """Metadata stored beside the signal. Pins are documented, not recorded."""
    return {
        "board": BOARD,
        "eeg_channels": EEG_CHANNELS,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "units": UNITS,
        "gain": GAIN,
        "resolution_bits": RESOLUTION_BITS,
        "reference": REFERENCE,
        "ground": GROUND,
        "reference_recorded": False,
        "ground_recorded": False,
    }


def channel_labels_for(summary: StreamSummary) -> tuple[str, ...]:
    """LSL labels, falling back to Ch1..ChN when the stream has none."""
    labels = [label for label in summary.channel_labels if label]
    if len(labels) < summary.channel_count:
        labels.extend(
            f"Ch{index + 1}" for index in range(len(labels), summary.channel_count)
        )
    return tuple(labels[: summary.channel_count])


def validate_cyton_stream(summary: StreamSummary) -> None:
    """Reject an inlet that is not an 8-channel Cyton at 250 Hz."""
    if summary.channel_count != EEG_CHANNELS:
        raise ValueError(
            f"Cyton EEG stream must have {EEG_CHANNELS} channels, "
            f"got {summary.channel_count}. "
            "BIAS and SRB2 are pins, not extra channels."
        )
    if abs(float(summary.nominal_srate) - SAMPLE_RATE_HZ) > 0.1:
        raise ValueError(
            f"Cyton EEG stream must be {SAMPLE_RATE_HZ:g} Hz, "
            f"got {summary.nominal_srate:g} Hz."
        )
