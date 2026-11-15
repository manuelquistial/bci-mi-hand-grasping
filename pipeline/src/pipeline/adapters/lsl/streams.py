"""Open LSL EEG, EMG, and marker inlets."""

from __future__ import annotations

import time
from typing import Any

from pipeline.emg import validate_emg_stream


def lsl_clock() -> float:
    """LSL clock when pylsl is available, otherwise the wall clock."""
    try:
        from pylsl import local_clock

        return float(local_clock())
    except Exception:
        return time.time()


def open_marker_stream(*, eeg_name: str, wait_seconds: float) -> Any | None:
    """Markers inlet paired with the EEG stream, if one is published."""
    from pipeline.reader import LslStreamReader

    try:
        return LslStreamReader.open(
            name=f"{eeg_name}_Markers",
            wait_seconds=wait_seconds,
        )
    except TimeoutError:
        return None


def open_eeg_stream(*, stream_name: str | None, wait_seconds: float) -> Any:
    from pipeline.reader import LslStreamReader

    if stream_name:
        return LslStreamReader.open(name=stream_name, wait_seconds=wait_seconds)
    return LslStreamReader.open(stream_type="EEG", wait_seconds=wait_seconds)


def open_emg_stream(*, wait_seconds: float) -> Any:
    """Forearm EMG inlet. Eight electrodes, type EMG."""
    from pipeline.reader import LslStreamReader

    try:
        reader = LslStreamReader.open(stream_type="EMG", wait_seconds=wait_seconds)
    except TimeoutError as exc:
        raise TimeoutError(
            "No EMG LSL stream. Expect 8 forearm electrodes: "
            "4 on the left arm and 4 on the right, "
            "2 anterior and 2 posterior on each forearm."
        ) from exc
    validate_emg_stream(reader.summary)
    return reader
