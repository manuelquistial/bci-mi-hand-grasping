"""Ports (interfaces) for playback adapters."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from dataset_loaders.base import DatasetMetadata


@runtime_checkable
class SampleSink(Protocol):
    """Destination for EEG samples (LSL outlet, in-memory recorder, etc.)."""

    def open(self, metadata: DatasetMetadata, stream_name: str | None = None) -> None:
        """Prepare the sink for streaming with the given metadata."""
        ...

    def push(self, sample: list[float]) -> None:
        """Push one sample (n_channels floats)."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


@runtime_checkable
class Notifier(Protocol):
    """User-facing messages (console or Qt dialogs)."""

    def info(self, message: str, *, title: str = "Info") -> None: ...

    def warning(self, message: str, *, title: str = "Warning") -> None: ...

    def error(self, message: str, *, title: str = "Error") -> None: ...

    def progress(self, message: str) -> None: ...


@runtime_checkable
class Clock(Protocol):
    """Playback timing control."""

    def start(self) -> None:
        """Mark the start of a playback session."""
        ...

    def wait_for_sample(self, sample_index: int, sampling_rate: float) -> None:
        """
        Block until the wall-clock time for sample_index at the given rate.
        sample_index is 0-based and refers to the sample just pushed.
        """
        ...


@runtime_checkable
class FilePicker(Protocol):
    """Optional GUI file selection."""

    def pick_file(self) -> str | None: ...
