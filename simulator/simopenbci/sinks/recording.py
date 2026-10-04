"""In-memory sample sink for tests."""
from __future__ import annotations

from dataset_loaders.base import DatasetMetadata
from dataset_loaders.events import TrialEvent


class RecordingSink:
    """Records samples in memory; does not require pylsl."""

    def __init__(self) -> None:
        self.samples: list[list[float]] = []
        self.metadata: DatasetMetadata | None = None
        self.stream_name: str | None = None
        self.opened = False
        self.closed = False

    def open(self, metadata: DatasetMetadata, stream_name: str | None = None) -> None:
        self.metadata = metadata
        self.stream_name = (stream_name or metadata.name).strip() or metadata.name
        self.samples.clear()
        self.opened = True
        self.closed = False

    def push(self, sample: list[float]) -> None:
        self.samples.append(list(sample))

    def close(self) -> None:
        self.closed = True


class RecordingMarkerSink:
    """Records trial markers in memory for tests."""

    def __init__(self) -> None:
        self.events: list[tuple[TrialEvent, float | None]] = []
        self.eeg_stream_name: str | None = None
        self.opened = False
        self.closed = False

    def open(self, eeg_stream_name: str) -> None:
        self.eeg_stream_name = eeg_stream_name
        self.events.clear()
        self.opened = True
        self.closed = False

    def push(self, event: TrialEvent, timestamp: float | None = None) -> None:
        self.events.append((event, timestamp))

    def close(self) -> None:
        self.closed = True
