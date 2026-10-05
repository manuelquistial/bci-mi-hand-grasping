"""Pull an LSL inlet into a session directory."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from pipeline.cyton import channel_labels_for, validate_cyton_stream
from pipeline.emg import validate_emg_stream
from pipeline.models import SampleChunk, StreamSummary
from pipeline.session import EmgWriter, SessionWriter


class ChunkSource(Protocol):
    summary: StreamSummary

    def pull_chunk(self, timeout: float = 0.0, max_samples: int = 1024) -> SampleChunk: ...

    def close(self) -> None: ...


Clock = Callable[[], float]


class SessionRecorder:
    """Background recorder. Markers are stamped with ``clock`` (LSL time)."""

    def __init__(
        self,
        reader: ChunkSource,
        writer: SessionWriter,
        *,
        clock: Clock,
        pull_timeout: float = 0.05,
        on_samples: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self._clock = clock
        self._pull_timeout = pull_timeout
        self._on_samples = on_samples
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        validate_cyton_stream(self.reader.summary)
        self._thread = threading.Thread(
            target=self._run,
            name=f"record-{self.writer.session_id}",
            daemon=True,
        )
        self._thread.start()

    def add_marker(self, kind: str, code: str, label: str) -> None:
        self.writer.append_marker(self._clock(), kind, code, label)

    def stop(self) -> Path:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.reader.close()
        self.writer.close()
        if self._error is not None:
            raise self._error
        return self.writer.root

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._pull_once()
            self._pull_once()
        except BaseException as exc:  # noqa: BLE001 — surfaced from stop()
            self._error = exc

    def _pull_once(self) -> None:
        chunk = self.reader.pull_chunk(timeout=self._pull_timeout, max_samples=2048)
        if not chunk.samples:
            return
        array = np.asarray(chunk.samples, dtype=np.float64)
        self.writer.append_samples(array, chunk.timestamps)
        if self._on_samples is not None:
            self._on_samples(array)


class EmgRecorder:
    """Pull the forearm EMG inlet into the session folder."""

    def __init__(
        self,
        reader: ChunkSource,
        writer: EmgWriter,
        *,
        pull_timeout: float = 0.05,
    ) -> None:
        validate_emg_stream(reader.summary)
        self.reader = reader
        self.writer = writer
        self._pull_timeout = pull_timeout
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"record-emg-{self.writer.root.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.reader.close()
        count = self.writer.sample_count
        self.writer.close()
        if self._error is not None:
            raise self._error
        return count

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._pull_once()
            self._pull_once()
        except BaseException as exc:  # noqa: BLE001 — surfaced from stop()
            self._error = exc

    def _pull_once(self) -> None:
        chunk = self.reader.pull_chunk(timeout=self._pull_timeout, max_samples=2048)
        if not chunk.samples:
            return
        array = np.asarray(chunk.samples, dtype=np.float64)
        self.writer.append_samples(array, chunk.timestamps)


def open_session_writer(root: Path, reader: ChunkSource) -> SessionWriter:
    """Create the session directory from a validated Cyton inlet."""
    validate_cyton_stream(reader.summary)
    labels = channel_labels_for(reader.summary)
    return SessionWriter(
        root,
        session_id=root.name,
        channel_labels=labels,
        sample_rate_hz=float(reader.summary.nominal_srate),
        stream_name=reader.summary.name,
    )
