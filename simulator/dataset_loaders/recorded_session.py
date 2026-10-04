"""Read a session directory written by the pipeline.

This loader only reads. The pipeline owns the recording format.
"""

from __future__ import annotations

import bisect
import json
import struct
from pathlib import Path
from typing import Iterator

from dataset_loaders.base import BaseDatasetLoader, DatasetMetadata
from dataset_loaders.events import TrialEvent

FORMAT = "pipeline-session-v1"
SESSION_NAME = "session.json"
SIGNAL_NAME = "signal.f32"
TIME_NAME = "time.f64"
MARKERS_NAME = "markers.jsonl"


def sample_index_for_time(timestamps: list[float], lsl_time: float) -> int:
    """Same mapping the pipeline uses: first sample at or after ``lsl_time``."""
    if not timestamps:
        return 0
    index = bisect.bisect_left(timestamps, lsl_time)
    if index >= len(timestamps):
        return len(timestamps) - 1
    return index


class RecordedSessionLoader(BaseDatasetLoader):
    """Pipeline session folder: signal.f32 + time.f64 + markers.jsonl + session.json."""

    id = "recorded_session"
    label = "Recorded session"

    def __init__(
        self,
        n_channels: int | None = None,
        sampling_rate: float | None = None,
        stream_name: str | None = None,
    ) -> None:
        self._channel_override = n_channels
        self._rate_override = sampling_rate
        self._name_override = stream_name
        self._root: Path | None = None
        self._metadata: DatasetMetadata | None = None

    @property
    def metadata(self) -> DatasetMetadata:
        if self._metadata is None:
            raise RuntimeError("Call bind() with the session directory before reading metadata")
        return self._metadata

    @classmethod
    def detect_from_path(cls, file_path: str | Path) -> bool:
        path = Path(file_path)
        meta_path = path / SESSION_NAME if path.is_dir() else None
        if meta_path is None or not meta_path.is_file():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return meta.get("format") == FORMAT

    def bind(self, file_path: str | Path) -> None:
        """Load session.json and check it against optional CLI overrides."""
        root = Path(file_path)
        meta_path = root / SESSION_NAME
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing {SESSION_NAME} in {root}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT:
            raise ValueError(f"Unsupported session format: {meta.get('format')!r}")
        labels = [str(label) for label in meta["channel_labels"]]
        if not labels:
            raise ValueError("session.json has no channel labels")
        rate = float(meta["sample_rate_hz"])
        if self._channel_override is not None and self._channel_override != len(labels):
            raise ValueError(
                f"session has {len(labels)} channels, CLI asked for {self._channel_override}"
            )
        if self._rate_override is not None and abs(self._rate_override - rate) > 0.1:
            raise ValueError(
                f"session is {rate:g} Hz, CLI asked for {self._rate_override:g} Hz"
            )
        name = self._name_override or str(meta.get("stream_name") or "OpenBCI_EEG")
        self._root = root
        self._metadata = DatasetMetadata(
            name=name,
            n_channels=len(labels),
            sampling_rate=rate,
            channel_names=labels,
        )

    def iter_samples(self, file_path: str | Path) -> Iterator[list[float]]:
        self.bind(file_path)
        assert self._root is not None
        n_channels = self.metadata.n_channels
        frame = 4 * n_channels
        signal_path = self._root / SIGNAL_NAME
        with signal_path.open("rb") as handle:
            while True:
                blob = handle.read(frame * 256)
                if not blob:
                    break
                if len(blob) % frame != 0:
                    raise ValueError(f"{SIGNAL_NAME} is truncated")
                count = len(blob) // 4
                values = struct.unpack(f"<{count}f", blob)
                for offset in range(0, count, n_channels):
                    yield list(values[offset : offset + n_channels])

    def load_events(self, file_path: str | Path) -> list[TrialEvent]:
        """Turn marker LSL times into sample indices for playback."""
        self.bind(file_path)
        assert self._root is not None
        time_path = self._root / TIME_NAME
        timestamps: list[float] = []
        if time_path.is_file():
            raw = time_path.read_bytes()
            if len(raw) % 8 != 0:
                raise ValueError(f"{TIME_NAME} is truncated")
            if raw:
                timestamps = list(struct.unpack(f"<{len(raw) // 8}d", raw))
        markers_path = self._root / MARKERS_NAME
        if not markers_path.is_file():
            return []
        events: list[TrialEvent] = []
        for line in markers_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            events.append(
                TrialEvent(
                    sample_index=sample_index_for_time(timestamps, float(item["lsl_time"])),
                    code=str(item.get("code") or "event"),
                    label=str(item.get("label") or item.get("code") or "event"),
                    kind=str(item.get("kind") or "trial_start"),
                )
            )
        events.sort(key=lambda event: (event.sample_index, event.code, event.label))
        return events
