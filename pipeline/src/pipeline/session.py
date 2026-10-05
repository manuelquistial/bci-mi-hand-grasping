"""On-disk EEG session written by the pipeline.

Layout of ``<root>/``:

- ``session.json`` — board profile, channel labels, nominal rate
- ``signal.f32`` — little-endian float32, shape ``(n_samples, n_channels)``, µV
- ``time.f64`` — little-endian float64 LSL timestamp per sample
- ``markers.jsonl`` — one JSON object per marker (LSL time, kind, code, label)
- ``emg.f32`` — little-endian float32, shape ``(n_emg, 8)``, forearm electrodes
- ``emg_time.f64`` — LSL timestamp per EMG sample

The simulator does not write this folder. It only reads the EEG files.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from pipeline.cyton import device_metadata
from pipeline.emg import channel_labels as emg_channel_labels
from pipeline.emg import montage as emg_montage

FORMAT = "pipeline-session-v1"
SESSION_NAME = "session.json"
SIGNAL_NAME = "signal.f32"
TIME_NAME = "time.f64"
MARKERS_NAME = "markers.jsonl"
EMG_NAME = "emg.f32"
EMG_TIME_NAME = "emg_time.f64"


@dataclass(frozen=True)
class SessionMarker:
    lsl_time: float
    kind: str
    code: str
    label: str


@dataclass(frozen=True)
class Recording:
    root: Path
    session_id: str
    sample_rate_hz: float
    units: str
    channel_labels: tuple[str, ...]
    stream_name: str
    samples: np.ndarray
    timestamps: np.ndarray
    markers: tuple[SessionMarker, ...]
    device: dict[str, object]
    emg_samples: np.ndarray
    emg_timestamps: np.ndarray
    emg_sample_rate_hz: float
    emg_channel_labels: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.samples.shape[1]) if self.samples.ndim == 2 else 0


def sample_index_for_time(timestamps: Sequence[float] | np.ndarray, lsl_time: float) -> int:
    """Map an LSL time onto the nearest sample at or after that instant.

    An empty recording maps to index 0. A time past the last sample maps to
    the last sample so playback can still emit the marker.
    """
    times = np.asarray(timestamps, dtype=np.float64)
    if times.size == 0:
        return 0
    index = int(np.searchsorted(times, float(lsl_time), side="left"))
    if index >= times.size:
        return int(times.size - 1)
    return index


class SessionWriter:
    """Append-only writer for one session directory."""

    def __init__(
        self,
        root: Path,
        *,
        session_id: str,
        channel_labels: Sequence[str],
        sample_rate_hz: float,
        stream_name: str,
        units: str = "uV",
    ) -> None:
        self.root = Path(root)
        if self.root.exists():
            raise FileExistsError(f"Session directory already exists: {self.root}")
        self.root.mkdir(parents=True)
        self.session_id = session_id
        self.channel_labels = tuple(channel_labels)
        self.sample_rate_hz = float(sample_rate_hz)
        self.stream_name = stream_name
        self.units = units
        self._n_channels = len(self.channel_labels)
        if self._n_channels < 1:
            raise ValueError("channel_labels must not be empty")
        self._lock = threading.Lock()
        self._n_samples = 0
        self._closed = False
        self._signal = (self.root / SIGNAL_NAME).open("ab")
        self._time = (self.root / TIME_NAME).open("ab")
        self._markers = (self.root / MARKERS_NAME).open("a", encoding="utf-8")
        self._write_session_json()

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._n_samples

    def append_samples(self, samples: np.ndarray, timestamps: Sequence[float]) -> None:
        """Append a chunk of shape ``(n_samples, n_channels)`` and one time each."""
        array = np.asarray(samples, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self._n_channels:
            raise ValueError(
                f"expected shape (*, {self._n_channels}), got {array.shape}"
            )
        times = np.asarray(list(timestamps), dtype=np.float64)
        if times.shape != (array.shape[0],):
            raise ValueError(
                f"expected {array.shape[0]} timestamps, got {times.shape[0]}"
            )
        if array.shape[0] == 0:
            return
        signal = np.ascontiguousarray(array, dtype="<f4")
        clock = np.ascontiguousarray(times, dtype="<f8")
        with self._lock:
            if self._closed:
                raise RuntimeError("SessionWriter is closed")
            signal.tofile(self._signal)
            clock.tofile(self._time)
            self._signal.flush()
            self._time.flush()
            self._n_samples += int(array.shape[0])

    def append_marker(self, lsl_time: float, kind: str, code: str, label: str) -> None:
        payload = {
            "lsl_time": float(lsl_time),
            "kind": kind,
            "code": code,
            "label": label,
        }
        line = json.dumps(payload, separators=(",", ":"))
        with self._lock:
            if self._closed:
                raise RuntimeError("SessionWriter is closed")
            self._markers.write(line + "\n")
            self._markers.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._signal.close()
            self._time.close()
            self._markers.close()
            self._write_session_json()

    def _write_session_json(self) -> None:
        existing: dict[str, object] = {}
        meta_path = self.root / SESSION_NAME
        if meta_path.is_file():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                existing = loaded
        document = {
            "format": FORMAT,
            "session_id": self.session_id,
            "stream_name": self.stream_name,
            "sample_rate_hz": self.sample_rate_hz,
            "units": self.units,
            "channel_labels": list(self.channel_labels),
            "n_samples": self._n_samples,
            "device": device_metadata(),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        if "emg" in existing:
            document["emg"] = existing["emg"]
        (self.root / SESSION_NAME).write_text(
            json.dumps(document, indent=2) + "\n",
            encoding="utf-8",
        )


class EmgWriter:
    """Append-only EMG beside an EEG session. The EEG writer creates the folder."""

    def __init__(self, root: Path, *, sample_rate_hz: float, stream_name: str) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Session directory does not exist: {self.root}")
        self.sample_rate_hz = float(sample_rate_hz)
        self.stream_name = stream_name
        self.channel_labels = emg_channel_labels()
        self._n_channels = len(self.channel_labels)
        self._lock = threading.Lock()
        self._n_samples = 0
        self._closed = False
        self._signal = (self.root / EMG_NAME).open("ab")
        self._time = (self.root / EMG_TIME_NAME).open("ab")

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._n_samples

    def append_samples(self, samples: np.ndarray, timestamps: Sequence[float]) -> None:
        array = np.asarray(samples, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self._n_channels:
            raise ValueError(
                f"expected shape (*, {self._n_channels}), got {array.shape}"
            )
        times = np.asarray(list(timestamps), dtype=np.float64)
        if times.shape != (array.shape[0],):
            raise ValueError(
                f"expected {array.shape[0]} timestamps, got {times.shape[0]}"
            )
        if array.shape[0] == 0:
            return
        signal = np.ascontiguousarray(array, dtype="<f4")
        clock = np.ascontiguousarray(times, dtype="<f8")
        with self._lock:
            if self._closed:
                raise RuntimeError("EmgWriter is closed")
            signal.tofile(self._signal)
            clock.tofile(self._time)
            self._signal.flush()
            self._time.flush()
            self._n_samples += int(array.shape[0])

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._signal.close()
            self._time.close()
            n_samples = self._n_samples
        _write_emg_metadata(
            self.root,
            stream_name=self.stream_name,
            sample_rate_hz=self.sample_rate_hz,
            n_samples=n_samples,
        )


def _write_emg_metadata(
    root: Path,
    *,
    stream_name: str,
    sample_rate_hz: float,
    n_samples: int,
) -> None:
    meta_path = root / SESSION_NAME
    document = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    document["emg"] = {
        "stream_name": stream_name,
        "sample_rate_hz": sample_rate_hz,
        "channel_count": len(emg_channel_labels()),
        "channel_labels": list(emg_channel_labels()),
        "n_samples": n_samples,
        "placement": "forearm",
        "channels": emg_montage(),
    }
    meta_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _load_emg(path: Path, meta: dict[str, object]) -> tuple[np.ndarray, np.ndarray, float, tuple[str, ...]]:
    raw_emg = meta.get("emg")
    emg_meta: dict[str, object] = raw_emg if isinstance(raw_emg, dict) else {}
    label_values = emg_meta.get("channel_labels")
    labels = tuple(str(label) for label in label_values) if isinstance(label_values, list) else ()
    if not labels:
        labels = emg_channel_labels()
    n_channels = len(labels)
    rate_value = emg_meta.get("sample_rate_hz")
    rate = float(rate_value) if isinstance(rate_value, (int, float)) else 0.0
    signal_path = path / EMG_NAME
    time_path = path / EMG_TIME_NAME
    if not signal_path.is_file():
        return (
            np.zeros((0, n_channels), dtype=np.float32),
            np.zeros((0,), dtype=np.float64),
            rate,
            labels,
        )
    signal_bytes = signal_path.read_bytes()
    time_bytes = time_path.read_bytes() if time_path.is_file() else b""
    frame = 4 * n_channels
    if frame and len(signal_bytes) % frame != 0:
        raise ValueError(f"{EMG_NAME} is truncated")
    n_signal = len(signal_bytes) // frame if frame else 0
    if len(time_bytes) % 8 != 0:
        raise ValueError(f"{EMG_TIME_NAME} is truncated")
    n_time = len(time_bytes) // 8
    if n_signal != n_time:
        raise ValueError(
            f"sample count mismatch: {EMG_NAME}={n_signal} {EMG_TIME_NAME}={n_time}"
        )
    if n_signal:
        samples = np.frombuffer(signal_bytes, dtype="<f4").reshape(n_signal, n_channels).copy()
        timestamps = np.frombuffer(time_bytes, dtype="<f8").copy()
    else:
        samples = np.zeros((0, n_channels), dtype=np.float32)
        timestamps = np.zeros((0,), dtype=np.float64)
    return samples, timestamps, rate, labels


def load_recording(root: Path | str) -> Recording:
    """Read a session directory back into memory."""
    path = Path(root)
    meta_path = path / SESSION_NAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {SESSION_NAME} in {path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"Unsupported session format: {meta.get('format')!r}")

    labels = tuple(str(label) for label in meta["channel_labels"])
    n_channels = len(labels)
    if n_channels < 1:
        raise ValueError("session.json has no channel labels")

    signal_path = path / SIGNAL_NAME
    time_path = path / TIME_NAME
    signal_bytes = signal_path.read_bytes() if signal_path.is_file() else b""
    time_bytes = time_path.read_bytes() if time_path.is_file() else b""
    frame = 4 * n_channels
    if frame and len(signal_bytes) % frame != 0:
        raise ValueError(f"{SIGNAL_NAME} is truncated")
    n_signal = len(signal_bytes) // frame if frame else 0
    if len(time_bytes) % 8 != 0:
        raise ValueError(f"{TIME_NAME} is truncated")
    n_time = len(time_bytes) // 8
    if n_signal != n_time:
        raise ValueError(
            f"sample count mismatch: {SIGNAL_NAME}={n_signal} {TIME_NAME}={n_time}"
        )

    if n_signal:
        samples = np.frombuffer(signal_bytes, dtype="<f4").reshape(n_signal, n_channels).copy()
        timestamps = np.frombuffer(time_bytes, dtype="<f8").copy()
    else:
        samples = np.zeros((0, n_channels), dtype=np.float32)
        timestamps = np.zeros((0,), dtype=np.float64)

    markers: list[SessionMarker] = []
    markers_path = path / MARKERS_NAME
    if markers_path.is_file():
        for line in markers_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            markers.append(
                SessionMarker(
                    lsl_time=float(item["lsl_time"]),
                    kind=str(item["kind"]),
                    code=str(item["code"]),
                    label=str(item["label"]),
                )
            )

    device = dict(meta.get("device") or {})
    emg_samples, emg_timestamps, emg_rate, emg_labels = _load_emg(path, meta)
    return Recording(
        root=path,
        session_id=str(meta["session_id"]),
        sample_rate_hz=float(meta["sample_rate_hz"]),
        units=str(meta.get("units") or "uV"),
        channel_labels=labels,
        stream_name=str(meta.get("stream_name") or ""),
        samples=samples,
        timestamps=timestamps,
        markers=tuple(markers),
        device=device,
        emg_samples=emg_samples,
        emg_timestamps=emg_timestamps,
        emg_sample_rate_hz=emg_rate,
        emg_channel_labels=emg_labels,
    )


def list_recordings(root: Path | str) -> list[Path]:
    """Return session directories under ``root``, newest written file first."""
    base = Path(root)
    if not base.is_dir():
        return []
    found = [path for path in base.iterdir() if (path / SESSION_NAME).is_file()]
    return sorted(found, key=lambda path: (path / SESSION_NAME).stat().st_mtime, reverse=True)
