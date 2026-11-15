"""EMG side files of an on-disk session (writer and reader)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

from pipeline.emg import channel_labels as emg_channel_labels
from pipeline.emg import montage as emg_montage
from pipeline.session_format import EMG_NAME, EMG_TIME_NAME, SESSION_NAME


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
