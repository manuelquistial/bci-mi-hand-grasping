"""LSL-backed (or injected) buffer for live EEG and EMG."""

from __future__ import annotations

import threading
from typing import Any, Callable, Sequence

import numpy as np

from pipeline.online.causal_filter import CausalNotchBandpass
from pipeline.preprocess import FilterSpec

EegOpener = Callable[..., Any]
EmgOpener = Callable[..., Any]


class OnlineStream:
    """Buffers incoming EEG/EMG samples and optionally filters EEG causally.

    When ``eeg_reader`` / ``emg_reader`` are omitted, callers push chunks with
    :meth:`push_eeg` / :meth:`push_emg` (tests and recorded playback).
    """

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        n_eeg_channels: int,
        n_emg_channels: int = 2,
        filter_spec: FilterSpec | None = None,
        keep_seconds: float = 12.0,
        eeg_opener: EegOpener | None = None,
        emg_opener: EmgOpener | None = None,
        apply_causal_filter: bool = True,
    ) -> None:
        self.sample_rate_hz = float(sample_rate_hz)
        self.n_eeg_channels = int(n_eeg_channels)
        self.n_emg_channels = int(n_emg_channels)
        self.keep_seconds = float(keep_seconds)
        self._eeg_opener = eeg_opener
        self._emg_opener = emg_opener
        self._lock = threading.Lock()
        self._eeg_times: list[float] = []
        self._eeg_raw: list[np.ndarray] = []
        self._eeg_filtered: list[np.ndarray] = []
        self._emg_times: list[float] = []
        self._emg_raw: list[np.ndarray] = []
        self._eeg_reader: Any | None = None
        self._emg_reader: Any | None = None
        self._filter: CausalNotchBandpass | None = None
        if apply_causal_filter:
            self._filter = CausalNotchBandpass(self.sample_rate_hz, filter_spec)

    def open(self, *, eeg_name: str | None = None, wait_seconds: float = 5.0) -> None:
        """Open LSL inlets when openers were provided at construction."""
        if self._eeg_opener is None:
            raise RuntimeError("No EEG opener configured")
        self._eeg_reader = self._eeg_opener(
            stream_name=eeg_name,
            wait_seconds=wait_seconds,
        )
        summary = self._eeg_reader.summary
        self.sample_rate_hz = float(summary.nominal_srate or self.sample_rate_hz)
        self.n_eeg_channels = int(summary.channel_count)
        if self._filter is not None:
            self._filter = CausalNotchBandpass(self.sample_rate_hz, self._filter.spec)
        if self._emg_opener is not None:
            self._emg_reader = self._emg_opener(wait_seconds=min(1.0, wait_seconds))
            self.n_emg_channels = int(self._emg_reader.summary.channel_count)

    def close(self) -> None:
        if self._eeg_reader is not None:
            self._eeg_reader.close()
            self._eeg_reader = None
        if self._emg_reader is not None:
            self._emg_reader.close()
            self._emg_reader = None

    def poll(self, timeout: float = 0.05) -> int:
        """Pull available LSL chunks into the buffers. Returns EEG samples added."""
        added = 0
        if self._eeg_reader is not None:
            chunk = self._eeg_reader.pull_chunk(timeout=timeout, max_samples=2048)
            if chunk.samples:
                self.push_eeg(np.asarray(chunk.samples, dtype=np.float64), chunk.timestamps)
                added = len(chunk.samples)
        if self._emg_reader is not None:
            chunk = self._emg_reader.pull_chunk(timeout=0.0, max_samples=2048)
            if chunk.samples:
                self.push_emg(np.asarray(chunk.samples, dtype=np.float64), chunk.timestamps)
        return added

    def push_eeg(self, samples: np.ndarray, timestamps: Sequence[float]) -> None:
        rows = np.asarray(samples, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        times = [float(value) for value in timestamps]
        if len(times) != rows.shape[0]:
            raise ValueError("EEG timestamps must match the sample count")
        if rows.shape[1] != self.n_eeg_channels:
            raise ValueError(
                f"Expected {self.n_eeg_channels} EEG channels, got {rows.shape[1]}"
            )
        filtered = self._filter.process(rows) if self._filter is not None else rows
        with self._lock:
            self._eeg_raw.extend(rows)
            self._eeg_filtered.extend(filtered)
            self._eeg_times.extend(times)
            self._trim()

    def push_emg(self, samples: np.ndarray, timestamps: Sequence[float]) -> None:
        rows = np.asarray(samples, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        times = [float(value) for value in timestamps]
        if len(times) != rows.shape[0]:
            raise ValueError("EMG timestamps must match the sample count")
        if rows.shape[1] < 1:
            raise ValueError("EMG needs at least one channel")
        with self._lock:
            self._emg_raw.extend(rows)
            self._emg_times.extend(times)
            self._trim()

    def slice_eeg(
        self,
        start: float,
        end: float,
        *,
        filtered: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(samples, times)`` with shape ``(n, channels)`` in ``[start, end)``."""
        with self._lock:
            source = self._eeg_filtered if filtered else self._eeg_raw
            return _slice(source, self._eeg_times, start, end)

    def slice_emg(self, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return _slice(self._emg_raw, self._emg_times, start, end)

    def latest_time(self) -> float | None:
        with self._lock:
            if not self._eeg_times:
                return None
            return self._eeg_times[-1]

    def reset_buffers(self) -> None:
        with self._lock:
            self._eeg_times = []
            self._eeg_raw = []
            self._eeg_filtered = []
            self._emg_times = []
            self._emg_raw = []
            if self._filter is not None:
                self._filter.reset(self.n_eeg_channels)

    def _trim(self) -> None:
        if not self._eeg_times and not self._emg_times:
            return
        newest = max(
            self._eeg_times[-1] if self._eeg_times else float("-inf"),
            self._emg_times[-1] if self._emg_times else float("-inf"),
        )
        cutoff = newest - self.keep_seconds
        _drop_before(self._eeg_times, self._eeg_raw, self._eeg_filtered, cutoff)
        _drop_before(self._emg_times, self._emg_raw, None, cutoff)


def _slice(
    rows: list[np.ndarray],
    times: list[float],
    start: float,
    end: float,
) -> tuple[np.ndarray, np.ndarray]:
    chosen = [
        (row, stamp)
        for row, stamp in zip(rows, times)
        if start <= stamp < end
    ]
    if not chosen:
        width = int(rows[0].shape[0]) if rows else 0
        return np.zeros((0, width), dtype=np.float64), np.zeros(0, dtype=np.float64)
    samples = np.vstack([item[0] for item in chosen])
    stamps = np.asarray([item[1] for item in chosen], dtype=np.float64)
    return samples, stamps


def _drop_before(
    times: list[float],
    primary: list[np.ndarray],
    secondary: list[np.ndarray] | None,
    cutoff: float,
) -> None:
    drop = 0
    while drop < len(times) and times[drop] < cutoff:
        drop += 1
    if not drop:
        return
    del times[:drop]
    del primary[:drop]
    if secondary is not None:
        del secondary[:drop]
