"""LSL ingest loop for an active online session."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from pipeline.cyton import channel_labels_for, validate_cyton_stream
from pipeline.emg import validate_emg_stream
from pipeline.epochs import is_activity_onset
from pipeline.live import parse_marker_text

if TYPE_CHECKING:
    from pipeline.application.online_session import OnlineRunManager


def run_ingest_loop(
    *,
    manager: OnlineRunManager,
    stream_name: str | None,
    require_cyton: bool,
    wait_seconds: float,
) -> None:
    reader = None
    marker_reader = None
    emg_reader = None
    deadline = time.monotonic() + wait_seconds
    last_error = "No LSL stream"
    try:
        while (
            manager._online_stop is not None
            and not manager._online_stop.is_set()
            and reader is None
        ):
            proc = manager._proc
            if proc is not None and proc.poll() is not None:
                tail = getattr(proc, "output_lines", [])
                detail = tail[-1] if tail else f"simulator exited ({proc.returncode})"
                raise RuntimeError(detail)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(last_error)
            try:
                reader = manager._eeg_opener(
                    stream_name=stream_name,
                    wait_seconds=min(1.0, remaining),
                )
            except TimeoutError as exc:
                last_error = str(exc)
        if reader is None:
            return
        if require_cyton:
            validate_cyton_stream(reader.summary)
            if manager._emg_opener is None:
                raise RuntimeError("EMG inlet is not configured")
            emg_reader = manager._emg_opener(wait_seconds=min(1.0, wait_seconds))
            validate_emg_stream(emg_reader.summary)
        manager._live.configure(
            reader.summary.channel_count,
            reader.summary.nominal_srate,
            channel_labels_for(reader.summary),
        )
        manager.maybe_start_control_bridge(reader.summary)
        if manager._marker_opener is not None:
            try:
                marker_reader = manager._marker_opener(
                    eeg_name=reader.summary.name,
                    wait_seconds=min(1.0, wait_seconds),
                )
            except (TimeoutError, OSError):
                marker_reader = None
        while manager._online_stop is not None and not manager._online_stop.is_set():
            if marker_reader is not None:
                ingest_markers(manager, marker_reader)
            if emg_reader is not None:
                _pull_emg(manager, emg_reader)
            chunk = reader.pull_chunk(timeout=0.1, max_samples=2048)
            if chunk.samples:
                samples = np.asarray(chunk.samples, dtype=np.float64)
                manager._live.append(samples, chunk.timestamps)
                manager._windower.push(samples, chunk.timestamps)
                with manager._lock:
                    bridge = manager._control_bridge
                    manager._online_samples += len(chunk.samples)
                if bridge is not None:
                    bridge.push_eeg(samples, chunk.timestamps)
                continue
            proc = manager._proc
            if proc is not None and proc.poll() is not None:
                break
    except Exception as exc:  # noqa: BLE001 — reported on /status
        with manager._lock:
            manager._online_error = str(exc)
    finally:
        if marker_reader is not None:
            marker_reader.close()
        if emg_reader is not None:
            emg_reader.close()
        if reader is not None:
            reader.close()
        manager.finish_listener()


def ingest_markers(manager: OnlineRunManager, reader: Any) -> None:
    chunk = reader.pull_chunk(timeout=0.0, max_samples=64)
    for sample, timestamp in zip(chunk.samples, chunk.timestamps):
        raw = sample[0] if sample else ""
        if not isinstance(raw, str):
            raw = str(raw)
        kind, label = parse_marker_text(raw)
        manager._live.add_marker(float(timestamp), kind, label)
        _window_dataset_activity(manager, float(timestamp), kind, raw, label)
        with manager._lock:
            bridge = manager._control_bridge
        if bridge is not None:
            bridge.on_marker(float(timestamp), kind, raw, label)


def _window_dataset_activity(
    manager: OnlineRunManager, timestamp: float, kind: str, raw: str, label: str
) -> None:
    with manager._lock:
        source = manager._online_source
        dataset = manager._online_dataset or ""
    if source != "simulator" or not is_activity_onset(dataset, kind, label):
        return
    parts = (raw or "").strip().split(":")
    code = parts[1] if len(parts) >= 3 and parts[1] else kind
    manager._windower.begin(timestamp, label, code)


def _pull_emg(manager: OnlineRunManager, reader: Any) -> None:
    chunk = reader.pull_chunk(timeout=0.0, max_samples=2048)
    if not chunk.samples:
        return
    with manager._lock:
        manager._emg_samples += len(chunk.samples)
