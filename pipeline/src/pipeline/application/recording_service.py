"""Local EEG and EMG session recording: start, markers, stop, and saved sessions."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Callable

from pipeline.cyton import channel_labels_for, validate_cyton_stream
from pipeline.emg import validate_emg_stream
from pipeline.live import LiveSignal
from pipeline.models import StreamSummary
from pipeline.recorder import EmgRecorder, SessionRecorder, open_session_writer
from pipeline.session import SESSION_NAME, EmgWriter, list_recordings, load_recording

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

StreamOpener = Callable[..., Any]


class RecordingService:
    """One recording at a time. Shares the service lock and live signal."""

    def __init__(
        self,
        *,
        lock: threading.Lock,
        live: LiveSignal,
        recordings_root: Path,
        eeg_opener: StreamOpener,
        emg_opener: StreamOpener | None,
        clock: Callable[[], float],
    ) -> None:
        self._lock = lock
        self._live = live
        self.recordings_root = recordings_root
        self._eeg_opener = eeg_opener
        self._emg_opener = emg_opener
        self._clock = clock
        self._starting = False
        self._recorder: SessionRecorder | None = None
        self._emg_recorder: EmgRecorder | None = None
        self._session_id: str | None = None

    def recording_options(self) -> list[dict[str, str]]:
        options: list[dict[str, str]] = []
        for path in list_recordings(self.recordings_root):
            try:
                meta = json.loads((path / SESSION_NAME).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            session_id = str(meta.get("session_id") or path.name)
            written = str(meta.get("written_at") or "")
            when = written[:16].replace("T", " ")
            label = f"{when} · {session_id}" if when else session_id
            options.append(
                {
                    "id": f"recording:{session_id}",
                    "label": label,
                    "dataset": "recorded_session",
                    "sessionDir": str(path),
                }
            )
        return options

    def start(
        self,
        session_id: str,
        *,
        stream_name: str | None = None,
        wait_seconds: float = 5.0,
    ) -> dict[str, Any]:
        self._require_session_id(session_id)
        root = self.recordings_root / session_id
        with self._lock:
            if self._recorder is not None or self._starting:
                raise RuntimeError("A recording is already running")
            if root.exists():
                raise FileExistsError(f"Session already recorded: {session_id}")
            self._starting = True
        reader = None
        emg_reader = None
        try:
            reader = self._eeg_opener(stream_name=stream_name or None, wait_seconds=wait_seconds)
            summary: StreamSummary = reader.summary
            validate_cyton_stream(summary)
            if self._emg_opener is None:
                raise RuntimeError("EMG inlet is not configured")
            emg_reader = self._emg_opener(wait_seconds=wait_seconds)
            emg_summary: StreamSummary = emg_reader.summary
            validate_emg_stream(emg_summary)
            writer = open_session_writer(root, reader)
            emg_writer = EmgWriter(
                root,
                sample_rate_hz=float(emg_summary.nominal_srate),
                stream_name=emg_summary.name,
            )
            self._live.configure(
                summary.channel_count,
                summary.nominal_srate,
                channel_labels_for(summary),
            )
            recorder = SessionRecorder(
                reader,
                writer,
                clock=self._clock,
                on_samples=self._live.append,
            )
            emg_recorder = EmgRecorder(emg_reader, emg_writer)
            recorder.start()
            emg_recorder.start()
        except Exception:
            self._live.close()
            if emg_reader is not None:
                emg_reader.close()
            if reader is not None:
                reader.close()
            with self._lock:
                self._starting = False
            raise
        with self._lock:
            self._recorder = recorder
            self._emg_recorder = emg_recorder
            self._session_id = session_id
            self._starting = False
        return {
            "session_id": session_id,
            "path": str(root),
            "stream_name": summary.name,
            "channels": summary.channel_count,
            "sample_rate_hz": summary.nominal_srate,
            "emg_stream_name": emg_summary.name,
            "emg_channels": emg_summary.channel_count,
            "emg_sample_rate_hz": emg_summary.nominal_srate,
        }

    def add_marker(self, kind: str, code: str, label: str) -> dict[str, Any]:
        with self._lock:
            recorder = self._recorder
        if recorder is None:
            raise RuntimeError("No recording in progress")
        if not kind or not label:
            raise ValueError("marker kind and label are required")
        recorder.add_marker(kind, code or kind, label)
        return {"accepted": True}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            recorder = self._recorder
            emg_recorder = self._emg_recorder
            self._recorder = None
            self._emg_recorder = None
            self._session_id = None
        self._live.close()
        if recorder is None and emg_recorder is None:
            return {"stopped": False}
        emg_samples = emg_recorder.stop() if emg_recorder is not None else 0
        path = recorder.stop() if recorder is not None else None
        if path is None:
            return {"stopped": True, "emg_samples": emg_samples}
        recording = load_recording(path)
        return {
            "stopped": True,
            "path": str(path),
            "session_id": recording.session_id,
            "n_samples": recording.n_samples,
            "n_markers": len(recording.markers),
            "emg_samples": recording.emg_samples.shape[0],
        }

    def status_slice(self) -> dict[str, Any] | None:
        with self._lock:
            recorder = self._recorder
            emg_recorder = self._emg_recorder
            session_id = self._session_id
        if recorder is None or session_id is None:
            return None
        return {
            "session_id": session_id,
            "samples": recorder.writer.sample_count,
            "emg_samples": emg_recorder.writer.sample_count if emg_recorder is not None else 0,
            "path": str(recorder.writer.root),
        }

    @staticmethod
    def _require_session_id(session_id: str) -> None:
        if not SESSION_ID_RE.fullmatch(session_id or ""):
            raise ValueError(
                "session_id must contain only letters, digits, '_' and '-'"
            )
