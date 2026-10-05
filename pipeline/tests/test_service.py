"""HTTP session service with a fake Cyton inlet (no pylsl)."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from pipeline.models import SampleChunk, StreamSummary
from pipeline.service import PipelineService, build_simulator_command, make_handler
from pipeline.session import load_recording


class _FakeReader:
    def __init__(self, chunks: list[SampleChunk], summary: StreamSummary) -> None:
        self.summary = summary
        self._chunks = list(chunks)
        self.closed = False

    def pull_chunk(self, timeout: float = 0.0, max_samples: int = 1024) -> SampleChunk:
        if self._chunks:
            return self._chunks.pop(0)
        time.sleep(min(timeout, 0.01))
        return SampleChunk(samples=[], timestamps=[])

    def close(self) -> None:
        self.closed = True


def _cyton_summary() -> StreamSummary:
    return StreamSummary(
        name="OpenBCI_EEG",
        stream_type="EEG",
        channel_count=8,
        nominal_srate=250.0,
        channel_format="float32",
        source_id="cyton",
        hostname="localhost",
        uid="uid",
        channel_labels=tuple(f"Ch{i + 1}" for i in range(8)),
    )


def _emg_summary() -> StreamSummary:
    return StreamSummary(
        name="Forearm_EMG",
        stream_type="EMG",
        channel_count=8,
        nominal_srate=1000.0,
        channel_format="float32",
        source_id="emg",
        hostname="localhost",
        uid="emg-uid",
        channel_labels=tuple(f"EMG{i + 1}" for i in range(8)),
    )


def _chunk() -> SampleChunk:
    rows = np.arange(16, dtype=np.float64).reshape(2, 8).tolist()
    return SampleChunk(samples=rows, timestamps=[1.0, 1.004])


class _Proc:
    def __init__(self) -> None:
        self.pid = 42
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = -9


def test_build_simulator_command_for_recorded_session(tmp_path: Path) -> None:
    command = build_simulator_command(
        simulator_project=tmp_path,
        lsl_name="OpenBCI_EEG",
        dataset="recorded_session",
        session_dir="/data/sess",
    )
    assert command[:4] == ["uv", "run", "--project", str(tmp_path)]
    assert "--dataset" in command and "recorded_session" in command
    assert "--file" in command and "/data/sess" in command
    assert "--no-dialog" in command


def test_record_start_stop_and_marker_over_http(tmp_path: Path) -> None:
    chunks = [_chunk(), _chunk()]

    def opener(*, stream_name: str | None, wait_seconds: float) -> _FakeReader:
        return _FakeReader(chunks, _cyton_summary())

    def emg_opener(*, wait_seconds: float) -> _FakeReader:
        return _FakeReader([_chunk(), _chunk()], _emg_summary())

    clock_values = iter([1.004, 1.5])

    def clock() -> float:
        return next(clock_values)

    service = PipelineService(
        tmp_path,
        eeg_opener=opener,
        emg_opener=emg_opener,
        clock=clock,
        process_starter=lambda command: _Proc(),  # type: ignore[return-value]
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        _post(base + "/v1/record/start", {"session_id": "lab1"})
        _post(
            base + "/v1/record/marker",
            {"kind": "trial_start", "code": "T1", "label": "left"},
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = json.loads(urllib.request.urlopen(base + "/v1/status").read())
            if status["recording"] and status["recording"]["samples"] >= 4:
                break
            time.sleep(0.02)
        signal = json.loads(urllib.request.urlopen(base + "/v1/signal").read())
        assert signal["active"] is True
        assert signal["channels"][0] == "Ch1"
        assert signal["samples"][0] == [0.0, 8.0, 0.0, 8.0]
        stopped = _post(base + "/v1/record/stop", {})
        idle = json.loads(urllib.request.urlopen(base + "/v1/signal").read())
        assert idle["active"] is False
        assert stopped["stopped"] is True
        assert stopped["n_samples"] == 4
        recording = load_recording(stopped["path"])
        assert recording.markers[0].label == "left"
        assert recording.markers[0].kind == "trial_start"
        assert recording.device["eeg_channels"] == 8
        assert stopped["emg_samples"] == 4
        assert recording.emg_samples.shape == (4, 8)
        assert recording.emg_channel_labels[0] == "left_forearm_anterior_1"
        assert recording.emg_channel_labels[4] == "right_forearm_anterior_1"
        assert recording.emg_sample_rate_hz == 1000.0
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_online_real_reads_the_forearm_emg(tmp_path: Path) -> None:
    service = PipelineService(
        tmp_path,
        eeg_opener=lambda **_: _FakeReader([_chunk()], _cyton_summary()),
        emg_opener=lambda **_: _FakeReader([_chunk()], _emg_summary()),
        process_starter=lambda command: _Proc(),  # type: ignore[return-value, arg-type]
    )
    service.start_online(source="cyton", lsl_name="OpenBCI_EEG", wait_seconds=0.2)
    deadline = time.monotonic() + 2
    seen = 0
    while time.monotonic() < deadline:
        online = service.status()["online"]
        seen = int(online["emg_samples_seen"]) if online else 0
        if seen >= 2:
            break
        time.sleep(0.02)
    service.stop_online()
    assert seen >= 2


def test_list_subjects_uses_the_injected_catalog(tmp_path: Path) -> None:
    service = PipelineService(
        tmp_path,
        subject_lister=lambda dataset: (
            [{"id": "sub-01", "label": "sub-01"}] if dataset == "liu2024_stroke" else []
        ),
    )
    assert service.list_subjects("liu2024_stroke")["subjects"] == [
        {"id": "sub-01", "label": "sub-01"}
    ]
    assert service.list_subjects("recorded_session")["subjects"] == []


def test_list_datasets_uses_the_injected_catalog(tmp_path: Path) -> None:
    service = PipelineService(
        tmp_path,
        dataset_lister=lambda: [{"id": "shuqfa2024", "label": "Shuqfa 2024 curated PhysioNet"}],
    )
    listed = service.list_datasets()
    assert listed["datasets"] == [
        {
            "id": "shuqfa2024",
            "label": "Shuqfa 2024 curated PhysioNet",
            "dataset": "shuqfa2024",
            "sessionDir": "",
        }
    ]


def test_list_datasets_includes_offline_recordings(tmp_path: Path) -> None:
    session = tmp_path / "lab1"
    session.mkdir()
    (session / "session.json").write_text(
        '{"session_id": "lab1", "written_at": "2026-10-04T23:40:00+00:00"}',
        encoding="utf-8",
    )
    service = PipelineService(
        tmp_path,
        dataset_lister=lambda: [
            {"id": "recorded_session", "label": "Recorded session"},
            {"id": "shuqfa2024", "label": "Shuqfa 2024 curated PhysioNet"},
        ],
    )
    listed = service.list_datasets()["datasets"]
    assert [item["id"] for item in listed] == ["shuqfa2024", "recording:lab1"]
    recording = listed[1]
    assert recording["dataset"] == "recorded_session"
    assert recording["sessionDir"] == str(session)
    assert "lab1" in recording["label"]


def test_pre_model_filter_returns_raw_and_filtered(tmp_path: Path) -> None:
    rate = 250.0
    times = np.arange(0, 8, 1.0 / rate)
    tone = np.sin(2 * np.pi * 12 * times) + 0.5 * np.sin(2 * np.pi * 50 * times)
    window = {
        "sample_rate_hz": rate,
        "channels": ["C3", "C4"],
        "samples": np.column_stack([tone, tone]).tolist(),
    }

    def loader(**_kwargs: object) -> dict:
        return window

    service = PipelineService(tmp_path, window_loader=loader)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        payload = _post(
            base + "/v1/pre-model/filter",
            {
                "dataset": "shuqfa2024",
                "subject": "S001",
                "notch_hz": 50,
                "band_low_hz": 8,
                "band_high_hz": 30,
                "order": 4,
            },
        )
        assert payload["subject"] == "S001"
        assert payload["channels"] == ["C3", "C4"]
        assert len(payload["raw"]) == 2
        assert len(payload["filtered"]) == 2
        assert len(payload["raw"][0]) == len(payload["filtered"][0])
        assert max(abs(value) for value in payload["filtered"][0]) < max(
            abs(value) for value in payload["raw"][0]
        )
    finally:
        server.shutdown()


def test_pre_model_start_is_its_own_source(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def starter(command: list[str]) -> _Proc:
        seen.append(command)
        return _Proc()

    service = PipelineService(
        tmp_path,
        eeg_opener=lambda **_: _FakeReader([_chunk()], _cyton_summary()),
        process_starter=starter,  # type: ignore[arg-type]
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        started = _post(
            base + "/v1/pre-model/start",
            {"dataset": "liu2024_stroke", "subject": "sub-05"},
        )
        assert started["source"] == "pre_model"
        assert "--dataset" in seen[0] and "liu2024_stroke" in seen[0]
        assert "--subject" in seen[0] and "sub-05" in seen[0]
        status = json.loads(urllib.request.urlopen(base + "/v1/status").read())
        assert status["online"]["source"] == "pre_model"
        stopped = _post(base + "/v1/pre-model/stop", {})
        assert stopped["stopped"] is True
        assert stopped["source"] == "pre_model"
    finally:
        server.shutdown()


def test_online_start_launches_simulator_command(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def starter(command: list[str]) -> _Proc:
        seen.append(command)
        return _Proc()

    def opener(*, stream_name: str | None, wait_seconds: float) -> _FakeReader:
        return _FakeReader([_chunk()], _cyton_summary())

    service = PipelineService(tmp_path, eeg_opener=opener, process_starter=starter)  # type: ignore[arg-type]
    started = service.start_online(
        source="simulator",
        lsl_name="OpenBCI_EEG",
        dataset="shuqfa2024",
        wait_seconds=0.1,
    )
    assert started["pid"] == 42
    assert "shuqfa2024" in seen[0]
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and service.status()["online"]["samples_seen"] < 2:
        time.sleep(0.02)
    assert service.signal()["active"] is True
    assert service.signal()["samples"][0] == [0.0, 8.0]
    stopped = service.stop_online()
    assert service.signal()["active"] is False
    assert stopped["stopped"] is True
    assert stopped["samples_seen"] >= 2


def test_online_signal_marks_where_each_trial_starts(tmp_path: Path) -> None:
    class _Markers:
        def __init__(self) -> None:
            self._chunks = [
                SampleChunk(samples=[["trial_start:T1:left"]], timestamps=[1.0]),
            ]
            self.closed = False

        def pull_chunk(self, timeout: float = 0.0, max_samples: int = 64) -> SampleChunk:
            if self._chunks:
                return self._chunks.pop(0)
            return SampleChunk(samples=[], timestamps=[])

        def close(self) -> None:
            self.closed = True

    markers = _Markers()
    service = PipelineService(
        tmp_path,
        eeg_opener=lambda **_: _FakeReader([_chunk()], _cyton_summary()),
        marker_opener=lambda **_: markers,
        process_starter=lambda command: _Proc(),  # type: ignore[return-value, arg-type]
    )
    service.start_online(source="simulator", lsl_name="OpenBCI_EEG", dataset="shuqfa2024")
    deadline = time.monotonic() + 2
    seen: list[dict] = []
    while time.monotonic() < deadline:
        seen = service.signal().get("markers") or []
        if seen:
            break
        time.sleep(0.02)
    service.stop_online()
    assert seen == [{"kind": "trial_start", "label": "left", "position": 0.0}]
    assert markers.closed is True


def test_online_start_replaces_a_previous_run(tmp_path: Path) -> None:
    launched: list[_Proc] = []

    def starter(command: list[str]) -> _Proc:
        proc = _Proc()
        proc.pid = 40 + len(launched)
        launched.append(proc)
        return proc

    def opener(*, stream_name: str | None, wait_seconds: float) -> _FakeReader:
        return _FakeReader([_chunk()], _cyton_summary())

    service = PipelineService(tmp_path, eeg_opener=opener, process_starter=starter)  # type: ignore[arg-type]
    first = service.start_online(source="simulator", lsl_name="OpenBCI_EEG", dataset="shuqfa2024")
    second = service.start_online(source="simulator", lsl_name="OpenBCI_EEG", dataset="shuqfa2024")
    assert first["pid"] == 40
    assert second["pid"] == 41
    assert launched[0].returncode == 0
    service.stop_online()


def test_finished_simulator_releases_the_online_slot(tmp_path: Path) -> None:
    proc = _Proc()

    class _DrainingReader(_FakeReader):
        def pull_chunk(self, timeout: float = 0.0, max_samples: int = 1024) -> SampleChunk:
            chunk = super().pull_chunk(timeout=timeout, max_samples=max_samples)
            if not chunk.samples:
                proc.returncode = 0
            return chunk

    def opener(*, stream_name: str | None, wait_seconds: float) -> _DrainingReader:
        return _DrainingReader([_chunk()], _cyton_summary())

    service = PipelineService(
        tmp_path,
        eeg_opener=opener,
        process_starter=lambda command: proc,  # type: ignore[return-value, arg-type]
    )
    service.start_online(source="simulator", lsl_name="OpenBCI_EEG", dataset="shuqfa2024")
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and service.status()["online"] is not None:
        time.sleep(0.02)
    assert service.status()["online"] is None


def test_online_real_windows_the_session_activity(tmp_path: Path) -> None:
    clock = {"now": 10.0}

    def tick() -> float:
        clock["now"] += 4.0
        return clock["now"]

    service = PipelineService(
        tmp_path,
        eeg_opener=lambda **_: _FakeReader([_chunk()], _cyton_summary()),
        emg_opener=lambda **_: _FakeReader([_chunk()], _emg_summary()),
        process_starter=lambda command: _Proc(),  # type: ignore[return-value, arg-type]
        clock=tick,
    )
    service.start_online(source="cyton", lsl_name="OpenBCI_EEG", wait_seconds=0.1)
    started = service.mark_online("imagery_start", "T1", "left")
    ended = service.mark_online("imagery_end", "T1", "left")
    service.stop_online()

    assert started["activity"]["started_at"] == 14.0
    assert started["activity"]["ended_at"] is None
    assert [(window["start"], window["end"]) for window in started["activity"]["windows"]] == [
        (14.0, 16.0),
        (14.5, 16.5),
        (15.0, 17.0),
        (15.5, 17.5),
        (16.0, 18.0),
    ]
    assert ended["activity"]["ended_at"] == 18.0
    assert service.status()["online"] is None


def test_online_simulated_windows_from_the_dataset_marker(tmp_path: Path) -> None:
    service = PipelineService(tmp_path, clock=lambda: 0.0)
    with service._lock:
        service._online_source = "simulator"
        service._online_dataset = "liu2024_stroke"

    class _Markers:
        def pull_chunk(self, timeout: float = 0.0, max_samples: int = 64) -> SampleChunk:
            return SampleChunk(
                samples=[["instruction:1:left"], ["trial_start:2:left"], ["rest:3:rest"]],
                timestamps=[8.0, 10.0, 14.0],
            )

    service._ingest_markers(_Markers())
    activity = service.status()["online"]["activity"]
    assert activity["label"] == "left"
    assert activity["code"] == "2"
    assert activity["started_at"] == 10.0
    assert activity["windows"][0]["start"] == 10.0
    assert activity["windows"][-1]["end"] == 14.0
    try:
        service.mark_online("imagery_start", "T1", "left")
    except RuntimeError as exc:
        assert "online real" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_online_marker_requires_a_running_session(tmp_path: Path) -> None:
    service = PipelineService(tmp_path, clock=lambda: 1.0)
    try:
        service.mark_online("imagery_start", "T2", "right")
    except RuntimeError as exc:
        assert "online real" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_recorded_session_requires_a_directory(tmp_path: Path) -> None:
    service = PipelineService(tmp_path)
    try:
        service.start_online(
            source="simulator",
            lsl_name="OpenBCI_EEG",
            dataset="recorded_session",
        )
    except ValueError as exc:
        assert "session_dir" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())
