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

    clock_values = iter([1.004, 1.5])

    def clock() -> float:
        return next(clock_values)

    service = PipelineService(
        tmp_path,
        eeg_opener=opener,
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
        stopped = _post(base + "/v1/record/stop", {})
        assert stopped["stopped"] is True
        assert stopped["n_samples"] == 4
        recording = load_recording(stopped["path"])
        assert recording.markers[0].label == "left"
        assert recording.markers[0].kind == "trial_start"
        assert recording.device["eeg_channels"] == 8
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_list_datasets_uses_the_injected_catalog(tmp_path: Path) -> None:
    service = PipelineService(
        tmp_path,
        dataset_lister=lambda: [{"id": "bci_iv2a", "label": "BCI Competition IV 2a"}],
    )
    listed = service.list_datasets()
    assert listed["datasets"] == [
        {
            "id": "bci_iv2a",
            "label": "BCI Competition IV 2a",
            "dataset": "bci_iv2a",
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
            {"id": "bci_iv2a", "label": "BCI Competition IV 2a"},
        ],
    )
    listed = service.list_datasets()["datasets"]
    assert [item["id"] for item in listed] == ["bci_iv2a", "recording:lab1"]
    recording = listed[1]
    assert recording["dataset"] == "recorded_session"
    assert recording["sessionDir"] == str(session)
    assert "lab1" in recording["label"]


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
        dataset="bci_iv2a",
        wait_seconds=0.1,
    )
    assert started["pid"] == 42
    assert "bci_iv2a" in seen[0]
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and service.status()["online"]["samples_seen"] < 2:
        time.sleep(0.02)
    stopped = service.stop_online()
    assert stopped["stopped"] is True
    assert stopped["samples_seen"] >= 2


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
    first = service.start_online(source="simulator", lsl_name="OpenBCI_EEG", dataset="bci_iv2a")
    second = service.start_online(source="simulator", lsl_name="OpenBCI_EEG", dataset="bci_iv2a")
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
    service.start_online(source="simulator", lsl_name="OpenBCI_EEG", dataset="bci_iv2a")
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and service.status()["online"] is not None:
        time.sleep(0.02)
    assert service.status()["online"] is None


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
