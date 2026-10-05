"""Local session service the game calls to record or to start online playback.

The browser never opens LSL. This process does.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import numpy as np

from pipeline.cyton import channel_labels_for, validate_cyton_stream
from pipeline.live import LiveSignal, parse_marker_text
from pipeline.models import StreamSummary
from pipeline.recorder import SessionRecorder, open_session_writer
from pipeline.session import SESSION_NAME, list_recordings, load_recording

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
ProcessStarter = Callable[[list[str]], "subprocess.Popen[Any]"]
StreamOpener = Callable[..., Any]


def default_recordings_root() -> Path:
    return Path(__file__).resolve().parents[2] / "recordings"


def default_simulator_project() -> Path:
    env = os.environ.get("SIMULATOR_PROJECT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "simulator"


# One simulator process publishes one stream. The name is not a user setting.
SIMULATOR_LSL_NAME = "OpenBCI_EEG"
DatasetLister = Callable[[], list[dict[str, str]]]
SubjectLister = Callable[[str], list[dict[str, str]]]


def query_simulator_datasets(simulator_project: Path) -> list[dict[str, str]]:
    """Ask the simulator which datasets it can play."""
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(simulator_project),
            "simopenbci",
            "--list-datasets",
        ],
        cwd=simulator_project,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "simulator exited").strip()
        last = detail.splitlines()[-1] if detail else "Could not list datasets"
        raise RuntimeError(last)
    data = json.loads(completed.stdout or "[]")
    if not isinstance(data, list):
        raise RuntimeError("dataset list must be a JSON array")
    datasets: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        datasets.append({"id": str(item["id"]), "label": str(item.get("label") or item["id"])})
    return datasets


def query_simulator_subjects(simulator_project: Path, dataset: str) -> list[dict[str, str]]:
    """Ask the simulator which participants a dataset can play."""
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(simulator_project),
            "simopenbci",
            "--dataset",
            dataset,
            "--list-subjects",
            "--json",
        ],
        cwd=simulator_project,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "simulator exited").strip()
        last = detail.splitlines()[-1] if detail else "Could not list subjects"
        raise RuntimeError(last)
    data = json.loads(completed.stdout or "[]")
    if not isinstance(data, list):
        raise RuntimeError("subject list must be a JSON array")
    subjects: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        subjects.append({"id": str(item["id"]), "label": str(item.get("label") or item["id"])})
    return subjects


def lsl_clock() -> float:
    """LSL clock when pylsl is available, otherwise the wall clock."""
    try:
        from pylsl import local_clock

        return float(local_clock())
    except Exception:
        return time.time()


def open_marker_stream(*, eeg_name: str, wait_seconds: float) -> Any | None:
    """Markers inlet paired with the EEG stream, if one is published."""
    from pipeline.reader import LslStreamReader

    try:
        return LslStreamReader.open(
            name=f"{eeg_name}_Markers",
            wait_seconds=wait_seconds,
        )
    except TimeoutError:
        return None


def open_eeg_stream(*, stream_name: str | None, wait_seconds: float) -> Any:
    from pipeline.reader import LslStreamReader

    if stream_name:
        return LslStreamReader.open(name=stream_name, wait_seconds=wait_seconds)
    return LslStreamReader.open(stream_type="EEG", wait_seconds=wait_seconds)


def start_simulator_process(command: list[str]) -> subprocess.Popen[Any]:
    cwd = None
    if "--project" in command:
        cwd = command[command.index("--project") + 1]
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process.output_lines = []  # type: ignore[attr-defined]

    def _drain() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            lines: list[str] = process.output_lines  # type: ignore[attr-defined]
            lines.append(line.rstrip())
            del lines[:-30]

    threading.Thread(target=_drain, name="sim-output", daemon=True).start()
    return process


def build_simulator_command(
    *,
    simulator_project: Path,
    lsl_name: str,
    dataset: str | None = None,
    session_dir: str | None = None,
    file: str | None = None,
    subject: str | None = None,
    data_root: str | None = None,
    run: str | None = None,
    max_samples: int | None = None,
) -> list[str]:
    """argv for ``uv run simopenbci`` against a recorded session or another dataset."""
    if dataset == "recorded_session" and not (session_dir or file):
        raise ValueError("recorded_session requires session_dir")
    command = [
        "uv",
        "run",
        "--project",
        str(simulator_project),
        "simopenbci",
        "--no-dialog",
        "--lsl-name",
        lsl_name,
    ]
    if dataset:
        command.extend(["--dataset", dataset])
    target = session_dir or file
    if target:
        command.extend(["--file", target])
    if subject:
        command.extend(["--subject", subject])
    if data_root:
        command.extend(["--data-root", data_root])
    if run:
        command.extend(["--run", run])
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    return command


class PipelineService:
    """One recording and one online run at a time."""

    def __init__(
        self,
        recordings_root: Path,
        *,
        eeg_opener: StreamOpener | None = None,
        marker_opener: Callable[..., Any] | None = None,
        clock: Callable[[], float] | None = None,
        simulator_project: Path | None = None,
        process_starter: ProcessStarter | None = None,
        dataset_lister: DatasetLister | None = None,
        subject_lister: SubjectLister | None = None,
    ) -> None:
        self.recordings_root = Path(recordings_root)
        self.recordings_root.mkdir(parents=True, exist_ok=True)
        self._eeg_opener = eeg_opener or open_eeg_stream
        self._marker_opener = open_marker_stream if marker_opener is None and eeg_opener is None else marker_opener
        self._clock = clock or lsl_clock
        self._simulator_project = Path(simulator_project or default_simulator_project())
        self._process_starter = process_starter or start_simulator_process
        self._dataset_lister = dataset_lister
        self._subject_lister = subject_lister
        self._datasets: list[dict[str, str]] | None = None
        self._subjects: dict[str, list[dict[str, str]]] = {}
        self._lock = threading.Lock()
        self._starting = False
        self._recorder: SessionRecorder | None = None
        self._session_id: str | None = None
        self._proc: subprocess.Popen[Any] | None = None
        self._online_stop: threading.Event | None = None
        self._listener: threading.Thread | None = None
        self._online_samples = 0
        self._online_error: str | None = None
        self._online_source: str | None = None
        self._live = LiveSignal()

    def list_datasets(self) -> dict[str, Any]:
        if self._datasets is None:
            lister = self._dataset_lister or (
                lambda: query_simulator_datasets(self._simulator_project)
            )
            self._datasets = lister()
        datasets: list[dict[str, str]] = [
            {
                "id": item["id"],
                "label": item["label"],
                "dataset": item["id"],
                "sessionDir": "",
            }
            for item in self._datasets
            if item["id"] != "recorded_session"
        ]
        datasets.extend(self._recording_options())
        return {"datasets": datasets}

    def list_subjects(self, dataset: str) -> dict[str, Any]:
        if not dataset or dataset == "recorded_session":
            return {"subjects": []}
        cached = self._subjects.get(dataset)
        if cached is None:
            lister = self._subject_lister or (
                lambda dataset_id: query_simulator_subjects(self._simulator_project, dataset_id)
            )
            cached = lister(dataset)
            self._subjects[dataset] = cached
        return {"subjects": cached}

    def _recording_options(self) -> list[dict[str, str]]:
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

    def signal(self) -> dict[str, Any]:
        """Latest EEG window for the game session background."""
        return self._live.snapshot()

    def start_recording(
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
        try:
            reader = self._eeg_opener(stream_name=stream_name or None, wait_seconds=wait_seconds)
            summary: StreamSummary = reader.summary
            validate_cyton_stream(summary)
            writer = open_session_writer(root, reader)
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
            recorder.start()
        except Exception:
            self._live.close()
            if reader is not None:
                reader.close()
            with self._lock:
                self._starting = False
            raise
        with self._lock:
            self._recorder = recorder
            self._session_id = session_id
            self._starting = False
        return {
            "session_id": session_id,
            "path": str(root),
            "stream_name": summary.name,
            "channels": summary.channel_count,
            "sample_rate_hz": summary.nominal_srate,
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

    def stop_recording(self) -> dict[str, Any]:
        with self._lock:
            recorder = self._recorder
            self._recorder = None
            self._session_id = None
        self._live.close()
        if recorder is None:
            return {"stopped": False}
        path = recorder.stop()
        recording = load_recording(path)
        return {
            "stopped": True,
            "path": str(path),
            "session_id": recording.session_id,
            "n_samples": recording.n_samples,
            "n_markers": len(recording.markers),
        }

    def start_online(
        self,
        *,
        source: str,
        lsl_name: str,
        dataset: str | None = None,
        session_dir: str | None = None,
        file: str | None = None,
        subject: str | None = None,
        data_root: str | None = None,
        run: str | None = None,
        max_samples: int | None = None,
        wait_seconds: float = 5.0,
    ) -> dict[str, Any]:
        if source not in {"simulator", "cyton"}:
            raise ValueError("source must be 'simulator' or 'cyton'")
        command: list[str] | None = None
        if source == "simulator":
            command = build_simulator_command(
                simulator_project=self._simulator_project,
                lsl_name=lsl_name,
                dataset=dataset,
                session_dir=session_dir,
                file=file,
                subject=subject,
                data_root=data_root,
                run=run,
                max_samples=max_samples,
            )
        # A refresh or a finished playback can leave the previous run registered.
        # Start means "run this session", so the previous one is closed first.
        self.stop_online()
        with self._lock:
            self._online_samples = 0
            self._online_error = None
            self._online_source = source
            self._online_stop = threading.Event()
        if command is not None:
            proc = self._process_starter(command)
            with self._lock:
                self._proc = proc
        stream_name = lsl_name if source == "simulator" else (lsl_name or None)
        listener = threading.Thread(
            target=self._listen,
            args=(stream_name, source == "cyton", wait_seconds),
            name="online-inlet",
            daemon=True,
        )
        with self._lock:
            self._listener = listener
        listener.start()
        return {
            "source": source,
            "lsl_name": lsl_name,
            "command": command,
            "pid": self._proc.pid if self._proc is not None else None,
        }

    def stop_online(self) -> dict[str, Any]:
        with self._lock:
            proc = self._proc
            stop = self._online_stop
            listener = self._listener
            self._proc = None
            self._listener = None
            self._online_stop = None
        if stop is not None:
            stop.set()
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        if listener is not None:
            listener.join(timeout=2)
        self._live.close()
        with self._lock:
            samples = self._online_samples
            error = self._online_error
            source = self._online_source
            self._online_source = None
        if proc is None and listener is None:
            return {"stopped": False}
        return {"stopped": True, "source": source, "samples_seen": samples, "error": error}

    def status(self) -> dict[str, Any]:
        with self._lock:
            recorder = self._recorder
            session_id = self._session_id
            proc = self._proc
            samples = self._online_samples
            error = self._online_error
            source = self._online_source
        recording = None
        if recorder is not None and session_id is not None:
            recording = {
                "session_id": session_id,
                "samples": recorder.writer.sample_count,
                "path": str(recorder.writer.root),
            }
        online = None
        if source is not None:
            online = {
                "source": source,
                "pid": proc.pid if proc is not None else None,
                "running": proc.poll() is None if proc is not None else True,
                "samples_seen": samples,
                "error": error,
            }
        return {"recording": recording, "online": online}

    def _ingest_markers(self, reader: Any) -> None:
        chunk = reader.pull_chunk(timeout=0.0, max_samples=64)
        for sample, timestamp in zip(chunk.samples, chunk.timestamps):
            raw = sample[0] if sample else ""
            if not isinstance(raw, str):
                raw = str(raw)
            kind, label = parse_marker_text(raw)
            self._live.add_marker(float(timestamp), kind, label)

    def _listen(self, stream_name: str | None, require_cyton: bool, wait_seconds: float) -> None:
        reader = None
        marker_reader = None
        deadline = time.monotonic() + wait_seconds
        last_error = "No LSL stream"
        try:
            while self._online_stop is not None and not self._online_stop.is_set() and reader is None:
                proc = self._proc
                if proc is not None and proc.poll() is not None:
                    tail = getattr(proc, "output_lines", [])
                    detail = tail[-1] if tail else f"simulator exited ({proc.returncode})"
                    raise RuntimeError(detail)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(last_error)
                try:
                    reader = self._eeg_opener(
                        stream_name=stream_name,
                        wait_seconds=min(1.0, remaining),
                    )
                except TimeoutError as exc:
                    last_error = str(exc)
            if reader is None:
                return
            if require_cyton:
                validate_cyton_stream(reader.summary)
            self._live.configure(
                reader.summary.channel_count,
                reader.summary.nominal_srate,
                channel_labels_for(reader.summary),
            )
            if self._marker_opener is not None:
                try:
                    marker_reader = self._marker_opener(
                        eeg_name=reader.summary.name,
                        wait_seconds=min(1.0, wait_seconds),
                    )
                except (TimeoutError, OSError):
                    marker_reader = None
            while self._online_stop is not None and not self._online_stop.is_set():
                if marker_reader is not None:
                    self._ingest_markers(marker_reader)
                chunk = reader.pull_chunk(timeout=0.1, max_samples=2048)
                if chunk.samples:
                    self._live.append(
                        np.asarray(chunk.samples, dtype=np.float64),
                        chunk.timestamps,
                    )
                    with self._lock:
                        self._online_samples += len(chunk.samples)
                    continue
                proc = self._proc
                if proc is not None and proc.poll() is not None:
                    break
        except Exception as exc:  # noqa: BLE001 — reported on /status
            with self._lock:
                self._online_error = str(exc)
        finally:
            if marker_reader is not None:
                marker_reader.close()
            if reader is not None:
                reader.close()
            self._finish_listener()

    def _finish_listener(self) -> None:
        with self._lock:
            if self._listener is not threading.current_thread():
                return
            self._listener = None
            self._proc = None
            self._online_stop = None
            if self._online_error is None:
                self._online_source = None

    @staticmethod
    def _require_session_id(session_id: str) -> None:
        if not SESSION_ID_RE.fullmatch(session_id or ""):
            raise ValueError(
                "session_id must contain only letters, digits, '_' and '-'"
            )


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


def make_handler(service: PipelineService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[pipeline] {self.address_string()} {fmt % args}")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send(204, {})

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/health":
                self._send(200, {"ok": True})
                return
            if path == "/v1/status":
                self._send(200, service.status())
                return
            if path == "/v1/signal":
                self._send(200, service.signal())
                return
            if path == "/v1/datasets":
                try:
                    self._send(200, service.list_datasets())
                except (RuntimeError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
                    self._send(400, {"error": str(exc)})
                return
            if path == "/v1/subjects":
                dataset = (parse_qs(urlparse(self.path).query).get("dataset") or [""])[0]
                try:
                    self._send(200, service.list_subjects(dataset))
                except (RuntimeError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
                    self._send(400, {"error": str(exc)})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                body = _read_json(self)
                if path == "/v1/record/start":
                    payload = service.start_recording(
                        str(body.get("session_id") or ""),
                        stream_name=_blank_to_none(body.get("stream_name")),
                        wait_seconds=float(body.get("wait_seconds") or 5),
                    )
                elif path == "/v1/record/marker":
                    payload = service.add_marker(
                        str(body.get("kind") or ""),
                        str(body.get("code") or ""),
                        str(body.get("label") or ""),
                    )
                elif path == "/v1/record/stop":
                    payload = service.stop_recording()
                elif path == "/v1/online/start":
                    payload = service.start_online(
                        source=str(body.get("source") or "simulator"),
                        lsl_name=str(body.get("lsl_name") or "OpenBCI_EEG"),
                        dataset=_blank_to_none(body.get("dataset")),
                        session_dir=_blank_to_none(body.get("session_dir")),
                        file=_blank_to_none(body.get("file")),
                        subject=_blank_to_none(body.get("subject")),
                        data_root=_blank_to_none(body.get("data_root")),
                        run=_blank_to_none(body.get("run")),
                        max_samples=_optional_int(body.get("max_samples")),
                        wait_seconds=float(body.get("wait_seconds") or 5),
                    )
                elif path == "/v1/online/stop":
                    payload = service.stop_online()
                else:
                    self._send(404, {"error": "not found"})
                    return
            except (ValueError, FileExistsError, RuntimeError, TimeoutError, OSError) as exc:
                self._send(400, {"error": str(exc)})
                return
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid JSON"})
                return
            self._send(200, payload)

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _blank_to_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    recordings_root: Path | None = None,
) -> None:
    """Block and serve the session API."""
    root = Path(recordings_root) if recordings_root is not None else default_recordings_root()
    service = PipelineService(root)
    server = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"pipeline serve http://{host}:{port}  recordings={root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        service.stop_recording()
        service.stop_online()
        server.server_close()
