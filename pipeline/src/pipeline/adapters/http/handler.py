"""BaseHTTPRequestHandler factory exposing :class:`PipelineService` as JSON."""

from __future__ import annotations

import json
import subprocess
from http.server import BaseHTTPRequestHandler
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from pipeline.adapters.http.parsing import (
    blank_to_none,
    number,
    optional_int,
    read_json,
)
from pipeline.config import CONTROL_WS_PORT
from pipeline.params import model_params

if TYPE_CHECKING:
    from pipeline.service import PipelineService


def make_handler(service: "PipelineService") -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def handle(self) -> None:
            try:
                super().handle()
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                # Browser / Next aborted a poll or closed keep-alive early.
                return

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
            if path == "/v1/online/models":
                query = parse_qs(urlparse(self.path).query)
                loso_raw = (query.get("loso") or ["0"])[0].lower()
                loso_only = loso_raw in {"1", "true", "yes"}
                try:
                    self._send(200, service.list_online_models(loso_only=loso_only))
                except (RuntimeError, OSError, json.JSONDecodeError) as exc:
                    self._send(400, {"error": str(exc)})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                body = read_json(self)
                if path == "/v1/record/start":
                    payload = service.start_recording(
                        str(body.get("session_id") or ""),
                        stream_name=blank_to_none(body.get("stream_name")),
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
                        dataset=blank_to_none(body.get("dataset")),
                        session_dir=blank_to_none(body.get("session_dir")),
                        file=blank_to_none(body.get("file")),
                        subject=blank_to_none(body.get("subject")),
                        data_root=blank_to_none(body.get("data_root")),
                        run=blank_to_none(body.get("run")),
                        max_samples=optional_int(body.get("max_samples")),
                        wait_seconds=float(body.get("wait_seconds") or 5),
                        registry=blank_to_none(body.get("registry")),
                        control_ws_port=int(body.get("control_ws_port") or CONTROL_WS_PORT),
                        control_strategy=str(
                            body.get("control_strategy") or "baseline_proba"
                        ),
                        control_window_k=int(body.get("control_window_k") or 3),
                        control_proportional_scale=float(
                            body.get("control_proportional_scale") or 1.0
                        ),
                    )
                elif path == "/v1/online/marker":
                    payload = service.mark_online(
                        str(body.get("kind") or ""),
                        str(body.get("code") or ""),
                        str(body.get("label") or ""),
                    )
                elif path == "/v1/online/stop":
                    payload = service.stop_online()
                elif path == "/v1/pre-model/start":
                    payload = service.start_online(
                        source="pre_model",
                        lsl_name="OpenBCI_EEG",
                        dataset=blank_to_none(body.get("dataset")),
                        session_dir=blank_to_none(body.get("session_dir")),
                        subject=blank_to_none(body.get("subject")),
                        wait_seconds=float(body.get("wait_seconds") or 5),
                    )
                elif path == "/v1/pre-model/stop":
                    payload = service.stop_online()
                elif path == "/v1/pre-model/filter":
                    payload = service.preview_filter(
                        dataset=str(body.get("dataset") or ""),
                        subject=blank_to_none(body.get("subject")),
                        session_dir=blank_to_none(body.get("session_dir")),
                        notch_hz=number(body.get("notch_hz"), 50),
                        band_low_hz=number(body.get("band_low_hz"), 8),
                        band_high_hz=number(body.get("band_high_hz"), 30),
                        order=int(number(body.get("order"), 4)),
                        spatial=str(body.get("spatial") or "none"),
                        alignment=str(body.get("alignment") or "none"),
                        features=str(body.get("features") or "none"),
                        classifier=str(body.get("classifier") or "none"),
                        evaluation=str(body.get("evaluation") or "cross_validation"),
                        params=model_params(body.get("hyperparameters")),
                    )
                else:
                    self._send(404, {"error": "not found"})
                    return
            except (
                ValueError,
                FileExistsError,
                RuntimeError,
                TimeoutError,
                OSError,
                subprocess.TimeoutExpired,
            ) as exc:
                self._send(400, {"error": str(exc)})
                return
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid JSON"})
                return
            self._send(200, payload)

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()
                self.wfile.write(body)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                return

    return Handler
