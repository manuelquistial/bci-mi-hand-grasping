"""Online session use-case: start/stop hubs, listener, markers, status slice."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

from pipeline.adapters.simulator.process import build_simulator_command
from pipeline.application.control_lifecycle import (
    build_status_slice,
    maybe_start_control_bridge,
)
from pipeline.application.ingest_loop import run_ingest_loop
from pipeline.application.online_hubs import resolve_observe_port, start_hub_pair
from pipeline.application.online_types import OnlineStartRequest
from pipeline.config import CONTROL_WS_PORT, OBSERVE_WS_PORT
from pipeline.live import LiveSignal
from pipeline.models import StreamSummary
from pipeline.windows import ActivityWindower

__all__ = ["OnlineRunManager", "OnlineStartRequest"]

ProcessStarter = Callable[[list[str]], subprocess.Popen[Any]]
StreamOpener = Callable[..., Any]


class OnlineRunManager:
    """Owns online-run mutable state; PipelineService delegates here."""

    def __init__(
        self,
        *,
        lock: threading.Lock,
        live: LiveSignal,
        windower: ActivityWindower,
        recordings_root: Path,
        simulator_project: Path,
        eeg_opener: StreamOpener,
        emg_opener: StreamOpener | None,
        marker_opener: StreamOpener | None,
        process_starter: ProcessStarter,
        clock: Callable[[], float],
    ) -> None:
        self._lock = lock
        self._live = live
        self._windower = windower
        self.recordings_root = Path(recordings_root)
        self._simulator_project = Path(simulator_project)
        self._eeg_opener = eeg_opener
        self._emg_opener = emg_opener
        self._marker_opener = marker_opener
        self._process_starter = process_starter
        self._clock = clock
        self._proc: subprocess.Popen[Any] | None = None
        self._online_stop: threading.Event | None = None
        self._listener: threading.Thread | None = None
        self._online_samples = 0
        self._emg_samples = 0
        self._online_error: str | None = None
        self._online_source: str | None = None
        self._online_dataset: str | None = None
        self._online_subject: str | None = None
        self._control_registry: str | None = None
        self._control_ws_port = CONTROL_WS_PORT
        self._observe_ws_port = OBSERVE_WS_PORT
        self._frozen_model: Any | None = None
        self._control_hub: Any | None = None
        self._observe_hub: Any | None = None
        self._control_bridge: Any | None = None
        self._control_decoder_config: Any | None = None

    def start(self, request: OnlineStartRequest) -> dict[str, Any]:
        if request.source not in {"simulator", "cyton", "pre_model"}:
            raise ValueError("source must be 'simulator', 'cyton', or 'pre_model'")
        command: list[str] | None = None
        if request.source in {"simulator", "pre_model"}:
            command = build_simulator_command(
                simulator_project=self._simulator_project,
                lsl_name=request.lsl_name,
                dataset=request.dataset,
                session_dir=request.session_dir,
                file=request.file,
                subject=request.subject,
                data_root=request.data_root,
                run=request.run,
                max_samples=request.max_samples,
            )
        self.stop()
        frozen = None
        hub = None
        observe_hub = None
        control_port = int(request.control_ws_port)
        observe_port = resolve_observe_port(control_port, request.observe_ws_port)
        if request.registry:
            from pipeline.online.control_decoder import ControlDecoderConfig
            from pipeline.online.registry import OnlineModelRegistry

            frozen = OnlineModelRegistry().load_model(request.registry)
            hubs = start_hub_pair(
                control_port=control_port, observe_port=observe_port
            )
            hub = hubs.control
            observe_hub = hubs.observe
            decoder = ControlDecoderConfig(
                strategy=request.control_strategy,  # type: ignore[arg-type]
                window_k=int(request.control_window_k),
                proportional_scale=float(request.control_proportional_scale),
            )
        else:
            decoder = None
        with self._lock:
            self._online_samples = 0
            self._emg_samples = 0
            self._online_error = None
            self._online_source = request.source
            self._online_dataset = request.dataset
            self._online_subject = request.subject
            self._control_registry = request.registry
            self._control_ws_port = control_port
            self._observe_ws_port = observe_port
            self._frozen_model = frozen
            self._control_hub = hub
            self._observe_hub = observe_hub
            self._control_bridge = None
            self._control_decoder_config = decoder
            self._windower.reset()
            self._online_stop = threading.Event()
        if command is not None:
            proc = self._process_starter(command)
            with self._lock:
                self._proc = proc
        stream_name = (
            request.lsl_name
            if request.source in {"simulator", "pre_model"}
            else (request.lsl_name or None)
        )
        listener = threading.Thread(
            target=self._listen,
            args=(stream_name, request.source == "cyton", request.wait_seconds),
            name="online-inlet",
            daemon=True,
        )
        with self._lock:
            self._listener = listener
        listener.start()
        return {
            "source": request.source,
            "lsl_name": request.lsl_name,
            "command": command,
            "pid": self._proc.pid if self._proc is not None else None,
            "subject": request.subject,
            "registry": request.registry,
            "control_ws_url": hub.url if hub is not None else None,
            "observe_ws_url": observe_hub.url if observe_hub is not None else None,
            "control_classifier": (
                None if frozen is None else frozen.control_name
            ),
            "control_strategy": request.control_strategy,
            "protocol_version": 2,
        }

    def stop(self) -> dict[str, Any]:
        with self._lock:
            proc = self._proc
            stop = self._online_stop
            listener = self._listener
            hub = self._control_hub
            observe_hub = self._observe_hub
            bridge = self._control_bridge
            self._proc = None
            self._listener = None
            self._online_stop = None
            self._control_hub = None
            self._observe_hub = None
            self._control_bridge = None
            self._frozen_model = None
            self._control_registry = None
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
        if hub is not None:
            hub.stop()
        if observe_hub is not None:
            observe_hub.stop()
        self._live.close()
        with self._lock:
            samples = self._online_samples
            error = self._online_error
            source = self._online_source
            published = bridge.published_count if bridge is not None else 0
            self._online_source = None
            self._online_dataset = None
            self._windower.reset()
        if proc is None and listener is None and hub is None and observe_hub is None:
            return {"stopped": False}
        return {
            "stopped": True,
            "source": source,
            "samples_seen": samples,
            "control_samples": published,
            "error": error,
        }

    def mark(self, kind: str, code: str, label: str) -> dict[str, Any]:
        if kind not in {"imagery_start", "imagery_end"}:
            raise ValueError("marker kind must be imagery_start or imagery_end")
        if not label:
            raise ValueError("marker label is required")
        now = float(self._clock())
        with self._lock:
            if self._online_source != "cyton":
                raise RuntimeError("No online real session in progress")
            bridge = self._control_bridge
        if kind == "imagery_start":
            current = self._windower.begin(now, label, code or kind)
        else:
            current = self._windower.close_open(now)
            if current is None:
                raise RuntimeError("Motor imagery has not started")
        if bridge is not None:
            bridge.on_marker(now, kind, f"{kind}:{code or kind}:{label}", label)
        return {"accepted": True, "activity": current}

    def status_slice(self) -> dict[str, Any] | None:
        return build_status_slice(self)

    def maybe_start_control_bridge(self, summary: StreamSummary) -> None:
        maybe_start_control_bridge(self, summary)

    def _listen(
        self, stream_name: str | None, require_cyton: bool, wait_seconds: float
    ) -> None:
        run_ingest_loop(
            manager=self,
            stream_name=stream_name,
            require_cyton=require_cyton,
            wait_seconds=wait_seconds,
        )

    def finish_listener(self) -> None:
        with self._lock:
            if self._listener is not threading.current_thread():
                return
            self._listener = None
            self._proc = None
            self._online_stop = None
            if self._online_error is None:
                self._online_source = None
