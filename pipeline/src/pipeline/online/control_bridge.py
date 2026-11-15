"""Live classification → versioned WebSocket control for the Phaser game.

Protocol v2 publishes normalized ``control`` ∈ [-1, 1] with trial binding.
Legacy ``left/right/timestamp`` samples remain available for compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from typing import Any, Sequence

import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from pipeline.epochs import is_activity_onset
from pipeline.live import parse_marker_text
from pipeline.online.control_decoder import (
    ControlDecoder,
    ControlDecoderConfig,
    ControlStrategy,
    signed_decision_margin,
)
from pipeline.online.emg_reject import EMGRejector, EmgRejectConfig
from pipeline.online.gate import DecisionGate, DecisionGateConfig
from pipeline.online.model import FrozenOnlineModel
from pipeline.online.observe import (
    build_pipeline_stage_message,
    build_stream_frame_message,
)
from pipeline.online.pipeline import OnlinePipeline
from pipeline.online.stream import OnlineStream
from pipeline.online.types import Direction, WindowDecision
from pipeline.online.windows import WindowManager

_LOG = logging.getLogger(__name__)
_HANDS = frozenset({"left", "right"})
PROTOCOL_VERSION = 2


def decision_to_control_sample(
    decision: WindowDecision,
    *,
    model: FrozenOnlineModel | None = None,
    trial_channels_by_time: np.ndarray | None = None,
    timestamp: float | None = None,
) -> dict[str, float] | None:
    """Legacy v1 sample ``{left,right,timestamp}`` (backward compatible)."""
    if decision.emg_rejected or decision.label not in _HANDS:
        return None
    stamp = float(timestamp if timestamp is not None else decision.end)
    if (
        model is not None
        and trial_channels_by_time is not None
        and decision.classifier
    ):
        pair = model.predict_proba_pair(
            trial_channels_by_time,
            classifier_name=decision.classifier,
        )
        if pair is not None:
            left, right = pair
            return {
                "left": float(left),
                "right": float(right),
                "timestamp": stamp,
            }
    score = float(decision.score or 0.0)
    conf = 1.0 / (1.0 + math.exp(-abs(score)))
    conf = min(0.99, max(0.51, conf))
    if decision.label == "left":
        return {"left": conf, "right": 1.0 - conf, "timestamp": stamp}
    return {"left": 1.0 - conf, "right": conf, "timestamp": stamp}


class ControlWsHub:
    """Broadcast JSON control messages to every connected game client."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8090) -> None:
        self.host = host
        self.port = int(port)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop: asyncio.Event | None = None
        self._clients: set[Any] = set()
        self._ready = threading.Event()
        self._error: str | None = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._ready.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name=f"ws-{self.port}", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise TimeoutError(f"Control WebSocket failed to bind on {self.url}")
        if self._error:
            raise RuntimeError(self._error)

    def stop(self) -> None:
        loop = self._loop
        stop = self._stop
        if loop is not None and stop is not None and loop.is_running():
            loop.call_soon_threadsafe(stop.set)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        self._loop = None
        self._stop = None
        self._clients.clear()

    def publish(self, message: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), loop)

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except OSError as exc:
            self._error = str(exc)
            self._ready.set()
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()

        async def handler(websocket: Any) -> None:
            self._clients.add(websocket)
            try:
                await websocket.wait_closed()
            except ConnectionClosed:
                pass
            finally:
                self._clients.discard(websocket)

        try:
            async with serve(handler, self.host, self.port):
                self._ready.set()
                assert self._stop is not None
                await self._stop.wait()
        except OSError as exc:
            self._error = str(exc)
            self._ready.set()
            return

        clients = list(self._clients)
        self._clients.clear()
        for client in clients:
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass

    async def _broadcast(self, message: dict[str, Any]) -> None:
        if not self._clients:
            return
        payload = json.dumps(message)
        dead: list[Any] = []
        for client in list(self._clients):
            try:
                await client.send(payload)
            except Exception:  # noqa: BLE001
                dead.append(client)
        for client in dead:
            self._clients.discard(client)


class LiveControlBridge:
    """Classify live EEG windows and publish trial-bound control commands."""

    def __init__(
        self,
        model: FrozenOnlineModel,
        *,
        dataset: str,
        hub: ControlWsHub,
        sample_rate_hz: float,
        channel_count: int,
        emg_threshold: float = 50.0,
        decoder_config: ControlDecoderConfig | None = None,
        emit_legacy_v1: bool = True,
        subject_id: str = "",
        log_path: str | None = None,
        observe_hub: ControlWsHub | None = None,
        channel_labels: Sequence[str] | None = None,
    ) -> None:
        if int(channel_count) != int(model.channel_count):
            raise ValueError(
                f"Live stream has {channel_count} channels, "
                f"frozen model expects {model.channel_count}"
            )
        self.model = model
        self.dataset = dataset
        self.hub = hub
        self.observe_hub = observe_hub
        self.subject_id = subject_id
        self.emit_legacy_v1 = bool(emit_legacy_v1)
        if channel_labels and len(channel_labels) == int(channel_count):
            self.channel_labels = [str(name) for name in channel_labels]
        else:
            self.channel_labels = [f"Ch{i + 1}" for i in range(int(channel_count))]
        self.stream = OnlineStream(
            sample_rate_hz=sample_rate_hz,
            n_eeg_channels=channel_count,
            n_emg_channels=2,
            filter_spec=model.filter,
            apply_causal_filter=True,
            keep_seconds=16.0,
        )
        self.windows = WindowManager(self.stream)
        self.pipeline = OnlinePipeline(
            model,
            EMGRejector(EmgRejectConfig(threshold=emg_threshold)),
        )
        self.gate = DecisionGate(DecisionGateConfig())
        classes = tuple(
            str(item) for item in getattr(model.head().classifier, "classes_", [])
        )
        if len(classes) == 2:
            cfg = decoder_config or ControlDecoderConfig()
            self.decoder = ControlDecoder(
                ControlDecoderConfig(
                    strategy=cfg.strategy,
                    window_k=cfg.window_k,
                    proportional_scale=cfg.proportional_scale,
                    class_order=(classes[0], classes[1]),
                )
            )
        else:
            self.decoder = ControlDecoder(decoder_config or ControlDecoderConfig())
        self._rest_start: float | None = None
        self._imagery_active = False
        self._lock = threading.Lock()
        self._published = 0
        self._sequence = 0
        self._trial_index = -1
        self._trial_id: str | None = None
        self._trial_cue: Direction | None = None
        self._last_sample: dict[str, Any] | None = None
        self._last_stage: dict[str, Any] | None = None
        self._last_stream_frame: dict[str, Any] | None = None
        self._samples_since_stream_frame = 0
        self._stream_frame_every = max(32, int(sample_rate_hz // 4))
        self._log_path = log_path
        self._log_lock = threading.Lock()

    @property
    def published_count(self) -> int:
        return self._published

    @property
    def last_stage(self) -> dict[str, Any] | None:
        return self._last_stage

    @property
    def last_stream_frame(self) -> dict[str, Any] | None:
        return self._last_stream_frame

    @property
    def last_sample(self) -> dict[str, Any] | None:
        return self._last_sample

    @property
    def strategy(self) -> ControlStrategy:
        return self.decoder.config.strategy

    def reset(self) -> None:
        with self._lock:
            self.pipeline.reset_session()
            self.gate.reset()
            self.windows.reset()
            self.decoder.reset_trial()
            self._rest_start = None
            self._imagery_active = False
            self._trial_id = None
            self._trial_cue = None

    def push_eeg(self, samples: np.ndarray, timestamps: list[float] | tuple[float, ...]) -> None:
        with self._lock:
            rows = np.asarray(samples, dtype=np.float64)
            self.stream.push_eeg(rows, timestamps)
            self._samples_since_stream_frame += int(rows.shape[0])
            if self._samples_since_stream_frame >= self._stream_frame_every:
                self._samples_since_stream_frame = 0
                self._publish_stream_frame()
            self._poll_windows()

    def on_marker(self, timestamp: float, kind: str, raw: str, label: str) -> None:
        with self._lock:
            if kind == "rest" or label == "rest":
                self._close_rest(timestamp)
                self._rest_start = timestamp
                return
            self._close_rest(timestamp)
            if kind in {"imagery_start", "imagery_end"}:
                if kind == "imagery_start" and label in _HANDS:
                    self._begin_imagery(timestamp, label)  # type: ignore[arg-type]
                elif kind == "imagery_end":
                    self._finish_imagery(force=True)
                return
            if not is_activity_onset(self.dataset, kind, label):
                return
            if label not in _HANDS:
                self._finish_imagery(force=True)
                return
            self._begin_imagery(timestamp, label)  # type: ignore[arg-type]

    def on_marker_text(self, timestamp: float, raw: str) -> None:
        kind, label = parse_marker_text(str(raw))
        self.on_marker(timestamp, kind, str(raw), label)

    def _close_rest(self, ended_at: float) -> None:
        if self._rest_start is None:
            return
        start = self._rest_start
        self._rest_start = None
        if ended_at <= start:
            return
        eeg, _ = self.stream.slice_eeg(start, ended_at, filtered=True)
        if eeg.shape[0] >= 16 and eeg.shape[1] == self.model.channel_count:
            self.pipeline.add_rest(eeg)

    def _begin_imagery(self, onset: float, cue: Direction) -> None:
        self._finish_imagery(force=True)
        self.gate.reset()
        self.decoder.reset_trial()
        self.windows.begin(onset)
        self._imagery_active = True
        self._trial_index += 1
        self._trial_cue = cue
        self._trial_id = f"trial_{self._trial_index:03d}"
        self._sequence = 0
        self.hub.publish(
            {
                "version": PROTOCOL_VERSION,
                "type": "trial_start",
                "trial_id": self._trial_id,
                "cue": cue,
                "subject_id": self.subject_id,
                "source_window_end": float(onset),
                "issued_at": time.time(),
            }
        )
        self._log(
            {
                "event": "trial_start",
                "trial_id": self._trial_id,
                "cue": cue,
                "subject_id": self.subject_id,
                "onset": float(onset),
            }
        )

    def _finish_imagery(self, *, force: bool) -> None:
        if not self._imagery_active:
            return
        self._poll_windows()
        if force or self.windows.all_complete():
            trial_id = self._trial_id
            self._imagery_active = False
            self.windows.reset()
            self.decoder.reset_trial()
            invalidate = {
                "version": PROTOCOL_VERSION,
                "type": "bci_control",
                "trial_id": trial_id,
                "sequence": self._sequence + 1,
                "source_window_end": None,
                "issued_at": time.time(),
                "control": 0.0,
                "valid": False,
                "reason": "trial_end",
            }
            self._sequence += 1
            self._last_sample = invalidate
            self.hub.publish(invalidate)
            if self.emit_legacy_v1:
                # Neutral legacy sample so v1 clients stop holding last bias.
                self.hub.publish(
                    {"left": 0.5, "right": 0.5, "timestamp": time.time(), "valid": False}
                )
            gate = self.gate.state()
            fused = gate.direction if gate.direction in _HANDS else None
            cue = self._trial_cue
            correct = (
                fused == cue if fused in _HANDS and cue in _HANDS else None
            )
            self.hub.publish(
                {
                    "version": PROTOCOL_VERSION,
                    "type": "trial_end",
                    "trial_id": trial_id,
                    "cue": cue,
                    "fused_label": fused,
                    "correct": correct,
                    "issued_at": time.time(),
                }
            )
            self._log(
                {
                    "event": "trial_end",
                    "trial_id": trial_id,
                    "cue": cue,
                    "fused_label": fused,
                    "correct": correct,
                    "gate_reason": gate.reason,
                }
            )
            self._trial_id = None
            self._trial_cue = None

    def _poll_windows(self) -> None:
        if not self._imagery_active or self._trial_id is None:
            return
        for window in self.windows.poll_ready():
            t0 = time.perf_counter()
            decision = self.pipeline.process_window(window)
            classify_ms = (time.perf_counter() - t0) * 1000.0
            self.gate.update(decision)
            aligned = self.pipeline.aligned_trial(window)
            margin = None
            p_left = p_right = None
            if aligned is not None and decision.classifier:
                head = self.model.head(decision.classifier)
                row = self.model.feature_vector(
                    aligned, classifier_name=decision.classifier
                )
                margin = signed_decision_margin(
                    head.classifier,
                    row,
                    class_order=self.decoder.config.class_order,
                )
                pair = self.model.predict_proba_pair(
                    aligned, classifier_name=decision.classifier
                )
                if pair is not None:
                    p_left, p_right = pair

            rejected = bool(
                decision.emg_rejected
                or decision.label not in _HANDS
                or decision.reason not in {"classified", ""}
            )
            decoded = self.decoder.decode(
                label=decision.label,
                score=decision.score,
                decision_margin=margin,
                p_left=p_left,
                p_right=p_right,
                rejected=rejected,
                reason=decision.reason,
            )
            self._sequence += 1
            message = {
                "version": PROTOCOL_VERSION,
                "type": "bci_control",
                "trial_id": self._trial_id,
                "sequence": self._sequence,
                "source_window_end": float(window.end),
                "issued_at": time.time(),
                "control": float(decoded.control) if decoded.valid else 0.0,
                "valid": bool(decoded.valid),
                "window_index": int(window.index),
                "strategy": decoded.strategy,
                "reason": decoded.reason,
            }
            self._last_sample = message
            self._published += 1
            self.hub.publish(message)
            self._publish_stage(
                window=window,
                decision=decision,
                aligned=aligned,
                margin=margin,
                p_left=p_left,
                p_right=p_right,
                decoded=decoded,
                classify_ms=classify_ms,
            )
            if self.emit_legacy_v1 and decoded.valid:
                # Map control ∈ [-1,1] back to left/right for old adapters.
                right = 0.5 + 0.5 * float(decoded.control)
                left = 1.0 - right
                self.hub.publish(
                    {
                        "left": left,
                        "right": right,
                        "timestamp": time.time(),
                        "trial_id": self._trial_id,
                        "valid": True,
                    }
                )
            elif self.emit_legacy_v1 and not decoded.valid:
                self.hub.publish(
                    {
                        "left": 0.5,
                        "right": 0.5,
                        "timestamp": time.time(),
                        "trial_id": self._trial_id,
                        "valid": False,
                    }
                )
            self._log(
                {
                    "event": "inference",
                    "subject_id": self.subject_id,
                    "trial_id": self._trial_id,
                    "cue": self._trial_cue,
                    "window_index": window.index,
                    "window_start": float(window.start),
                    "window_end": float(window.end),
                    "label": decision.label,
                    "score": decision.score,
                    "raw_margin": decoded.raw_margin,
                    "integrated_margin": decoded.integrated_margin,
                    "control": decoded.control,
                    "valid": decoded.valid,
                    "ea_ready": self.pipeline.ea_ready,
                    "emg_rejected": decision.emg_rejected,
                    "reason": decision.reason,
                    "strategy": decoded.strategy,
                    "classify_ms": classify_ms,
                }
            )

    def _publish_stream_frame(self) -> None:
        if self.observe_hub is None:
            return
        newest = self.stream.latest_time()
        if newest is None:
            return
        span = min(2.0, max(0.5, self.stream.keep_seconds / 6.0))
        start = newest - span
        raw, _ = self.stream.slice_eeg(start, newest + 1e-9, filtered=False)
        filtered, _ = self.stream.slice_eeg(start, newest + 1e-9, filtered=True)
        if raw.shape[0] < 8:
            return
        message = build_stream_frame_message(
            channels=self.channel_labels,
            raw=raw,
            filtered=filtered,
            sample_rate_hz=self.stream.sample_rate_hz,
        )
        self._last_stream_frame = message
        self.observe_hub.publish(message)

    def _publish_stage(
        self,
        *,
        window: Any,
        decision: WindowDecision,
        aligned: np.ndarray | None,
        margin: float | None,
        p_left: float | None,
        p_right: float | None,
        decoded: Any,
        classify_ms: float,
    ) -> None:
        if self.observe_hub is None or self._trial_id is None:
            return
        raw, _ = self.stream.slice_eeg(window.start, window.end, filtered=False)
        filtered = np.asarray(window.eeg, dtype=np.float64)
        if raw.shape[0] == 0:
            raw = filtered
        features = None
        if aligned is not None and decision.classifier:
            try:
                features = self.model.raw_features(aligned).reshape(-1)
            except Exception:  # noqa: BLE001
                features = None
        gate = self.gate.state()
        message = build_pipeline_stage_message(
            trial_id=self._trial_id,
            cue=self._trial_cue,
            window_index=int(window.index),
            window_start=float(window.start),
            window_end=float(window.end),
            channels=self.channel_labels,
            raw_window=raw,
            filtered_window=filtered,
            aligned_trial=aligned,
            csp_filters=self.model.csp_filters,
            features=features,
            p_left=p_left,
            p_right=p_right,
            label=decision.label,
            margin=margin,
            control=float(decoded.control) if decoded.valid else 0.0,
            valid=bool(decoded.valid),
            strategy=str(decoded.strategy),
            gate={
                "command": gate.command,
                "stable": gate.stable,
                "direction": gate.direction,
                "valid_count": gate.valid_count,
                "majority_count": gate.majority_count,
                "consecutive": gate.consecutive,
                "reason": gate.reason,
            },
            emg_rejected=bool(decision.emg_rejected),
            reason=str(decision.reason or decoded.reason or ""),
            classify_ms=classify_ms,
        )
        self._last_stage = message
        self.observe_hub.publish(message)

    def _log(self, row: dict[str, Any]) -> None:
        if not self._log_path:
            return
        line = json.dumps(row, ensure_ascii=True)
        with self._log_lock:
            with open(self._log_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
