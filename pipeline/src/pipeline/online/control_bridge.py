"""Live classification → versioned WebSocket control for the Phaser game.

Protocol v2 publishes normalized ``control`` ∈ [-1, 1] with trial binding.
Legacy ``left/right/timestamp`` samples remain available for compatibility.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Sequence

import numpy as np

from pipeline.adapters.ws.hub import ControlWsHub, JsonWsHub
from pipeline.config import CONTROL_WS_PORT
from pipeline.epochs import is_activity_onset
from pipeline.live import parse_marker_text
from pipeline.online.control_decoder import (
    ControlDecoderConfig,
    ControlStrategy,
)
from pipeline.online.control_inference import build_decoder
from pipeline.online.control_legacy import (
    decision_to_control_sample,
    legacy_neutral_sample,
)
from pipeline.online.control_messages import (
    PROTOCOL_VERSION,
    invalidate_message,
    trial_end_log,
    trial_end_message,
    trial_outcome,
    trial_start_log,
    trial_start_message,
)
from pipeline.online.control_log import JsonlEventLog
from pipeline.online.control_observe import (
    ObservePublisher,
    RecentMarkers,
    resolve_channel_labels,
)
from pipeline.online.control_poll import poll_and_publish
from pipeline.online.emg_reject import EMGRejector, EmgRejectConfig
from pipeline.online.gate import DecisionGate, DecisionGateConfig
from pipeline.online.model import FrozenOnlineModel
from pipeline.online.pipeline import OnlinePipeline
from pipeline.online.stream import OnlineStream
from pipeline.online.types import Direction
from pipeline.online.windows import WindowManager

_LOG = logging.getLogger(__name__)
_HANDS = frozenset({"left", "right"})

__all__ = [
    "CONTROL_WS_PORT",
    "ControlWsHub",
    "JsonWsHub",
    "LiveControlBridge",
    "PROTOCOL_VERSION",
    "decision_to_control_sample",
]


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
        self.channel_labels = resolve_channel_labels(channel_labels, channel_count)
        self._observe = ObservePublisher(observe_hub, self.channel_labels)
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
        self.decoder = build_decoder(model, decoder_config)
        self._rest_start: float | None = None
        self._imagery_active = False
        self._lock = threading.Lock()
        self._published = 0
        self._sequence = 0
        self._trial_index = -1
        self._trial_id: str | None = None
        self._trial_cue: Direction | None = None
        self._last_sample: dict[str, Any] | None = None
        self._samples_since_stream_frame = 0
        self._stream_frame_every = max(32, int(sample_rate_hz // 4))
        self._markers = RecentMarkers()
        self._events = JsonlEventLog(log_path)

    @property
    def published_count(self) -> int:
        return self._published

    @property
    def last_stage(self) -> dict[str, Any] | None:
        return self._observe.last_stage

    @property
    def last_stream_frame(self) -> dict[str, Any] | None:
        return self._observe.last_stream_frame

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

    def _flush_observe(
        self,
        *,
        stream_meta: dict[str, Any] | None = None,
        stages: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Publish observe traffic outside ``_lock`` so control never waits on it."""
        if stream_meta is not None:
            self._observe.publish_stream_frame(
                self.stream,
                markers=stream_meta["markers"],
                windows=stream_meta["windows"],
                imagery_onset=stream_meta["imagery_onset"],
            )
        for job in stages or ():
            trial_id = job.pop("trial_id", None)
            self._observe.publish_stage(trial_id, **job)

    def push_eeg(self, samples: np.ndarray, timestamps: list[float] | tuple[float, ...]) -> None:
        stream_meta: dict[str, Any] | None = None
        stages: list[dict[str, Any]] = []
        with self._lock:
            rows = np.asarray(samples, dtype=np.float64)
            self.stream.push_eeg(rows, timestamps)
            self._samples_since_stream_frame += int(rows.shape[0])
            if self._samples_since_stream_frame >= self._stream_frame_every:
                self._samples_since_stream_frame = 0
                onset = self.windows.onset if self._imagery_active else None
                spans = self.windows.spans() if self._imagery_active else []
                stream_meta = {
                    "markers": self._markers.items,
                    "windows": [
                        {"index": i, "start": s, "end": e}
                        for i, (s, e) in enumerate(spans)
                    ],
                    "imagery_onset": onset,
                }
            stages = poll_and_publish(self)
        self._flush_observe(stream_meta=stream_meta, stages=stages)

    def on_marker(self, timestamp: float, kind: str, raw: str, label: str) -> None:
        stages: list[dict[str, Any]] = []
        with self._lock:
            self._markers.note(
                dataset=self.dataset,
                timestamp=timestamp,
                kind=kind,
                label=label,
                newest=self.stream.latest_time(),
            )
            if kind == "rest" or label == "rest":
                self._close_rest(timestamp)
                self._rest_start = timestamp
            elif kind in {"imagery_start", "imagery_end"}:
                self._close_rest(timestamp)
                if kind == "imagery_start" and label in _HANDS:
                    stages = self._begin_imagery(timestamp, label)  # type: ignore[arg-type]
                elif kind == "imagery_end":
                    stages = self._finish_imagery(force=True)
            elif not is_activity_onset(self.dataset, kind, label):
                self._close_rest(timestamp)
            elif label not in _HANDS:
                self._close_rest(timestamp)
                stages = self._finish_imagery(force=True)
            else:
                self._close_rest(timestamp)
                stages = self._begin_imagery(timestamp, label)  # type: ignore[arg-type]
        self._flush_observe(stages=stages)

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

    def _begin_imagery(self, onset: float, cue: Direction) -> list[dict[str, Any]]:
        stages = self._finish_imagery(force=True)
        self.gate.reset()
        self.decoder.reset_trial()
        self.windows.begin(onset)
        self._imagery_active = True
        self._trial_index += 1
        self._trial_cue = cue
        self._trial_id = f"trial_{self._trial_index:03d}"
        self._sequence = 0
        start = trial_start_message(
            trial_id=self._trial_id,
            cue=cue,
            subject_id=self.subject_id,
            onset=onset,
        )
        self.hub.publish(start)
        self._events.write(trial_start_log(start))
        return stages

    def _finish_imagery(self, *, force: bool) -> list[dict[str, Any]]:
        if not self._imagery_active:
            return []
        stages = poll_and_publish(self)
        if force or self.windows.all_complete():
            trial_id = self._trial_id
            self._imagery_active = False
            self.windows.reset()
            self.decoder.reset_trial()
            invalidate = invalidate_message(trial_id, self._sequence + 1)
            self._sequence += 1
            self._last_sample = invalidate
            self.hub.publish(invalidate)
            if self.emit_legacy_v1:
                self.hub.publish(legacy_neutral_sample())
            gate = self.gate.state()
            cue = self._trial_cue
            fused, correct = trial_outcome(gate.direction, cue)
            end = trial_end_message(
                trial_id=trial_id, cue=cue, fused=fused, correct=correct
            )
            self.hub.publish(end)
            self._events.write(trial_end_log(end, gate.reason))
            self._trial_id = None
            self._trial_cue = None
        return stages
