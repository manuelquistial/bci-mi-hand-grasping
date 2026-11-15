"""Observe-hub frame builders for the live control bridge.

Presentation itself lives in ``application.observe_presenter``; this module
only gathers inputs from the stream/model/gate and delegates to it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pipeline.application.observe_presenter import (
    present_pipeline_stage,
    present_stream_frame,
)
from pipeline.online.model import FrozenOnlineModel
from pipeline.online.stream import OnlineStream
from pipeline.online.types import Direction


def build_stream_frame(
    stream: OnlineStream, channel_labels: Sequence[str]
) -> dict[str, Any] | None:
    newest = stream.latest_time()
    if newest is None:
        return None
    span = min(2.0, max(0.5, stream.keep_seconds / 6.0))
    start = newest - span
    raw, _ = stream.slice_eeg(start, newest + 1e-9, filtered=False)
    filtered, _ = stream.slice_eeg(start, newest + 1e-9, filtered=True)
    if raw.shape[0] < 8:
        return None
    return present_stream_frame(
        channels=channel_labels,
        raw=raw,
        filtered=filtered,
        sample_rate_hz=stream.sample_rate_hz,
    )


def build_pipeline_stage(
    *,
    stream: OnlineStream,
    model: FrozenOnlineModel,
    gate_state: Any,
    channel_labels: Sequence[str],
    trial_id: str,
    cue: Direction | None,
    window: Any,
    ev: Any,
    decoded: Any,
) -> dict[str, Any]:
    decision, aligned = ev.decision, ev.aligned
    raw, _ = stream.slice_eeg(window.start, window.end, filtered=False)
    filtered = np.asarray(window.eeg, dtype=np.float64)
    if raw.shape[0] == 0:
        raw = filtered
    features = None
    if aligned is not None and decision.classifier:
        try:
            features = model.raw_features(aligned).reshape(-1)
        except Exception:  # noqa: BLE001
            features = None
    gate = gate_state
    return present_pipeline_stage(
        trial_id=trial_id,
        cue=cue,
        window=window,
        decision=decision,
        aligned=aligned,
        margin=ev.margin,
        p_left=ev.p_left,
        p_right=ev.p_right,
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
        reason=str(decision.reason or decoded.reason or ""),
        classify_ms=ev.classify_ms,
        channels=channel_labels,
        raw_window=raw,
        filtered_window=filtered,
        csp_filters=model.csp_filters,
        features=features,
    )


def resolve_channel_labels(
    channel_labels: Sequence[str] | None, channel_count: int
) -> list[str]:
    if channel_labels and len(channel_labels) == int(channel_count):
        return [str(name) for name in channel_labels]
    return [f"Ch{i + 1}" for i in range(int(channel_count))]


class ObservePublisher:
    """Publish stream frames and pipeline stages to the observe hub."""

    def __init__(self, hub: Any | None, channel_labels: Sequence[str]) -> None:
        self.hub = hub
        self.channel_labels = list(channel_labels)
        self.last_stage: dict[str, Any] | None = None
        self.last_stream_frame: dict[str, Any] | None = None

    def publish_stream_frame(self, stream: OnlineStream) -> None:
        if self.hub is None:
            return
        message = build_stream_frame(stream, self.channel_labels)
        if message is None:
            return
        self.last_stream_frame = message
        self.hub.publish(message)

    def publish_stage(self, trial_id: str | None, **kwargs: Any) -> None:
        if self.hub is None or trial_id is None:
            return
        message = build_pipeline_stage(
            channel_labels=self.channel_labels, trial_id=trial_id, **kwargs
        )
        self.last_stage = message
        self.hub.publish(message)
