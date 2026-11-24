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
from pipeline.epochs import is_activity_onset
from pipeline.online.model import FrozenOnlineModel
from pipeline.online.stream import OnlineStream
from pipeline.online.types import Direction


def marker_role(dataset: str, kind: str, label: str) -> str:
    if kind == "imagery_end":
        return "imagery_end"
    if kind == "rest" or label == "rest":
        return "rest"
    if kind == "imagery_start" or is_activity_onset(dataset, kind, label):
        return "imagery_onset"
    return "other"


def build_stream_frame(
    stream: OnlineStream,
    channel_labels: Sequence[str],
    *,
    markers: Sequence[dict[str, Any]] | None = None,
    windows: Sequence[dict[str, Any]] | None = None,
    imagery_onset: float | None = None,
) -> dict[str, Any] | None:
    newest = stream.latest_time()
    if newest is None:
        return None
    # Keep the live EEG slice short so observe never blocks control.
    # Window ribbons + MI onset still describe the full 4 s schedule.
    span = min(2.0, max(0.5, stream.keep_seconds / 6.0))
    start = float(newest) - span
    end = float(newest)
    raw, _ = stream.slice_eeg(start, end + 1e-9, filtered=False)
    filtered, _ = stream.slice_eeg(start, end + 1e-9, filtered=True)
    if raw.shape[0] < 8:
        return None
    # Cap channels on the wire — full montages make JSON huge and stall the game.
    max_channels = 8
    labels = list(channel_labels)[:max_channels]
    if raw.shape[1] > max_channels:
        raw = raw[:, :max_channels]
        filtered = filtered[:, :max_channels]
    visible = [
        item
        for item in (markers or ())
        if start - 1e-6 <= float(item["t"]) <= end + 1e-6
    ]
    return present_stream_frame(
        channels=labels,
        raw=raw,
        filtered=filtered,
        sample_rate_hz=stream.sample_rate_hz,
        t_start=float(start),
        t_end=float(end),
        markers=visible,
        windows=windows,
        imagery_onset=imagery_onset,
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
    imagery_onset: float | None = None,
) -> dict[str, Any]:
    decision, aligned = ev.decision, ev.aligned
    raw, _ = stream.slice_eeg(window.start, window.end, filtered=False)
    filtered = np.asarray(window.eeg, dtype=np.float64)
    if raw.shape[0] == 0:
        raw = filtered
    # Always expose CSP log-vars for the monitor, even when the head rejected.
    feature_source = aligned
    if feature_source is None and filtered.ndim == 2 and filtered.size:
        feature_source = filtered.T
    features = None
    if feature_source is not None:
        try:
            features = model.raw_features(feature_source).reshape(-1)
        except Exception:  # noqa: BLE001
            features = None
    gate = gate_state
    predictions: list[dict[str, Any]] = []
    for pred in decision.predictions or ():
        p_left = p_right = None
        if aligned is not None and pred.label is not None:
            try:
                pair = model.predict_proba_pair(
                    aligned, classifier_name=pred.classifier
                )
            except Exception:  # noqa: BLE001
                pair = None
            if pair is not None:
                p_left, p_right = pair
        predictions.append(
            {
                "classifier": str(pred.classifier),
                "label": None if pred.label is None else str(pred.label),
                "score": None if pred.score is None else float(pred.score),
                "p_left": None if p_left is None else float(p_left),
                "p_right": None if p_right is None else float(p_right),
            }
        )
    # Cap EEG montage traces on the wire; CSP still uses the full aligned trial.
    max_channels = 8
    labels = list(channel_labels)[:max_channels]
    if raw.ndim == 2 and raw.shape[1] > max_channels:
        raw = raw[:, :max_channels]
    if filtered.ndim == 2 and filtered.shape[1] > max_channels:
        filtered = filtered[:, :max_channels]
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
        channels=labels,
        raw_window=raw,
        filtered_window=filtered,
        csp_filters=model.csp_filters,
        features=features,
        imagery_onset=imagery_onset,
        predictions=predictions,
    )


def resolve_channel_labels(
    channel_labels: Sequence[str] | None, channel_count: int
) -> list[str]:
    if channel_labels and len(channel_labels) == int(channel_count):
        return [str(name) for name in channel_labels]
    return [f"Ch{i + 1}" for i in range(int(channel_count))]


class RecentMarkers:
    """Keep dataset markers that still fall inside the stream ring buffer."""

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    def note(
        self, *, dataset: str, timestamp: float, kind: str, label: str, newest: float | None
    ) -> None:
        self._items.append(
            {
                "t": float(timestamp),
                "kind": str(kind),
                "label": str(label),
                "role": marker_role(dataset, kind, label),
            }
        )
        if newest is None:
            return
        keep_from = float(newest) - 16.0
        self._items = [item for item in self._items if float(item["t"]) >= keep_from]

    @property
    def items(self) -> list[dict[str, Any]]:
        return list(self._items)


class ObservePublisher:
    """Publish stream frames and pipeline stages to the observe hub."""

    def __init__(self, hub: Any | None, channel_labels: Sequence[str]) -> None:
        self.hub = hub
        self.channel_labels = list(channel_labels)
        self.last_stage: dict[str, Any] | None = None
        self.last_stream_frame: dict[str, Any] | None = None

    def publish_stream_frame(
        self,
        stream: OnlineStream,
        *,
        markers: Sequence[dict[str, Any]] | None = None,
        windows: Sequence[dict[str, Any]] | None = None,
        imagery_onset: float | None = None,
    ) -> None:
        if self.hub is None:
            return
        try:
            message = build_stream_frame(
                stream,
                self.channel_labels,
                markers=markers,
                windows=windows,
                imagery_onset=imagery_onset,
            )
            if message is None:
                return
            self.last_stream_frame = message
            self.hub.publish(message)
        except Exception:  # noqa: BLE001 — never stall control for observe
            return

    def publish_stage(
        self, trial_id: str | None, *, imagery_onset: float | None = None, **kwargs: Any
    ) -> None:
        if self.hub is None or trial_id is None:
            return
        try:
            message = build_pipeline_stage(
                channel_labels=self.channel_labels,
                trial_id=trial_id,
                imagery_onset=imagery_onset,
                **kwargs,
            )
            self.last_stage = message
            self.hub.publish(message)
        except Exception:  # noqa: BLE001 — never stall control for observe
            return
