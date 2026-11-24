"""Build observe WebSocket payloads from online pipeline state."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pipeline.online.observe import (
    build_pipeline_stage_message,
    build_stream_frame_message,
)
from pipeline.online.types import WindowDecision


def present_stream_frame(
    *,
    channels: Sequence[str],
    raw: np.ndarray,
    filtered: np.ndarray,
    sample_rate_hz: float,
    t_start: float,
    t_end: float,
    markers: Sequence[dict[str, Any]] | None = None,
    windows: Sequence[dict[str, Any]] | None = None,
    imagery_onset: float | None = None,
) -> dict[str, Any]:
    return build_stream_frame_message(
        channels=channels,
        raw=raw,
        filtered=filtered,
        sample_rate_hz=sample_rate_hz,
        t_start=t_start,
        t_end=t_end,
        markers=markers,
        windows=windows,
        imagery_onset=imagery_onset,
    )


def present_pipeline_stage(
    *,
    trial_id: str,
    cue: str | None,
    window: Any,
    decision: WindowDecision,
    aligned: np.ndarray | None,
    margin: float | None,
    p_left: float | None,
    p_right: float | None,
    control: float,
    valid: bool,
    strategy: str,
    gate: dict[str, Any],
    reason: str,
    classify_ms: float,
    channels: Sequence[str],
    raw_window: np.ndarray,
    filtered_window: np.ndarray,
    csp_filters: np.ndarray | None,
    features: Sequence[float] | None,
    imagery_onset: float | None = None,
    predictions: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return build_pipeline_stage_message(
        trial_id=trial_id,
        cue=cue,
        window_index=int(window.index),
        window_start=float(window.start),
        window_end=float(window.end),
        imagery_onset=imagery_onset,
        channels=channels,
        raw_window=raw_window,
        filtered_window=filtered_window,
        aligned_trial=aligned,
        csp_filters=csp_filters,
        features=features,
        predictions=predictions,
        p_left=p_left,
        p_right=p_right,
        label=decision.label,
        margin=margin,
        control=control,
        valid=valid,
        strategy=strategy,
        gate=gate,
        emg_rejected=bool(decision.emg_rejected),
        reason=reason,
        classify_ms=classify_ms,
    )
