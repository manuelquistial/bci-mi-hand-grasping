"""Observability payloads for the online pipeline monitor (separate from game control)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

OBSERVE_PROTOCOL_VERSION = 1
DEFAULT_MAX_POINTS = 240


def downsample_channels_by_time(
    data_channels_by_time: np.ndarray,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
) -> list[list[float]]:
    """``(channels, samples)`` → list of per-channel series, decimated for the UI."""
    array = np.asarray(data_channels_by_time, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        return []
    channels, samples = array.shape
    count = min(int(samples), max(8, int(max_points)))
    if count < samples:
        index = np.unique(np.linspace(0, samples - 1, count).astype(int))
        array = array[:, index]
    return [[round(float(value), 4) for value in row] for row in array]


def downsample_samples_by_channel(
    data_samples_by_channel: np.ndarray,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
) -> list[list[float]]:
    """``(samples, channels)`` → list of per-channel series."""
    array = np.asarray(data_samples_by_channel, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        return []
    return downsample_channels_by_time(array.T, max_points=max_points)


def csp_component_traces(
    trial_channels_by_time: np.ndarray,
    csp_filters: np.ndarray,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
) -> list[list[float]]:
    """Project EA-aligned trial through frozen CSP filters → component×time traces."""
    trial = np.asarray(trial_channels_by_time, dtype=np.float64)
    filters = np.asarray(csp_filters, dtype=np.float64)
    if trial.ndim != 2 or filters.ndim != 2:
        return []
    if trial.shape[0] != filters.shape[0]:
        return []
    projected = filters.T @ trial
    return downsample_channels_by_time(projected, max_points=max_points)


def feature_bars(feature_row: Sequence[float] | np.ndarray) -> list[float]:
    values = np.asarray(feature_row, dtype=np.float64).reshape(-1)
    return [round(float(value), 5) for value in values]


def build_pipeline_stage_message(
    *,
    trial_id: str,
    cue: str | None,
    window_index: int,
    window_start: float,
    window_end: float,
    channels: Sequence[str],
    raw_window: np.ndarray,
    filtered_window: np.ndarray,
    aligned_trial: np.ndarray | None,
    csp_filters: np.ndarray | None,
    features: Sequence[float] | None,
    p_left: float | None,
    p_right: float | None,
    label: str | None,
    margin: float | None,
    control: float,
    valid: bool,
    strategy: str,
    gate: dict[str, Any],
    emg_rejected: bool,
    reason: str,
    classify_ms: float,
    imagery_onset: float | None = None,
    predictions: Sequence[dict[str, Any]] | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict[str, Any]:
    """Versioned observability event for one imagery window."""
    csp_traces: list[list[float]] = []
    if aligned_trial is not None and csp_filters is not None:
        csp_traces = csp_component_traces(
            aligned_trial, csp_filters, max_points=max_points
        )
    # Keep EA display to the same capped montage as raw/filtered channels.
    aligned_view = aligned_trial
    if aligned_view is not None:
        aligned_arr = np.asarray(aligned_view, dtype=np.float64)
        n_show = min(len(channels), aligned_arr.shape[0] if aligned_arr.ndim == 2 else 0)
        aligned_view = aligned_arr[:n_show] if n_show else None
    aligned_series = (
        downsample_channels_by_time(aligned_view, max_points=max_points)
        if aligned_view is not None
        else []
    )
    return {
        "version": OBSERVE_PROTOCOL_VERSION,
        "type": "pipeline_stage",
        "trial_id": trial_id,
        "cue": cue,
        "window_index": int(window_index),
        "window_start": float(window_start),
        "window_end": float(window_end),
        "imagery_onset": None if imagery_onset is None else float(imagery_onset),
        "channels": list(channels),
        "raw": downsample_samples_by_channel(raw_window, max_points=max_points),
        "filtered": downsample_samples_by_channel(
            filtered_window, max_points=max_points
        ),
        "aligned": aligned_series,
        "csp_traces": csp_traces,
        "csp_channels": [f"CSP{i + 1}" for i in range(len(csp_traces))],
        "features": feature_bars(features) if features is not None else [],
        "predictions": list(predictions or ()),
        "p_left": None if p_left is None else float(p_left),
        "p_right": None if p_right is None else float(p_right),
        "label": label,
        "margin": None if margin is None else float(margin),
        "control": float(control),
        "valid": bool(valid),
        "strategy": strategy,
        "gate": gate,
        "emg_rejected": bool(emg_rejected),
        "reason": reason,
        "classify_ms": round(float(classify_ms), 3),
    }


def build_stream_frame_message(
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
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict[str, Any]:
    """Continuous ring-buffer frame (raw + causal filtered) with timeline cues."""
    return {
        "version": OBSERVE_PROTOCOL_VERSION,
        "type": "stream_frame",
        "channels": list(channels),
        "sample_rate_hz": float(sample_rate_hz),
        "t_start": float(t_start),
        "t_end": float(t_end),
        "imagery_onset": None if imagery_onset is None else float(imagery_onset),
        "markers": list(markers or ()),
        "windows": list(windows or ()),
        "raw": downsample_samples_by_channel(raw, max_points=max_points),
        "filtered": downsample_samples_by_channel(filtered, max_points=max_points),
    }
