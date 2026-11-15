"""Contract tests for online observability payloads."""

from __future__ import annotations

import numpy as np

from pipeline.online.observe import (
    OBSERVE_PROTOCOL_VERSION,
    build_pipeline_stage_message,
    build_stream_frame_message,
    csp_component_traces,
    downsample_channels_by_time,
    downsample_samples_by_channel,
)


def test_downsample_channels_by_time_caps_points() -> None:
    data = np.arange(16 * 1000, dtype=np.float64).reshape(16, 1000)
    series = downsample_channels_by_time(data, max_points=40)
    assert len(series) == 16
    assert all(len(row) <= 40 for row in series)
    assert all(isinstance(value, float) for value in series[0])


def test_downsample_samples_by_channel() -> None:
    data = np.random.default_rng(0).normal(size=(200, 8))
    series = downsample_samples_by_channel(data, max_points=50)
    assert len(series) == 8
    assert all(len(row) <= 50 for row in series)


def test_csp_component_traces() -> None:
    trial = np.random.default_rng(1).normal(size=(8, 128))
    filters = np.eye(8)[:, :4]
    traces = csp_component_traces(trial, filters, max_points=32)
    assert len(traces) == 4
    assert all(len(row) <= 32 for row in traces)


def test_pipeline_stage_message_shape() -> None:
    raw = np.random.default_rng(2).normal(size=(100, 8))
    filtered = raw * 0.5
    aligned = filtered.T
    filters = np.eye(8)[:, :2]
    message = build_pipeline_stage_message(
        trial_id="trial_000",
        cue="left",
        window_index=2,
        window_start=1.0,
        window_end=3.0,
        channels=[f"Ch{i}" for i in range(8)],
        raw_window=raw,
        filtered_window=filtered,
        aligned_trial=aligned,
        csp_filters=filters,
        features=[0.1, 0.2, 0.3, 0.4],
        p_left=0.7,
        p_right=0.3,
        label="left",
        margin=1.2,
        control=-0.4,
        valid=True,
        strategy="baseline_proba",
        gate={"command": "left", "stable": True, "reason": "consecutive_3"},
        emg_rejected=False,
        reason="classified",
        classify_ms=1.5,
        max_points=48,
    )
    assert message["version"] == OBSERVE_PROTOCOL_VERSION
    assert message["type"] == "pipeline_stage"
    assert message["trial_id"] == "trial_000"
    assert len(message["raw"]) == 8
    assert len(message["filtered"]) == 8
    assert len(message["aligned"]) == 8
    assert len(message["csp_traces"]) == 2
    assert message["features"] == [0.1, 0.2, 0.3, 0.4]
    assert message["p_left"] == 0.7
    assert message["control"] == -0.4
    assert message["gate"]["command"] == "left"


def test_stream_frame_message_shape() -> None:
    raw = np.random.default_rng(3).normal(size=(80, 4))
    filtered = raw + 0.1
    message = build_stream_frame_message(
        channels=["Cz", "C3", "C4", "Fz"],
        raw=raw,
        filtered=filtered,
        sample_rate_hz=250.0,
        max_points=40,
    )
    assert message["type"] == "stream_frame"
    assert message["sample_rate_hz"] == 250.0
    assert message["channels"] == ["Cz", "C3", "C4", "Fz"]
    assert len(message["raw"]) == 4
    assert len(message["filtered"]) == 4
