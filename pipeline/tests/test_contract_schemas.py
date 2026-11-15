"""Frozen wire-schema contracts for control WS v2 and observe WS v1.

These snapshots must stay stable across the hexagonal refactor (WP1–WP2).
"""

from __future__ import annotations

import numpy as np

from pipeline.online.control_bridge import PROTOCOL_VERSION
from pipeline.online.observe import (
    OBSERVE_PROTOCOL_VERSION,
    build_pipeline_stage_message,
    build_stream_frame_message,
)

CONTROL_BCI_KEYS = frozenset(
    {
        "version",
        "type",
        "trial_id",
        "sequence",
        "source_window_end",
        "issued_at",
        "control",
        "valid",
        "window_index",
        "strategy",
        "reason",
    }
)

CONTROL_TRIAL_START_KEYS = frozenset(
    {
        "version",
        "type",
        "trial_id",
        "cue",
        "subject_id",
        "source_window_end",
        "issued_at",
    }
)

OBSERVE_STAGE_KEYS = frozenset(
    {
        "version",
        "type",
        "trial_id",
        "cue",
        "window_index",
        "window_start",
        "window_end",
        "channels",
        "raw",
        "filtered",
        "aligned",
        "csp_traces",
        "csp_channels",
        "features",
        "p_left",
        "p_right",
        "label",
        "margin",
        "control",
        "valid",
        "strategy",
        "gate",
        "emg_rejected",
        "reason",
        "classify_ms",
    }
)

OBSERVE_STREAM_KEYS = frozenset(
    {
        "version",
        "type",
        "channels",
        "sample_rate_hz",
        "raw",
        "filtered",
    }
)

STATUS_ONLINE_KEYS = frozenset(
    {
        "source",
        "pid",
        "running",
        "samples_seen",
        "emg_samples_seen",
        "error",
        "activity",
        "registry",
        "control_ws_url",
        "observe_ws_url",
        "control_samples",
        "last_control_sample",
        "last_pipeline_stage",
        "last_stream_frame",
    }
)


def test_control_v2_bci_control_key_set() -> None:
    message = {
        "version": PROTOCOL_VERSION,
        "type": "bci_control",
        "trial_id": "trial_000",
        "sequence": 1,
        "source_window_end": 2.0,
        "issued_at": 1.0,
        "control": -0.5,
        "valid": True,
        "window_index": 0,
        "strategy": "baseline_proba",
        "reason": "classified",
    }
    assert set(message) == CONTROL_BCI_KEYS
    assert message["version"] == 2
    assert message["type"] == "bci_control"


def test_control_v2_trial_start_key_set() -> None:
    message = {
        "version": PROTOCOL_VERSION,
        "type": "trial_start",
        "trial_id": "trial_000",
        "cue": "left",
        "subject_id": "S001",
        "source_window_end": 1.0,
        "issued_at": 1.0,
    }
    assert set(message) == CONTROL_TRIAL_START_KEYS
    assert message["type"] == "trial_start"


def test_observe_pipeline_stage_key_set() -> None:
    raw = np.zeros((32, 4))
    message = build_pipeline_stage_message(
        trial_id="trial_000",
        cue="left",
        window_index=0,
        window_start=0.0,
        window_end=2.0,
        channels=["Ch1", "Ch2", "Ch3", "Ch4"],
        raw_window=raw,
        filtered_window=raw,
        aligned_trial=raw.T,
        csp_filters=np.eye(4)[:, :2],
        features=[0.1, 0.2],
        p_left=0.6,
        p_right=0.4,
        label="left",
        margin=0.5,
        control=-0.2,
        valid=True,
        strategy="baseline_proba",
        gate={"command": "hold", "stable": False, "reason": "no_valid_windows"},
        emg_rejected=False,
        reason="classified",
        classify_ms=1.0,
        max_points=16,
    )
    assert set(message) == OBSERVE_STAGE_KEYS
    assert message["version"] == OBSERVE_PROTOCOL_VERSION
    assert message["type"] == "pipeline_stage"


def test_observe_stream_frame_key_set() -> None:
    raw = np.zeros((24, 2))
    message = build_stream_frame_message(
        channels=["Cz", "C3"],
        raw=raw,
        filtered=raw,
        sample_rate_hz=250.0,
        max_points=16,
    )
    assert set(message) == OBSERVE_STREAM_KEYS
    assert message["type"] == "stream_frame"


def test_status_online_key_contract_documented() -> None:
    """Document the status.online surface the monitor and game rely on."""
    assert "observe_ws_url" in STATUS_ONLINE_KEYS
    assert "last_pipeline_stage" in STATUS_ONLINE_KEYS
    assert len(STATUS_ONLINE_KEYS) == 14
