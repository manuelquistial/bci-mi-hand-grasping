"""Protocol v2 message and log-row builders for the live control bridge."""

from __future__ import annotations

import time
from typing import Any

from pipeline.online.types import Direction

PROTOCOL_VERSION = 2


def trial_start_message(
    *, trial_id: str, cue: Direction, subject_id: str, onset: float
) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": "trial_start",
        "trial_id": trial_id,
        "cue": cue,
        "subject_id": subject_id,
        "source_window_end": float(onset),
        "issued_at": time.time(),
    }


def trial_start_log(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": "trial_start",
        "trial_id": message["trial_id"],
        "cue": message["cue"],
        "subject_id": message["subject_id"],
        "onset": message["source_window_end"],
    }


def invalidate_message(trial_id: str | None, sequence: int) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": "bci_control",
        "trial_id": trial_id,
        "sequence": sequence,
        "source_window_end": None,
        "issued_at": time.time(),
        "control": 0.0,
        "valid": False,
        "reason": "trial_end",
    }


def trial_end_message(
    *, trial_id: str | None, cue: Direction | None, fused: Any, correct: bool | None
) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": "trial_end",
        "trial_id": trial_id,
        "cue": cue,
        "fused_label": fused,
        "correct": correct,
        "issued_at": time.time(),
    }


def trial_end_log(message: dict[str, Any], gate_reason: str) -> dict[str, Any]:
    return {
        "event": "trial_end",
        "trial_id": message["trial_id"],
        "cue": message["cue"],
        "fused_label": message["fused_label"],
        "correct": message["correct"],
        "gate_reason": gate_reason,
    }


def control_message(
    *, trial_id: str | None, sequence: int, window: Any, decoded: Any
) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": "bci_control",
        "trial_id": trial_id,
        "sequence": sequence,
        "source_window_end": float(window.end),
        "issued_at": time.time(),
        "control": float(decoded.control) if decoded.valid else 0.0,
        "valid": bool(decoded.valid),
        "window_index": int(window.index),
        "strategy": decoded.strategy,
        "reason": decoded.reason,
    }


def inference_log(
    *,
    subject_id: str,
    trial_id: str | None,
    cue: Direction | None,
    window: Any,
    decision: Any,
    decoded: Any,
    ea_ready: bool,
    classify_ms: float,
) -> dict[str, Any]:
    return {
        "event": "inference",
        "subject_id": subject_id,
        "trial_id": trial_id,
        "cue": cue,
        "window_index": window.index,
        "window_start": float(window.start),
        "window_end": float(window.end),
        "label": decision.label,
        "score": decision.score,
        "raw_margin": decoded.raw_margin,
        "integrated_margin": decoded.integrated_margin,
        "control": decoded.control,
        "valid": decoded.valid,
        "ea_ready": ea_ready,
        "emg_rejected": decision.emg_rejected,
        "reason": decision.reason,
        "strategy": decoded.strategy,
        "classify_ms": classify_ms,
    }


def trial_outcome(gate_direction: Any, cue: Direction | None) -> tuple[Any, bool | None]:
    """Return ``(fused_label, correct)`` from the gate direction and cue."""
    hands = ("left", "right")
    fused = gate_direction if gate_direction in hands else None
    correct = fused == cue if fused in hands and cue in hands else None
    return fused, correct
