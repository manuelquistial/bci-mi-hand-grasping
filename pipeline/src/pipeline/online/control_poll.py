"""Poll ready imagery windows and publish control + observe events."""

from __future__ import annotations

from typing import Any

from pipeline.online.control_inference import decode_evaluation, evaluate_window
from pipeline.online.control_legacy import legacy_neutral_sample, legacy_sample_from_control
from pipeline.online.control_messages import control_message, inference_log


def poll_and_publish(bridge: Any) -> list[dict[str, Any]]:
    """Publish control immediately; return observe stage jobs for outside the lock."""
    if not bridge._imagery_active or bridge._trial_id is None:
        return []
    pending_stages: list[dict[str, Any]] = []
    for window in bridge.windows.poll_ready():
        ev = evaluate_window(
            bridge.pipeline, bridge.model, bridge.decoder, bridge.gate, window
        )
        decoded = decode_evaluation(bridge.decoder, ev)
        bridge._sequence += 1
        message = control_message(
            trial_id=bridge._trial_id,
            sequence=bridge._sequence,
            window=window,
            decoded=decoded,
        )
        bridge._last_sample = message
        bridge._published += 1
        bridge.hub.publish(message)
        pending_stages.append(
            {
                "trial_id": bridge._trial_id,
                "stream": bridge.stream,
                "model": bridge.model,
                "gate_state": bridge.gate.state(),
                "cue": bridge._trial_cue,
                "window": window,
                "ev": ev,
                "decoded": decoded,
                "imagery_onset": bridge.windows.onset,
            }
        )
        if bridge.emit_legacy_v1:
            bridge.hub.publish(
                legacy_sample_from_control(decoded.control, bridge._trial_id)
                if decoded.valid
                else legacy_neutral_sample(bridge._trial_id)
            )
        bridge._events.write(
            inference_log(
                subject_id=bridge.subject_id,
                trial_id=bridge._trial_id,
                cue=bridge._trial_cue,
                window=window,
                decision=ev.decision,
                decoded=decoded,
                ea_ready=bridge.pipeline.ea_ready,
                classify_ms=ev.classify_ms,
            )
        )
    return pending_stages
