"""Legacy v1 ``left/right/timestamp`` control samples (backward compatible)."""

from __future__ import annotations

import math
import time

import numpy as np

from pipeline.online.model import FrozenOnlineModel
from pipeline.online.types import WindowDecision

_HANDS = frozenset({"left", "right"})


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


def legacy_neutral_sample(trial_id: str | None = None) -> dict[str, object]:
    """Neutral v1 sample so old clients stop holding the last bias."""
    sample: dict[str, object] = {"left": 0.5, "right": 0.5, "timestamp": time.time()}
    if trial_id is not None:
        sample["trial_id"] = trial_id
    sample["valid"] = False
    return sample


def legacy_sample_from_control(control: float, trial_id: str | None) -> dict[str, object]:
    """Map control ∈ [-1, 1] back to left/right for old adapters."""
    right = 0.5 + 0.5 * float(control)
    left = 1.0 - right
    return {
        "left": left,
        "right": right,
        "timestamp": time.time(),
        "trial_id": trial_id,
        "valid": True,
    }
