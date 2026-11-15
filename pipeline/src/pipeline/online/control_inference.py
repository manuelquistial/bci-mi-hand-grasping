"""Per-window classification helpers for the live control bridge."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from pipeline.online.control_decoder import (
    ControlDecoder,
    ControlDecoderConfig,
    signed_decision_margin,
)
from pipeline.online.model import FrozenOnlineModel
from pipeline.online.pipeline import OnlinePipeline
from pipeline.online.types import WindowDecision

_HANDS = frozenset({"left", "right"})


@dataclass(frozen=True)
class WindowEvaluation:
    decision: WindowDecision
    aligned: np.ndarray | None
    margin: float | None
    p_left: float | None
    p_right: float | None
    classify_ms: float


def build_decoder(
    model: FrozenOnlineModel, decoder_config: ControlDecoderConfig | None
) -> ControlDecoder:
    """Bind decoder class order to the frozen classifier when it is binary."""
    classes = tuple(
        str(item) for item in getattr(model.head().classifier, "classes_", [])
    )
    if len(classes) == 2:
        cfg = decoder_config or ControlDecoderConfig()
        return ControlDecoder(
            ControlDecoderConfig(
                strategy=cfg.strategy,
                window_k=cfg.window_k,
                proportional_scale=cfg.proportional_scale,
                class_order=(classes[0], classes[1]),
            )
        )
    return ControlDecoder(decoder_config or ControlDecoderConfig())


def evaluate_window(
    pipeline: OnlinePipeline,
    model: FrozenOnlineModel,
    decoder: ControlDecoder,
    gate: Any,
    window: Any,
) -> WindowEvaluation:
    """Classify one window and gather margin/probabilities for decoding."""
    t0 = time.perf_counter()
    decision = pipeline.process_window(window)
    classify_ms = (time.perf_counter() - t0) * 1000.0
    gate.update(decision)
    aligned = pipeline.aligned_trial(window)
    margin = None
    p_left = p_right = None
    if aligned is not None and decision.classifier:
        head = model.head(decision.classifier)
        row = model.feature_vector(aligned, classifier_name=decision.classifier)
        margin = signed_decision_margin(
            head.classifier,
            row,
            class_order=decoder.config.class_order,
        )
        pair = model.predict_proba_pair(aligned, classifier_name=decision.classifier)
        if pair is not None:
            p_left, p_right = pair
    return WindowEvaluation(decision, aligned, margin, p_left, p_right, classify_ms)


def decode_evaluation(decoder: ControlDecoder, ev: WindowEvaluation) -> Any:
    decision = ev.decision
    rejected = bool(
        decision.emg_rejected
        or decision.label not in _HANDS
        or decision.reason not in {"classified", ""}
    )
    return decoder.decode(
        label=decision.label,
        score=decision.score,
        decision_margin=ev.margin,
        p_left=ev.p_left,
        p_right=ev.p_right,
        rejected=rejected,
        reason=decision.reason,
    )
