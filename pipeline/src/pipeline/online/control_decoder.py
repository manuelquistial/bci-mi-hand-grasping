"""Deterministic control adaptation after the frozen MI classifier.

Strategies operate on the raw SVM decision margin. They do not retrain or
alter model weights. Margins are trial-local; the buffer resets on every
imagery start/end.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

ControlStrategy = Literal[
    "baseline_proba",
    "discrete",
    "temporal_discrete",
    "temporal_proportional",
]


@dataclass(frozen=True)
class ControlDecoderConfig:
    """Tunable decoder parameters (not learned from the held-out subject)."""

    strategy: ControlStrategy = "baseline_proba"
    window_k: int = 3
    proportional_scale: float = 1.0
    class_order: tuple[str, str] = ("left", "right")


@dataclass(frozen=True)
class DecodedControl:
    """One normalized horizontal command for the game."""

    control: float
    valid: bool
    strategy: ControlStrategy
    raw_margin: float | None
    integrated_margin: float | None
    buffer_size: int
    label: str | None
    reason: str = ""


@dataclass
class ControlDecoder:
    """Map classifier margins → control ∈ [-1, 1] with optional temporal pooling."""

    config: ControlDecoderConfig = field(default_factory=ControlDecoderConfig)
    _margins: deque[float] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        k = max(1, int(self.config.window_k))
        self._margins = deque(maxlen=k)

    def reset_trial(self) -> None:
        self._margins.clear()

    def decode(
        self,
        *,
        label: str | None,
        score: float | None,
        decision_margin: float | None = None,
        p_left: float | None = None,
        p_right: float | None = None,
        rejected: bool = False,
        reason: str = "",
    ) -> DecodedControl:
        """Produce one command from a single window classification."""
        if rejected or label not in {"left", "right"}:
            return DecodedControl(
                control=0.0,
                valid=False,
                strategy=self.config.strategy,
                raw_margin=None if decision_margin is None else float(decision_margin),
                integrated_margin=None,
                buffer_size=len(self._margins),
                label=label,
                reason=reason or "rejected",
            )

        margin = self._resolve_margin(
            label=label,
            score=score,
            decision_margin=decision_margin,
            p_left=p_left,
            p_right=p_right,
        )
        if margin is None or not math.isfinite(margin):
            return DecodedControl(
                control=0.0,
                valid=False,
                strategy=self.config.strategy,
                raw_margin=None,
                integrated_margin=None,
                buffer_size=len(self._margins),
                label=label,
                reason="non_finite_margin",
            )

        strategy = self.config.strategy
        if strategy == "baseline_proba":
            control = self._baseline_control(
                label=label,
                score=score,
                p_left=p_left,
                p_right=p_right,
                margin=margin,
            )
            return DecodedControl(
                control=_clamp(control),
                valid=True,
                strategy=strategy,
                raw_margin=margin,
                integrated_margin=margin,
                buffer_size=1,
                label=label,
                reason="ok",
            )

        self._margins.append(margin)
        integrated = float(sum(self._margins) / len(self._margins))
        if strategy == "discrete":
            control = self._sign_to_control(margin)
            used = margin
        elif strategy == "temporal_discrete":
            control = self._sign_to_control(integrated)
            used = integrated
        elif strategy == "temporal_proportional":
            scale = float(self.config.proportional_scale)
            control = float(math.tanh(scale * integrated))
            used = integrated
        else:
            return DecodedControl(
                control=0.0,
                valid=False,
                strategy=strategy,
                raw_margin=margin,
                integrated_margin=None,
                buffer_size=len(self._margins),
                label=label,
                reason=f"unknown_strategy:{strategy}",
            )
        return DecodedControl(
            control=_clamp(control),
            valid=True,
            strategy=strategy,
            raw_margin=margin,
            integrated_margin=used,
            buffer_size=len(self._margins),
            label=label,
            reason="ok",
        )

    def _resolve_margin(
        self,
        *,
        label: str,
        score: float | None,
        decision_margin: float | None,
        p_left: float | None,
        p_right: float | None,
    ) -> float | None:
        if decision_margin is not None and math.isfinite(float(decision_margin)):
            return float(decision_margin)
        # Reconstruct a signed margin from class-conditional score.
        # classes_ order left,right → positive favors right.
        if score is not None and math.isfinite(float(score)):
            value = abs(float(score))
            return value if label == self.config.class_order[1] else -value
        if p_left is not None and p_right is not None:
            return float(p_right) - float(p_left)
        return None

    def _baseline_control(
        self,
        *,
        label: str,
        score: float | None,
        p_left: float | None,
        p_right: float | None,
        margin: float,
    ) -> float:
        if p_left is not None and p_right is not None:
            return float(p_right) - float(p_left)
        # Legacy sigmoid confidence assigned to the predicted class.
        conf = 1.0 / (1.0 + math.exp(-abs(float(score or margin))))
        conf = min(0.99, max(0.51, conf))
        left = conf if label == "left" else 1.0 - conf
        right = 1.0 - left
        return right - left

    def _sign_to_control(self, margin: float) -> float:
        if margin > 0:
            return 1.0  # right (classes_[1])
        if margin < 0:
            return -1.0  # left (classes_[0])
        return 0.0


def signed_decision_margin(
    classifier: object,
    feature_row: np.ndarray,
    *,
    class_order: Sequence[str] = ("left", "right"),
) -> float | None:
    """Return decision_function signed so positive favors ``class_order[1]``."""
    decision = getattr(classifier, "decision_function", None)
    if not callable(decision):
        return None
    values = np.asarray(decision(feature_row), dtype=np.float64).reshape(-1)
    if values.size != 1:
        return None
    classes = [str(item) for item in getattr(classifier, "classes_", [])]
    if len(classes) != 2:
        return None
    raw = float(values[0])
    # sklearn: positive favors classes_[1]
    if classes[1] == class_order[1] and classes[0] == class_order[0]:
        return raw
    if classes[1] == class_order[0] and classes[0] == class_order[1]:
        return -raw
    return raw


def _clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, float(value)))
