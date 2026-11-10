"""Fuse per-window decisions into a stable move/hold command."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pipeline.online.types import Command, Direction, GateState, WindowDecision


@dataclass(frozen=True)
class DecisionGateConfig:
    """Stability rules before the avatar is allowed to move."""

    min_consecutive: int = 3
    majority_of: int = 3
    require_majority_of_valid: bool = True


class DecisionGate:
    """Collect valid SVM/LR window labels and emit move/hold.

    A direction is stable when either:
    - the last ``min_consecutive`` valid decisions agree, or
    - at least ``majority_of`` valid decisions agree on the same side.

    Mixed or insufficient evidence yields ``hold`` (never reverses).
    """

    def __init__(self, config: DecisionGateConfig | None = None) -> None:
        self.config = config or DecisionGateConfig()
        self._decisions: list[WindowDecision] = []

    def reset(self) -> None:
        self._decisions = []

    def update(self, decision: WindowDecision) -> GateState:
        self._decisions.append(decision)
        return self.state()

    def extend(self, decisions: Sequence[WindowDecision]) -> GateState:
        self._decisions.extend(decisions)
        return self.state()

    def state(self) -> GateState:
        valid = [
            item.label
            for item in self._decisions
            if not item.emg_rejected and item.label in {"left", "right"}
        ]
        if not valid:
            return GateState(
                command="hold",
                stable=False,
                direction=None,
                valid_count=0,
                majority_count=0,
                consecutive=0,
                reason="no_valid_windows",
            )
        consecutive = _trailing_run(valid)
        majority_label, majority_count = _majority(valid)
        consecutive_ok = (
            consecutive >= self.config.min_consecutive
            and valid[-1] is not None
            and _trailing_label(valid) is not None
        )
        majority_ok = majority_count >= self.config.majority_of
        direction: Direction | None = None
        reason = "unstable"
        if consecutive_ok:
            direction = _trailing_label(valid)
            reason = f"consecutive_{consecutive}"
        elif majority_ok and majority_label is not None:
            direction = majority_label
            reason = f"majority_{majority_count}"
        stable = direction is not None
        command: Command = direction if stable and direction is not None else "hold"
        return GateState(
            command=command,
            stable=stable,
            direction=direction,
            valid_count=len(valid),
            majority_count=majority_count,
            consecutive=consecutive,
            reason=reason,
        )


def _trailing_run(labels: Sequence[Direction]) -> int:
    if not labels:
        return 0
    last = labels[-1]
    run = 0
    for label in reversed(labels):
        if label != last:
            break
        run += 1
    return run


def _trailing_label(labels: Sequence[Direction]) -> Direction | None:
    return labels[-1] if labels else None


def _majority(labels: Sequence[Direction]) -> tuple[Direction | None, int]:
    left = sum(1 for label in labels if label == "left")
    right = sum(1 for label in labels if label == "right")
    if left == right:
        return None, left
    if left > right:
        return "left", left
    return "right", right
