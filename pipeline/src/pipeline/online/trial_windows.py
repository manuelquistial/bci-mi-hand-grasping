"""Shared per-trial window processing for replay and simulator sessions."""

from __future__ import annotations

from typing import Any

from pipeline.online.avatar import AvatarController
from pipeline.online.gate import DecisionGate
from pipeline.online.log import OnlineLogger
from pipeline.online.pipeline import OnlinePipeline
from pipeline.online.types import OnlineTrialLog
from pipeline.online.windows import WindowManager


def process_ready_windows(
    *,
    pipeline: OnlinePipeline,
    logger: OnlineLogger,
    gates: dict[str, DecisionGate],
    avatar: AvatarController,
    control_name: str,
    trial: OnlineTrialLog,
    windows: WindowManager,
    log_avatar: bool = True,
) -> None:
    """Classify every ready window, update gates and the avatar, log results."""
    for window in windows.poll_ready():
        decision = pipeline.process_window(window)
        logger.log_window(trial, decision)
        for name, head_decision in pipeline.decisions_by_classifier(decision).items():
            gate = gates[name].update(head_decision)
            logger.log_gate(trial, gate, classifier=name)
        state = avatar.apply(gates[control_name].state())
        if log_avatar:
            logger.log_avatar(trial, state)


def fused_directions(gates: dict[str, DecisionGate]) -> dict[str, Any]:
    return {name: gate.state().direction for name, gate in gates.items()}
