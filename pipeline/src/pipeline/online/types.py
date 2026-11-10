"""Shared types for the online motor-imagery control loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Direction = Literal["left", "right"]
Command = Literal["left", "right", "hold"]
TrialPhase = Literal["cue", "imagery", "rest", "done"]


@dataclass(frozen=True)
class ClassifierPrediction:
    """One head's label for a window (after shared CSP features)."""

    classifier: str
    label: Direction | None
    score: float | None


@dataclass(frozen=True)
class WindowDecision:
    """One classification outcome for a single 2 s window.

    When several classifiers run in parallel, ``classifier`` names the head and
    ``predictions`` holds every head's result for the same window.
    """

    window_index: int
    start: float
    end: float
    label: Direction | None
    score: float | None
    emg_rejected: bool
    reason: str = ""
    classifier: str | None = None
    predictions: tuple[ClassifierPrediction, ...] = ()


@dataclass(frozen=True)
class GateState:
    """Stability-gate output after the windows seen so far in a trial."""

    command: Command
    stable: bool
    direction: Direction | None
    valid_count: int
    majority_count: int
    consecutive: int
    reason: str


@dataclass(frozen=True)
class AvatarState:
    """Avatar progress toward the cued target."""

    position: float
    steps_taken: int
    steps_to_goal: int
    target: Direction
    goal_reached: bool


@dataclass
class OnlineTrialLog:
    """Per-trial record for later accuracy / gate analysis."""

    trial_index: int
    cue: Direction
    window_decisions: list[WindowDecision] = field(default_factory=list)
    gate_states: list[GateState] = field(default_factory=list)
    avatar_steps: list[AvatarState] = field(default_factory=list)
    fused_label: Direction | None = None
    correct: bool | None = None
    fused_by_classifier: dict[str, Direction | None] = field(default_factory=dict)
    correct_by_classifier: dict[str, bool | None] = field(default_factory=dict)
