"""Step the avatar only on stable, cue-matching decisions."""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.online.types import AvatarState, Command, Direction, GateState


@dataclass(frozen=True)
class AvatarConfig:
    """How many stable correct updates are required to reach the goal."""

    steps_to_goal: int = 4  # allowed range 3–5
    step_size: float | None = None  # defaults to 1 / steps_to_goal


class AvatarController:
    """Hold still unless the fused decision is stable and matches the cue."""

    def __init__(self, config: AvatarConfig | None = None) -> None:
        cfg = config or AvatarConfig()
        if cfg.steps_to_goal < 3 or cfg.steps_to_goal > 5:
            raise ValueError("steps_to_goal must be between 3 and 5")
        self.config = cfg
        self._step = (
            float(cfg.step_size)
            if cfg.step_size is not None
            else 1.0 / float(cfg.steps_to_goal)
        )
        self._position = 0.0
        self._steps_taken = 0
        self._target: Direction = "right"

    def begin_trial(self, target: Direction) -> AvatarState:
        self._target = target
        self._position = 0.0
        self._steps_taken = 0
        return self.state()

    def apply(self, gate: GateState) -> AvatarState:
        """Advance one step toward the target only on stable correct commands."""
        if self.goal_reached:
            return self.state()
        command: Command = gate.command
        if not gate.stable or command == "hold":
            return self.state()
        if command != self._target:
            # Wrong but stable: hold. Never reverse toward the opposite side.
            return self.state()
        direction = 1.0 if self._target == "right" else -1.0
        self._position = float(
            np_clip(self._position + direction * self._step, -1.0, 1.0)
        )
        self._steps_taken += 1
        return self.state()

    @property
    def goal_reached(self) -> bool:
        return self._steps_taken >= self.config.steps_to_goal

    def state(self) -> AvatarState:
        return AvatarState(
            position=self._position,
            steps_taken=self._steps_taken,
            steps_to_goal=self.config.steps_to_goal,
            target=self._target,
            goal_reached=self.goal_reached,
        )


def np_clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
