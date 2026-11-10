"""Orchestrate cue → imagery (5 windows) → rest for each online trial."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from pipeline.online.avatar import AvatarController
from pipeline.online.gate import DecisionGate
from pipeline.online.log import OnlineLogger
from pipeline.online.pipeline import OnlinePipeline
from pipeline.online.stream import OnlineStream
from pipeline.online.types import Direction, OnlineTrialLog, TrialPhase
from pipeline.online.windows import WindowManager

Clock = Callable[[], float]
Sleep = Callable[[float], None]


@dataclass(frozen=True)
class TrialTiming:
    """Timings requested for the online control loop."""

    cue_seconds: float = 2.0
    imagery_seconds: float = 4.0
    rest_seconds: float = 3.0
    poll_seconds: float = 0.05


class TrialLoop:
    """Run the online trial cycle on top of stream / window / pipeline modules.

    Avatar motion is pure Python state here. The Phaser game can later consume
    the same move/hold commands through a bridge; this loop stays testable
    without the browser.
    """

    def __init__(
        self,
        stream: OnlineStream,
        windows: WindowManager,
        pipeline: OnlinePipeline,
        gate: DecisionGate,
        avatar: AvatarController,
        logger: OnlineLogger | None = None,
        timing: TrialTiming | None = None,
        *,
        clock: Clock | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self.stream = stream
        self.windows = windows
        self.pipeline = pipeline
        self.gate = gate
        self.avatar = avatar
        self.logger = logger or OnlineLogger()
        self.timing = timing or TrialTiming()
        self._clock = clock or _default_clock
        self._sleep = sleep or _default_sleep
        self.phase: TrialPhase = "done"
        self._trial_logs: list[OnlineTrialLog] = []

    def run_session(self, cues: Sequence[Direction]) -> dict:
        """Run every cue in order. Rest before the first trial seeds EA."""
        if not cues:
            return self.logger.summary(
                offline_accuracy=self.pipeline.model.offline_accuracy
            )
        self._seed_rest()
        for index, cue in enumerate(cues):
            self.run_trial(index, cue)
        return self.logger.summary(
            offline_accuracy=self.pipeline.model.offline_accuracy
        )

    def run_trial(self, trial_index: int, cue: Direction) -> OnlineTrialLog:
        trial = self.logger.start_trial(trial_index, cue)
        self.gate.reset()
        self.avatar.begin_trial(cue)

        self.phase = "cue"
        self._wait_phase(self.timing.cue_seconds)

        self.stream.poll()
        onset = self.stream.latest_time()
        if onset is None:
            onset = self._clock()
        self.windows.begin(float(onset))
        self.phase = "imagery"
        imagery_end = self._clock() + self.timing.imagery_seconds
        while self._clock() < imagery_end or not self.windows.all_complete():
            self.stream.poll()
            for window in self.windows.poll_ready():
                decision = self.pipeline.process_window(window)
                self.logger.log_window(trial, decision)
                gate = self.gate.update(decision)
                self.logger.log_gate(trial, gate)
                avatar = self.avatar.apply(gate)
                self.logger.log_avatar(trial, avatar)
                if avatar.goal_reached:
                    break
            if self.avatar.goal_reached and self.windows.all_complete():
                break
            if self._clock() >= imagery_end and self.windows.all_complete():
                break
            self._sleep(self.timing.poll_seconds)

        fused = self.gate.state().direction
        self.logger.end_trial(trial, fused_label=fused)
        self._trial_logs.append(trial)

        self.phase = "rest"
        rest_start = self.stream.latest_time()
        self._wait_phase(self.timing.rest_seconds)
        rest_end = self.stream.latest_time()
        if rest_start is not None and rest_end is not None and rest_end > rest_start:
            self._ingest_rest(rest_start, rest_end)
        elif self.stream.latest_time() is not None:
            end = float(self.stream.latest_time())
            self._ingest_rest(end - self.timing.rest_seconds, end)
        self.phase = "done"
        self.windows.reset()
        return trial

    def _seed_rest(self) -> None:
        """Collect an initial rest segment so EA is ready before imagery."""
        self.phase = "rest"
        start = self.stream.latest_time()
        self._wait_phase(self.timing.rest_seconds)
        end = self.stream.latest_time()
        if start is not None and end is not None and end > start:
            self._ingest_rest(start, end)
        elif end is not None:
            self._ingest_rest(end - self.timing.rest_seconds, end)

    def _ingest_rest(self, start: float, end: float) -> None:
        eeg, _times = self.stream.slice_eeg(start, end, filtered=True)
        if eeg.shape[0] >= 16:
            self.pipeline.add_rest(eeg)

    def _wait_phase(self, seconds: float) -> None:
        end = self._clock() + float(seconds)
        while self._clock() < end:
            self.stream.poll()
            remaining = end - self._clock()
            self._sleep(min(self.timing.poll_seconds, max(0.0, remaining)))


def _default_clock() -> float:
    import time

    return time.monotonic()


def _default_sleep(seconds: float) -> None:
    import time

    if seconds > 0:
        time.sleep(seconds)
