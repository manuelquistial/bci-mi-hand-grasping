"""Run the online pipeline against a live simulator LSL playback.

Flow: start ``simopenbci`` → read EEG + Markers → rest seeds EA → each
left/right imagery onset cuts five 2 s windows → EMG gate → frozen
filter/EA/CSP/classifier → decision gate → JSONL + summary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pipeline.epochs import is_activity_onset
from pipeline.live import parse_marker_text
from pipeline.online.avatar import AvatarConfig, AvatarController
from pipeline.online.emg_reject import EMGRejector, EmgRejectConfig
from pipeline.online.gate import DecisionGate, DecisionGateConfig
from pipeline.online.log import OnlineLogger
from pipeline.online.model import FrozenOnlineModel, load_frozen_model
from pipeline.online.pipeline import OnlinePipeline
from pipeline.online.stream import OnlineStream
from pipeline.online.trial_windows import fused_directions, process_ready_windows
from pipeline.online.types import Direction, OnlineTrialLog
from pipeline.online.windows import WindowManager
from pipeline.adapters.lsl.streams import open_eeg_stream, open_marker_stream
from pipeline.adapters.simulator.process import (
    build_simulator_command, start_simulator_process,
)
from pipeline.config import SIMULATOR_LSL_NAME, default_simulator_project

_HANDS = frozenset({"left", "right"})
ProcessStarter = Callable[[list[str]], Any]
EegOpener = Callable[..., Any]
MarkerOpener = Callable[..., Any]


@dataclass(frozen=True)
class SimulatorOnlineConfig:
    dataset: str = "shuqfa2024"
    subject: str | None = None
    session_dir: str | None = None
    lsl_name: str = SIMULATOR_LSL_NAME
    wait_seconds: float = 8.0
    max_trials: int | None = None
    emg_threshold: float = 50.0
    steps_to_goal: int = 4
    log_path: str | Path | None = None
    idle_timeout_seconds: float = 15.0


class SimulatorOnlineSession:
    """End-to-end online classification on one simulator dataset playback."""

    def __init__(
        self,
        model: FrozenOnlineModel,
        config: SimulatorOnlineConfig | None = None,
        *,
        simulator_project: Path | None = None,
        process_starter: ProcessStarter | None = None,
        eeg_opener: EegOpener | None = None,
        marker_opener: MarkerOpener | None = None,
    ) -> None:
        self.model = model
        self.config = config or SimulatorOnlineConfig()
        self._project = Path(simulator_project or default_simulator_project())
        self._process_starter = process_starter or start_simulator_process
        self._eeg_opener = eeg_opener or open_eeg_stream
        self._marker_opener = marker_opener or open_marker_stream
        self.logger = OnlineLogger(
            self.config.log_path,
            classifier_names=model.classifier_names,
            control_classifier=model.control_name,
        )
        self.pipeline = OnlinePipeline(
            model,
            EMGRejector(EmgRejectConfig(threshold=self.config.emg_threshold)),
        )
        self.gates = {n: DecisionGate(DecisionGateConfig()) for n in model.classifier_names}
        self.gate = self.gates[model.control_name]
        self.avatar = AvatarController(
            AvatarConfig(steps_to_goal=self.config.steps_to_goal)
        )
        self.stream: OnlineStream | None = None
        self.windows: WindowManager | None = None
        self._proc: Any | None = None
        self._eeg: Any | None = None
        self._markers: Any | None = None
        self._rest_start: float | None = None
        self._trial: OnlineTrialLog | None = None
        self._trial_index = 0
        self._imagery_active = False

    @classmethod
    def from_model_path(
        cls,
        path: str | Path,
        config: SimulatorOnlineConfig | None = None,
        **kwargs: Any,
    ) -> "SimulatorOnlineSession":
        return cls(load_frozen_model(path), config, **kwargs)

    def run(self) -> dict[str, Any]:
        """Start the simulator, consume the stream, return the accuracy summary."""
        try:
            self._start()
            self._loop()
        finally:
            self._stop()
            self.logger.close()
        return self.logger.summary(
            offline_accuracy=self.model.offline_accuracy,
            offline_accuracies=self.model.offline_accuracies,
        )

    def _start(self) -> None:
        command = build_simulator_command(
            simulator_project=self._project,
            lsl_name=self.config.lsl_name,
            dataset=self.config.dataset,
            session_dir=self.config.session_dir,
            subject=self.config.subject,
        )
        self._proc = self._process_starter(command)
        self._eeg = self._eeg_opener(
            stream_name=self.config.lsl_name,
            wait_seconds=self.config.wait_seconds,
        )
        summary = self._eeg.summary
        rate = float(summary.nominal_srate or self.model.sample_rate_hz)
        channels = int(summary.channel_count)
        if channels != self.model.channel_count:
            raise ValueError(
                f"Stream has {channels} channels, frozen model expects "
                f"{self.model.channel_count}"
            )
        self.stream = OnlineStream(
            sample_rate_hz=rate,
            n_eeg_channels=channels,
            n_emg_channels=2,
            filter_spec=self.model.filter,
            apply_causal_filter=True,
            keep_seconds=16.0,
        )
        self.windows = WindowManager(self.stream)
        wait = min(2.0, self.config.wait_seconds)
        try:
            try:
                self._markers = self._marker_opener(
                    eeg_name=summary.name, wait_seconds=wait
                )
            except TypeError:
                self._markers = self._marker_opener(
                    eeg_name=self.config.lsl_name, wait_seconds=wait
                )
        except TimeoutError:
            self._markers = None
        if self._markers is None:
            raise RuntimeError(
                "No Markers stream. The online simulator session needs dataset markers."
            )

    def _loop(self) -> None:
        assert self.stream is not None and self.windows is not None
        last_sample_at = time.monotonic()
        while True:
            if self._proc is not None and self._proc.poll() is not None:
                # Drain whatever is left, then finish the open trial.
                self._pull(timeout=0.0)
                self._finish_imagery_if_ready(force=True)
                break
            added = self._pull(timeout=0.1)
            if added:
                last_sample_at = time.monotonic()
            self._finish_imagery_if_ready(force=False)
            if (
                self.config.max_trials is not None
                and self._trial_index >= self.config.max_trials
                and not self._imagery_active
            ):
                break
            if (
                added == 0
                and time.monotonic() - last_sample_at > self.config.idle_timeout_seconds
            ):
                self._finish_imagery_if_ready(force=True)
                break

    def _pull(self, timeout: float) -> int:
        assert self.stream is not None
        added = 0
        if self._eeg is not None:
            chunk = self._eeg.pull_chunk(timeout=timeout, max_samples=2048)
            if chunk.samples:
                self.stream.push_eeg(
                    np.asarray(chunk.samples, dtype=np.float64),
                    chunk.timestamps,
                )
                added = len(chunk.samples)
        if self._markers is not None:
            chunk = self._markers.pull_chunk(timeout=0.0, max_samples=64)
            for raw, stamp in zip(chunk.samples, chunk.timestamps):
                text = raw[0] if isinstance(raw, (list, tuple)) else raw
                kind, label = parse_marker_text(str(text))
                self._on_marker(float(stamp), kind, str(text), label)
        return added

    def _on_marker(self, timestamp: float, kind: str, raw: str, label: str) -> None:
        dataset = self.config.dataset
        if kind == "rest" or label == "rest":
            self._close_rest(timestamp)
            self._rest_start = timestamp
            return
        self._close_rest(timestamp)
        if not is_activity_onset(dataset, kind, label):
            return
        if label not in _HANDS:
            # both_fists / both_feet: keep streaming, do not classify.
            self._finish_imagery_if_ready(force=True)
            return
        self._begin_imagery(timestamp, label)  # type: ignore[arg-type]

    def _close_rest(self, ended_at: float) -> None:
        assert self.stream is not None
        if self._rest_start is None:
            return
        start = self._rest_start
        self._rest_start = None
        if ended_at <= start:
            return
        eeg, _ = self.stream.slice_eeg(start, ended_at, filtered=True)
        if eeg.shape[0] >= 16 and eeg.shape[1] == self.model.channel_count:
            self.pipeline.add_rest(eeg)

    def _begin_imagery(self, onset: float, cue: Direction) -> None:
        assert self.windows is not None
        self._finish_imagery_if_ready(force=True)
        if (
            self.config.max_trials is not None
            and self._trial_index >= self.config.max_trials
        ):
            return
        for gate in self.gates.values():
            gate.reset()
        self.avatar.begin_trial(cue)
        self.windows.begin(onset)
        self._trial = self.logger.start_trial(self._trial_index, cue)
        self._trial_index += 1
        self._imagery_active = True

    def _finish_imagery_if_ready(self, *, force: bool) -> None:
        assert self.windows is not None
        if not self._imagery_active or self._trial is None:
            return
        process_ready_windows(
            pipeline=self.pipeline,
            logger=self.logger,
            gates=self.gates,
            avatar=self.avatar,
            control_name=self.model.control_name,
            trial=self._trial,
            windows=self.windows,
        )
        newest = self.stream.latest_time() if self.stream is not None else None
        activity_done = self.windows.all_complete() or (
            newest is not None and self.windows.activity_ended(newest)
        )
        if force or activity_done or self.avatar.goal_reached:
            fused_by = fused_directions(self.gates)
            fused = fused_by.get(self.model.control_name)
            self.logger.end_trial(
                self._trial,
                fused_label=fused,
                fused_by_classifier=fused_by,
            )
            self._trial = None
            self._imagery_active = False
            self.windows.reset()

    def _stop(self) -> None:
        if self._markers is not None:
            self._markers.close()
            self._markers = None
        if self._eeg is not None:
            self._eeg.close()
            self._eeg = None
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._proc = None
