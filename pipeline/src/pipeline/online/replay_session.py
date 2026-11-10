"""Online pipeline against a dataset file (no LSL) — for Slurm / headless hosts.

Loads continuous EEG + markers through the simulator dump path, feeds
:class:`OnlineStream`, and reuses the same windowing / EA / CSP / gate stack
as the live simulator session.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pipeline.epochs import is_activity_onset
from pipeline.online.avatar import AvatarConfig, AvatarController
from pipeline.online.emg_reject import EMGRejector, EmgRejectConfig
from pipeline.online.gate import DecisionGate, DecisionGateConfig
from pipeline.online.log import OnlineLogger
from pipeline.online.model import FrozenOnlineModel, load_frozen_model
from pipeline.online.pipeline import OnlinePipeline
from pipeline.online.stream import OnlineStream
from pipeline.online.types import Direction, OnlineTrialLog
from pipeline.online.windows import WindowManager
from pipeline.service import default_simulator_project

_HANDS = frozenset({"left", "right"})
Dumper = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ReplayOnlineConfig:
    dataset: str = "shuqfa2024"
    subject: str | None = None
    session_dir: str | None = None
    max_trials: int | None = None
    emg_threshold: float = 50.0
    steps_to_goal: int = 4
    log_path: str | Path | None = None
    chunk_samples: int = 64
    dump_seconds: float = 200.0


class ReplayOnlineSession:
    """Classify a recorded dataset subject without starting LSL."""

    def __init__(
        self,
        model: FrozenOnlineModel,
        config: ReplayOnlineConfig | None = None,
        *,
        simulator_project: Path | None = None,
        stream_loader: Dumper | None = None,
        marker_loader: Dumper | None = None,
    ) -> None:
        self.model = model
        self.config = config or ReplayOnlineConfig()
        self._project = Path(simulator_project or default_simulator_project())
        self._stream_loader = stream_loader or self._dump_stream
        self._marker_loader = marker_loader or self._dump_markers
        self.logger = OnlineLogger(
            self.config.log_path,
            classifier_names=model.classifier_names,
            control_classifier=model.control_name,
        )
        self.pipeline = OnlinePipeline(
            model,
            EMGRejector(EmgRejectConfig(threshold=self.config.emg_threshold)),
        )
        self.gates = {
            name: DecisionGate(DecisionGateConfig()) for name in model.classifier_names
        }
        self.avatar = AvatarController(
            AvatarConfig(steps_to_goal=self.config.steps_to_goal)
        )

    @classmethod
    def from_model_path(
        cls,
        path: str | Path,
        config: ReplayOnlineConfig | None = None,
        **kwargs: Any,
    ) -> "ReplayOnlineSession":
        return cls(load_frozen_model(path), config, **kwargs)

    def run(self) -> dict[str, Any]:
        stream_bundle = self._stream_loader(
            dataset=self.config.dataset,
            subject=self.config.subject,
            session_dir=self.config.session_dir,
            seconds=self.config.dump_seconds,
        )
        marker_bundle = self._marker_loader(
            dataset=self.config.dataset,
            subject=self.config.subject,
            session_dir=self.config.session_dir,
        )
        samples = np.asarray(stream_bundle["samples"], dtype=np.float64)
        rate = float(stream_bundle.get("sample_rate_hz") or self.model.sample_rate_hz)
        if samples.ndim != 2 or samples.shape[0] < 16:
            raise RuntimeError("Replay stream has no EEG samples")
        if samples.shape[1] != self.model.channel_count:
            raise ValueError(
                f"Replay has {samples.shape[1]} channels, model expects "
                f"{self.model.channel_count}"
            )
        markers = [
            {
                "onset": float(item.get("onset") or 0.0),
                "kind": str(item.get("kind") or ""),
                "label": str(item.get("label") or ""),
            }
            for item in marker_bundle.get("trials") or []
        ]
        markers.sort(key=lambda item: item["onset"])

        stream = OnlineStream(
            sample_rate_hz=rate,
            n_eeg_channels=int(samples.shape[1]),
            n_emg_channels=2,
            filter_spec=self.model.filter,
            apply_causal_filter=True,
            keep_seconds=16.0,
        )
        windows = WindowManager(stream)
        trial: OnlineTrialLog | None = None
        trial_index = 0
        imagery_active = False
        rest_start: float | None = None
        marker_i = 0
        chunk = max(1, int(self.config.chunk_samples))

        try:
            for start in range(0, samples.shape[0], chunk):
                block = samples[start : start + chunk]
                times = [(start + offset) / rate for offset in range(block.shape[0])]
                stream.push_eeg(block, times)
                # Quiet EMG so the rejector stays open on dataset replay.
                stream.push_emg(np.zeros((block.shape[0], 2), dtype=np.float64), times)
                newest = times[-1]
                while marker_i < len(markers) and markers[marker_i]["onset"] <= newest:
                    marker = markers[marker_i]
                    marker_i += 1
                    kind = marker["kind"]
                    label = marker["label"]
                    onset = marker["onset"]
                    if kind == "rest" or label == "rest":
                        if imagery_active and trial is not None:
                            imagery_active = self._end_trial(
                                trial, windows, force=True
                            )
                            trial = None
                        if rest_start is not None:
                            self._close_rest(stream, rest_start, onset)
                        rest_start = onset
                        continue
                    if rest_start is not None:
                        self._close_rest(stream, rest_start, onset)
                        rest_start = None
                    if not is_activity_onset(self.config.dataset, kind, label):
                        continue
                    if label not in _HANDS:
                        if imagery_active and trial is not None:
                            imagery_active = self._end_trial(
                                trial, windows, force=True
                            )
                            trial = None
                        continue
                    if (
                        self.config.max_trials is not None
                        and trial_index >= self.config.max_trials
                    ):
                        continue
                    if imagery_active and trial is not None:
                        self._end_trial(trial, windows, force=True)
                    for gate in self.gates.values():
                        gate.reset()
                    self.avatar.begin_trial(label)  # type: ignore[arg-type]
                    windows.begin(onset)
                    trial = self.logger.start_trial(trial_index, label)
                    trial_index += 1
                    imagery_active = True
                if imagery_active and trial is not None:
                    still = self._poll_imagery(trial, windows)
                    if not still:
                        imagery_active = False
                        trial = None
                if (
                    self.config.max_trials is not None
                    and trial_index >= self.config.max_trials
                    and not imagery_active
                ):
                    break
            if imagery_active and trial is not None:
                self._end_trial(trial, windows, force=True)
            if rest_start is not None and stream.latest_time() is not None:
                self._close_rest(stream, rest_start, float(stream.latest_time()))
        finally:
            self.logger.close()
        return self.logger.summary(
            offline_accuracy=self.model.offline_accuracy,
            offline_accuracies=self.model.offline_accuracies,
        )

    def _poll_imagery(self, trial: OnlineTrialLog, windows: WindowManager) -> bool:
        for window in windows.poll_ready():
            decision = self.pipeline.process_window(window)
            self.logger.log_window(trial, decision)
            per_head = self.pipeline.decisions_by_classifier(decision)
            for name, head_decision in per_head.items():
                gate = self.gates[name].update(head_decision)
                self.logger.log_gate(trial, gate, classifier=name)
            avatar = self.avatar.apply(self.gates[self.model.control_name].state())
            self.logger.log_avatar(trial, avatar)
        if windows.all_complete() or self.avatar.goal_reached:
            return self._end_trial(trial, windows, force=True)
        return True

    def _end_trial(
        self,
        trial: OnlineTrialLog,
        windows: WindowManager,
        *,
        force: bool,
    ) -> bool:
        del force
        for window in windows.poll_ready():
            decision = self.pipeline.process_window(window)
            self.logger.log_window(trial, decision)
            per_head = self.pipeline.decisions_by_classifier(decision)
            for name, head_decision in per_head.items():
                gate = self.gates[name].update(head_decision)
                self.logger.log_gate(trial, gate, classifier=name)
            self.avatar.apply(self.gates[self.model.control_name].state())
        fused_by = {name: gate.state().direction for name, gate in self.gates.items()}
        self.logger.end_trial(
            trial,
            fused_label=fused_by.get(self.model.control_name),
            fused_by_classifier=fused_by,
        )
        windows.reset()
        return False

    def _close_rest(self, stream: OnlineStream, start: float, end: float) -> None:
        if end <= start:
            return
        eeg, _ = stream.slice_eeg(start, end, filtered=True)
        if eeg.shape[0] >= 16 and eeg.shape[1] == self.model.channel_count:
            self.pipeline.add_rest(eeg)

    def _dump_stream(
        self,
        *,
        dataset: str,
        subject: str | None,
        session_dir: str | None,
        seconds: float,
    ) -> dict[str, Any]:
        command = [
            "uv",
            "run",
            "--project",
            str(self._project),
            "simopenbci",
            "--no-dialog",
            "--dataset",
            dataset,
            "--dump-seconds",
            str(seconds),
        ]
        if session_dir:
            command.extend(["--file", session_dir])
        if subject:
            command.extend(["--subject", subject])
        completed = subprocess.run(
            command,
            cwd=self._project,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "dump-seconds failed").strip()
            raise RuntimeError(detail.splitlines()[-1])
        data = json.loads(completed.stdout or "{}")
        if not isinstance(data, dict) or not data.get("samples"):
            raise RuntimeError("dump-seconds returned no samples")
        return data

    def _dump_markers(
        self,
        *,
        dataset: str,
        subject: str | None,
        session_dir: str | None,
    ) -> dict[str, Any]:
        command = [
            "uv",
            "run",
            "--project",
            str(self._project),
            "simopenbci",
            "--no-dialog",
            "--dataset",
            dataset,
            "--dump-trials",
            "--trial-seconds",
            "4",
        ]
        if session_dir:
            command.extend(["--file", session_dir])
        if subject:
            command.extend(["--subject", subject])
        completed = subprocess.run(
            command,
            cwd=self._project,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "dump-trials failed").strip()
            raise RuntimeError(detail.splitlines()[-1])
        data = json.loads(completed.stdout or "{}")
        if not isinstance(data, dict):
            raise RuntimeError("dump-trials returned invalid JSON")
        return data
