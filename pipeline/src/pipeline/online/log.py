"""JSONL logging of window decisions, EMG rejections, and gate states."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from pipeline.online.types import (
    AvatarState,
    GateState,
    OnlineTrialLog,
    WindowDecision,
)


class OnlineLogger:
    """Append-only session log. Online accuracy is tracked per classifier."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        classifier_names: tuple[str, ...] | list[str] | None = None,
        control_classifier: str | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._handle: TextIO | None = None
        self._trials: list[OnlineTrialLog] = []
        self._names = tuple(classifier_names or ())
        self._control = control_classifier or (self._names[0] if self._names else None)
        self._correct = 0
        self._scored = 0
        self._correct_by: dict[str, int] = defaultdict(int)
        self._scored_by: dict[str, int] = defaultdict(int)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def start_trial(self, trial_index: int, cue: str) -> OnlineTrialLog:
        trial = OnlineTrialLog(trial_index=trial_index, cue=cue)  # type: ignore[arg-type]
        self._trials.append(trial)
        self._write({"event": "trial_start", "trial_index": trial_index, "cue": cue})
        return trial

    def log_window(self, trial: OnlineTrialLog, decision: WindowDecision) -> None:
        trial.window_decisions.append(decision)
        self._write(
            {
                "event": "window",
                "trial_index": trial.trial_index,
                "decision": asdict(decision),
            }
        )

    def log_gate(
        self,
        trial: OnlineTrialLog,
        gate: GateState,
        *,
        classifier: str | None = None,
    ) -> None:
        trial.gate_states.append(gate)
        payload: dict[str, Any] = {
            "event": "gate",
            "trial_index": trial.trial_index,
            "gate": asdict(gate),
        }
        if classifier is not None:
            payload["classifier"] = classifier
        self._write(payload)

    def log_avatar(self, trial: OnlineTrialLog, avatar: AvatarState) -> None:
        trial.avatar_steps.append(avatar)
        self._write(
            {
                "event": "avatar",
                "trial_index": trial.trial_index,
                "avatar": asdict(avatar),
            }
        )

    def end_trial(
        self,
        trial: OnlineTrialLog,
        *,
        fused_label: str | None,
        fused_by_classifier: dict[str, str | None] | None = None,
    ) -> None:
        trial.fused_label = fused_label  # type: ignore[assignment]
        by_clf = fused_by_classifier or {}
        if fused_label in {"left", "right"}:
            trial.correct = fused_label == trial.cue
            self._scored += 1
            if trial.correct:
                self._correct += 1
        trial.fused_by_classifier = {
            name: (label if label in {"left", "right"} else None)  # type: ignore[misc]
            for name, label in by_clf.items()
        }
        for name, label in by_clf.items():
            if label not in {"left", "right"}:
                trial.correct_by_classifier[name] = None
                continue
            ok = label == trial.cue
            trial.correct_by_classifier[name] = ok
            self._scored_by[name] += 1
            if ok:
                self._correct_by[name] += 1
        self._write(
            {
                "event": "trial_end",
                "trial_index": trial.trial_index,
                "cue": trial.cue,
                "fused_label": fused_label,
                "fused_by_classifier": by_clf,
                "correct": trial.correct,
                "correct_by_classifier": trial.correct_by_classifier,
                "online_accuracy": self.online_accuracy,
                "online_accuracy_by_classifier": self.online_accuracy_by_classifier,
            }
        )

    @property
    def online_accuracy(self) -> float | None:
        if self._scored == 0:
            return None
        return self._correct / self._scored

    @property
    def online_accuracy_by_classifier(self) -> dict[str, float | None]:
        names = self._names or tuple(self._scored_by.keys())
        out: dict[str, float | None] = {}
        for name in names:
            scored = self._scored_by.get(name, 0)
            out[name] = None if scored == 0 else self._correct_by[name] / scored
        return out

    @property
    def trials(self) -> list[OnlineTrialLog]:
        return list(self._trials)

    def summary(
        self,
        *,
        offline_accuracy: float | None = None,
        offline_accuracies: dict[str, float | None] | None = None,
    ) -> dict[str, Any]:
        by_clf: dict[str, Any] = {}
        offline = offline_accuracies or {}
        names = self._names or tuple(
            set(self._scored_by) | set(offline) | set(self.online_accuracy_by_classifier)
        )
        for name in names:
            by_clf[name] = {
                "online_accuracy": self.online_accuracy_by_classifier.get(name),
                "offline_accuracy": offline.get(name),
                "n_scored": self._scored_by.get(name, 0),
                "n_correct": self._correct_by.get(name, 0),
            }
        return {
            "n_trials": len(self._trials),
            "n_scored": self._scored,
            "n_correct": self._correct,
            "control_classifier": self._control,
            "online_accuracy": self.online_accuracy,
            "offline_accuracy": offline_accuracy,
            "classifiers": by_clf,
        }

    def _write(self, payload: dict[str, Any]) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(payload, default=_json_default) + "\n")
        self._handle.flush()


def _json_default(value: Any) -> Any:
    if isinstance(value, float):
        return value
    return str(value)
