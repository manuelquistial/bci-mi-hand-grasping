"""Per-window frozen pipeline: bandpass (stream) → EA → CSP → log-var → N heads."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pipeline.alignment.euclidean import alignment_matrix, apply_alignment, trial_covariance
from pipeline.online.emg_reject import EMGRejector
from pipeline.online.model import FrozenOnlineModel
from pipeline.online.types import ClassifierPrediction, WindowDecision
from pipeline.online.windows import ImageryWindow


class OnlinePipeline:
    """Classify imagery windows with a frozen offline model (one or more heads).

    Euclidean alignment uses a reference covariance estimated only from this
    session's rest segments. CSP filters stay frozen. Every classifier head
    scores the same CSP log-variance features.
    """

    def __init__(
        self,
        model: FrozenOnlineModel,
        emg_rejector: EMGRejector | None = None,
        *,
        min_rest_segments: int = 1,
    ) -> None:
        self.model = model
        self.emg_rejector = emg_rejector or EMGRejector()
        self.min_rest_segments = int(min_rest_segments)
        self._rest_trials: list[np.ndarray] = []
        self._ea_weights: np.ndarray | None = None

    def reset_session(self) -> None:
        self._rest_trials = []
        self._ea_weights = None

    def add_rest(self, eeg_samples_by_time: np.ndarray) -> np.ndarray:
        """Incorporate one rest segment ``(samples, channels)`` into the EA reference."""
        array = np.asarray(eeg_samples_by_time, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 2:
            raise ValueError("Rest segment needs at least two samples")
        if array.shape[1] != self.model.channel_count:
            raise ValueError(
                f"Rest has {array.shape[1]} channels, model expects {self.model.channel_count}"
            )
        trial = array.T
        self._rest_trials.append(trial)
        self._ea_weights = alignment_matrix(self._rest_trials)
        return self._ea_weights

    @property
    def ea_ready(self) -> bool:
        return (
            self._ea_weights is not None
            and len(self._rest_trials) >= self.min_rest_segments
        )

    def process_window(self, window: ImageryWindow) -> WindowDecision:
        """EMG gate first; then EA → CSP once → every frozen classifier head."""
        skipped = self._skip_decision(window)
        if skipped is not None:
            return skipped
        assert self._ea_weights is not None
        aligned = apply_alignment(window.eeg, self._ea_weights)
        trial = aligned.T
        predictions: list[ClassifierPrediction] = []
        for name, label, score in self.model.predict_all(trial):
            if label not in {"left", "right"}:
                predictions.append(
                    ClassifierPrediction(classifier=name, label=None, score=score)
                )
                continue
            predictions.append(
                ClassifierPrediction(
                    classifier=name,
                    label=label,  # type: ignore[arg-type]
                    score=score,
                )
            )
        control = next(
            (
                item
                for item in predictions
                if item.classifier == self.model.control_name
            ),
            predictions[0] if predictions else None,
        )
        if control is None:
            return WindowDecision(
                window_index=window.index,
                start=window.start,
                end=window.end,
                label=None,
                score=None,
                emg_rejected=False,
                reason="no_predictions",
                predictions=(),
            )
        reason = "classified" if control.label is not None else "unexpected_label"
        return WindowDecision(
            window_index=window.index,
            start=window.start,
            end=window.end,
            label=control.label,
            score=control.score,
            emg_rejected=False,
            reason=reason,
            classifier=control.classifier,
            predictions=tuple(predictions),
        )

    def aligned_trial(self, window: ImageryWindow) -> np.ndarray | None:
        """Return EA-aligned ``(channels, samples)`` for probability mapping, if ready."""
        if self._ea_weights is None or not self.ea_ready:
            return None
        if window.eeg.ndim != 2 or window.eeg.shape[0] < 16:
            return None
        return apply_alignment(window.eeg, self._ea_weights).T

    def decisions_by_classifier(self, decision: WindowDecision) -> dict[str, WindowDecision]:
        """Expand one multi-head window into one ``WindowDecision`` per classifier."""
        if not decision.predictions:
            name = decision.classifier or self.model.control_name
            return {name: decision}
        out: dict[str, WindowDecision] = {}
        for pred in decision.predictions:
            out[pred.classifier] = WindowDecision(
                window_index=decision.window_index,
                start=decision.start,
                end=decision.end,
                label=pred.label,
                score=pred.score,
                emg_rejected=decision.emg_rejected,
                reason=decision.reason if pred.label is not None else "unexpected_label",
                classifier=pred.classifier,
                predictions=decision.predictions,
            )
        return out

    def process_windows(self, windows: Sequence[ImageryWindow]) -> list[WindowDecision]:
        return [self.process_window(window) for window in windows]

    def rest_reference_covariance(self) -> np.ndarray | None:
        if not self._rest_trials:
            return None
        return sum(trial_covariance(trial) for trial in self._rest_trials) / len(
            self._rest_trials
        )

    def _skip_decision(self, window: ImageryWindow) -> WindowDecision | None:
        empty_preds = tuple(
            ClassifierPrediction(classifier=name, label=None, score=None)
            for name in self.model.classifier_names
        )
        verdict = self.emg_rejector.check(window)
        if verdict.rejected:
            return WindowDecision(
                window_index=window.index,
                start=window.start,
                end=window.end,
                label=None,
                score=None,
                emg_rejected=True,
                reason=verdict.reason,
                classifier=self.model.control_name,
                predictions=empty_preds,
            )
        if not self.ea_ready or self._ea_weights is None:
            return WindowDecision(
                window_index=window.index,
                start=window.start,
                end=window.end,
                label=None,
                score=None,
                emg_rejected=False,
                reason="ea_not_ready",
                classifier=self.model.control_name,
                predictions=empty_preds,
            )
        if window.eeg.shape[0] < 16:
            return WindowDecision(
                window_index=window.index,
                start=window.start,
                end=window.end,
                label=None,
                score=None,
                emg_rejected=False,
                reason="short_window",
                classifier=self.model.control_name,
                predictions=empty_preds,
            )
        return None
