"""Frozen online model types: shared filter/CSP plus linear classifier heads.

Supported heads: ``logistic_regression`` and ``linear_svm`` — the same
estimators built by :func:`pipeline.classify.models.build_model`.

Shared front-end (filter + CSP) runs once per window; every head scores the
same log-variance features with its own scaler and classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pipeline.features.csp import log_variances
from pipeline.preprocess import FilterSpec

ONLINE_CLASSIFIERS = ("logistic_regression", "linear_svm")


@dataclass(frozen=True)
class FrozenClassifierHead:
    """One frozen classifier sitting on shared CSP features."""

    name: str
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    classifier: Any
    class_labels: tuple[str, ...]
    offline_accuracy: float | None = None


@dataclass(frozen=True)
class FrozenOnlineModel:
    """Shared filter/CSP plus one or more frozen classifier heads."""

    sample_rate_hz: float
    filter: FilterSpec
    csp_filters: np.ndarray  # (channels, components)
    heads: tuple[FrozenClassifierHead, ...]
    channel_count: int
    csp_pairs: int
    meta: dict[str, Any] | None = None
    control_classifier: str | None = None

    def __post_init__(self) -> None:
        if not self.heads:
            raise ValueError("Frozen model needs at least one classifier head")
        names = [head.name for head in self.heads]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate classifier heads: {names}")
        if self.control_classifier is not None and self.control_classifier not in names:
            raise ValueError(
                f"control_classifier {self.control_classifier!r} is not in {names}"
            )

    @property
    def classifier_names(self) -> tuple[str, ...]:
        return tuple(head.name for head in self.heads)

    @property
    def control_name(self) -> str:
        return self.control_classifier or self.heads[0].name

    def head(self, name: str | None = None) -> FrozenClassifierHead:
        target = name or self.control_name
        for item in self.heads:
            if item.name == target:
                return item
        raise KeyError(target)

    # --- backward-compatible single-head accessors ---
    @property
    def classifier_name(self) -> str:
        return self.control_name

    @property
    def classifier(self) -> Any:
        return self.head().classifier

    @property
    def class_labels(self) -> tuple[str, ...]:
        return self.head().class_labels

    @property
    def scaler_mean(self) -> np.ndarray:
        return self.head().scaler_mean

    @property
    def scaler_scale(self) -> np.ndarray:
        return self.head().scaler_scale

    @property
    def offline_accuracy(self) -> float | None:
        return self.head().offline_accuracy

    @property
    def offline_accuracies(self) -> dict[str, float | None]:
        return {head.name: head.offline_accuracy for head in self.heads}

    def raw_features(self, trial_channels_by_time: np.ndarray) -> np.ndarray:
        """CSP log-variances before per-head scaling."""
        return log_variances(trial_channels_by_time, self.csp_filters)

    def feature_vector(
        self,
        trial_channels_by_time: np.ndarray,
        *,
        classifier_name: str | None = None,
    ) -> np.ndarray:
        """Scaled feature row for one head."""
        head = self.head(classifier_name)
        features = self.raw_features(trial_channels_by_time)
        scaled = (features - head.scaler_mean) / head.scaler_scale
        return scaled.reshape(1, -1)

    def predict(
        self,
        trial_channels_by_time: np.ndarray,
        *,
        classifier_name: str | None = None,
    ) -> tuple[str, float]:
        """Predict with one head (control head by default)."""
        head = self.head(classifier_name)
        row = self.feature_vector(trial_channels_by_time, classifier_name=head.name)
        label = str(head.classifier.predict(row)[0])
        score = _decision_score(head.classifier, row, label)
        return label, score

    def predict_all(
        self,
        trial_channels_by_time: np.ndarray,
    ) -> list[tuple[str, str, float]]:
        """Score every head. Returns ``(classifier_name, label, score)``."""
        features = self.raw_features(trial_channels_by_time)
        out: list[tuple[str, str, float]] = []
        for head in self.heads:
            row = ((features - head.scaler_mean) / head.scaler_scale).reshape(1, -1)
            label = str(head.classifier.predict(row)[0])
            score = _decision_score(head.classifier, row, label)
            out.append((head.name, label, score))
        return out

    def predict_proba_pair(
        self,
        trial_channels_by_time: np.ndarray,
        *,
        classifier_name: str | None = None,
    ) -> tuple[float, float] | None:
        """Return ``(p_left, p_right)`` in ``[0, 1]`` for the game control adapter."""
        head = self.head(classifier_name)
        row = self.feature_vector(trial_channels_by_time, classifier_name=head.name)
        classes = [str(item) for item in getattr(head.classifier, "classes_", [])]
        proba_fn = getattr(head.classifier, "predict_proba", None)
        if callable(proba_fn):
            scores = np.asarray(proba_fn(row), dtype=np.float64).reshape(-1)
            by_class = {
                name: float(value) for name, value in zip(classes, scores, strict=False)
            }
            if "left" in by_class and "right" in by_class:
                return by_class["left"], by_class["right"]
        decision = getattr(head.classifier, "decision_function", None)
        if callable(decision) and len(classes) == 2:
            value = float(np.asarray(decision(row), dtype=np.float64).reshape(-1)[0])
            # Positive decision_function maps to classes_[1] in sklearn binary SVM/LR.
            p_second = float(1.0 / (1.0 + np.exp(-value)))
            p_first = 1.0 - p_second
            by_class = {classes[0]: p_first, classes[1]: p_second}
            if "left" in by_class and "right" in by_class:
                return by_class["left"], by_class["right"]
        return None


def parse_classifier_names(
    names: Sequence[str] | str | None = None,
    *,
    classifier_name: str | None = None,
) -> tuple[str, ...]:
    """Normalize ``--classifiers a,b`` / single name into a unique ordered tuple."""
    collected: list[str] = []
    if isinstance(names, str):
        collected.extend(part.strip() for part in names.split(",") if part.strip())
    elif names is not None:
        collected.extend(str(item).strip() for item in names if str(item).strip())
    if classifier_name:
        collected.append(str(classifier_name).strip())
    if not collected:
        collected = ["linear_svm"]
    unique: list[str] = []
    for name in collected:
        if name not in ONLINE_CLASSIFIERS:
            raise ValueError(
                f"Online freeze supports {ONLINE_CLASSIFIERS}, got {name!r}"
            )
        if name not in unique:
            unique.append(name)
    return tuple(unique)


def _decision_score(model: Any, row: np.ndarray, label: str) -> float:
    decision = getattr(model, "decision_function", None)
    if callable(decision):
        values = np.asarray(decision(row), dtype=np.float64).reshape(-1)
        if values.size == 1:
            classes = [str(item) for item in getattr(model, "classes_", [])]
            if len(classes) == 2 and label == classes[0]:
                return float(-values[0])
            return float(values[0])
        classes = [str(item) for item in getattr(model, "classes_", [])]
        if label in classes:
            return float(values[classes.index(label)])
        return float(np.max(values))
    proba = getattr(model, "predict_proba", None)
    if callable(proba):
        scores = np.asarray(proba(row), dtype=np.float64).reshape(-1)
        classes = [str(item) for item in getattr(model, "classes_", [])]
        if label in classes:
            return float(scores[classes.index(label)])
        return float(np.max(scores))
    return 0.0
