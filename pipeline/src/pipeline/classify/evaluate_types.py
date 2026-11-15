"""Value objects shared by the classifier evaluation modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCHEMES = ("split", "cross_validation", "loso", "nested", "nested_loso")


@dataclass(frozen=True)
class SubjectTrials:
    """Imagery trials of one participant, already filtered and aligned."""

    subject: str
    trials: list[np.ndarray]
    labels: list[str]
    dataset_id: str = ""
    session_id: str = ""
    trial_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FoldScore:
    name: str
    accuracy: float


@dataclass(frozen=True)
class Prediction:
    accuracy: float
    y_true: np.ndarray
    y_pred: np.ndarray
    scores: np.ndarray | None
    score_labels: list[str]
    predict_seconds: float = 0.0


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    detail: str
    folds: list[FoldScore]
    labels: list[str]
    matrix: list[list[int]]
    metrics: dict[str, float | None]
    report: dict | None = None
