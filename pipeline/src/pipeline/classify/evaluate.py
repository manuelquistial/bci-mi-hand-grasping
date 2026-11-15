"""Train, validation, and test, or cross-validation, or leave-one-subject-out.

Supervised features are fit inside each training split. The test split never
enters that fit.
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from pipeline.classify.evaluate_types import (
    SCHEMES,
    Evaluation,
    FoldScore,
    Prediction,
    SubjectTrials,
)
from pipeline.classify.fold_scoring import _in_order, _score, _score_fold, _with_metrics
from pipeline.classify.models import CLASSIFIERS
from pipeline.params import ModelParams

__all__ = [
    "SCHEMES",
    "Evaluation",
    "FoldScore",
    "Prediction",
    "SubjectTrials",
    "_score",
    "_with_metrics",
    "evaluate_classifier",
]

_SEED = 42



def evaluate_classifier(
    classifier: str,
    scheme: str,
    feature: str,
    groups: Sequence[SubjectTrials],
    *,
    sample_rate_hz: float,
    channels: Sequence[str],
    params: ModelParams | None = None,
    workers: int = 1,
    manifest: dict | None = None,
) -> Evaluation:
    """Score one classifier with the chosen evaluation.

    ``workers`` above 1 runs folds side by side on the shared trials. The
    pre-model request leaves this at 1, so that path stays single-threaded.
    """
    name = classifier.strip().lower()
    if name not in CLASSIFIERS or name == "none":
        raise ValueError(f"Unknown classifier {classifier!r}")
    kind = scheme.strip().lower()
    if kind not in SCHEMES:
        raise ValueError(f"Unknown evaluation {scheme!r}")
    if feature == "none":
        raise ValueError("Choose a feature method before the classifier")
    tuned = params or ModelParams()
    if kind in {"nested", "nested_loso"}:
        from pipeline.classify.nested import evaluate_nested

        return evaluate_nested(
            name,
            feature,
            groups,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            params=tuned,
            workers=workers,
            outer="loso" if kind == "nested_loso" else "group",
            manifest=manifest,
        )
    if kind == "loso":
        return _loso(name, feature, groups, sample_rate_hz, channels, tuned, workers)
    group = _combine(groups)
    if kind == "split":
        return _split(
            name,
            feature,
            group,
            sample_rate_hz,
            channels,
            participants=len(groups),
            params=tuned,
            workers=workers,
        )
    return _cross_validate(
        name,
        feature,
        group,
        sample_rate_hz,
        channels,
        participants=len(groups),
        params=tuned,
        workers=workers,
    )


def _combine(groups: Sequence[SubjectTrials]) -> SubjectTrials:
    """Pool every participant. Alignment, when used, is already per person."""
    if len(groups) == 1:
        return groups[0]
    trials: list[np.ndarray] = []
    labels: list[str] = []
    for group in groups:
        trials.extend(group.trials)
        labels.extend(group.labels)
    return SubjectTrials(f"{len(groups)} participants", trials, labels)


def _split(
    classifier: str,
    feature: str,
    group: SubjectTrials,
    rate: float,
    channels: Sequence[str],
    participants: int = 1,
    params: ModelParams | None = None,
    workers: int = 1,
) -> Evaluation:
    tuned = params or ModelParams()
    test_fraction = float(tuned.test_fraction)
    validation_fraction = float(tuned.validation_fraction)
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1 - test_fraction:
        raise ValueError("Test and validation fractions must leave room for training")
    labels = np.asarray(group.labels)
    index = np.arange(labels.size)
    try:
        rest, test = train_test_split(
            index, test_size=test_fraction, stratify=labels, random_state=_SEED
        )
        train, validation = train_test_split(
            rest,
            test_size=validation_fraction / (1.0 - test_fraction),
            stratify=labels[rest],
            random_state=_SEED,
        )
    except ValueError as exc:
        raise ValueError(
            "Train, validation, and test need at least three trials of each class"
        ) from exc
    pieces = (("train", train), ("validation", validation), ("test", test))
    scored = _in_order(
        pieces,
        workers,
        lambda item: _score_fold(
            classifier, feature, group, train, item[1], rate, channels, tuned, workers
        ),
    )
    model_train, held_validation, held_test = scored
    folds = [
        FoldScore("train", model_train.accuracy),
        FoldScore("validation", held_validation.accuracy),
        FoldScore("test", held_test.accuracy),
    ]
    cohort = f"{participants} participants · " if participants > 1 else ""
    detail = (
        f"{cohort}test {held_test.accuracy:.0%} · validation {held_validation.accuracy:.0%} · "
        f"train {model_train.accuracy:.0%}"
    )
    return _with_metrics(held_test.accuracy, detail, folds, [held_test])


def _cross_validate(
    classifier: str,
    feature: str,
    group: SubjectTrials,
    rate: float,
    channels: Sequence[str],
    participants: int = 1,
    params: ModelParams | None = None,
    workers: int = 1,
) -> Evaluation:
    tuned = params or ModelParams()
    labels = np.asarray(group.labels)
    smallest = min(int(np.sum(labels == label)) for label in set(group.labels))
    requested = int(tuned.folds)
    if requested < 2:
        raise ValueError("Cross-validation needs at least two folds")
    folds_n = min(requested, smallest)
    if folds_n < 2:
        raise ValueError("Cross-validation needs at least two trials of each class")
    splitter = StratifiedKFold(n_splits=folds_n, shuffle=True, random_state=_SEED)
    index = np.arange(labels.size)
    pairs = list(enumerate(splitter.split(index, labels), start=1))

    def run(item: tuple[int, tuple[np.ndarray, np.ndarray]]) -> tuple[FoldScore, Prediction]:
        number, (train, test) = item
        predicted = _score_fold(
            classifier, feature, group, train, test, rate, channels, tuned, workers
        )
        return FoldScore(f"fold {number}", predicted.accuracy), predicted

    scored = _in_order(pairs, workers, run)
    scores = [item[0] for item in scored]
    predictions = [item[1] for item in scored]
    mean = float(np.mean([item.accuracy for item in scores]))
    spread = float(np.std([item.accuracy for item in scores]))
    cohort = f"{participants} participants · " if participants > 1 else ""
    detail = f"{folds_n}-fold · {cohort}{mean:.0%} ± {spread:.0%}"
    return _with_metrics(mean, detail, scores, predictions)


def _loso(
    classifier: str,
    feature: str,
    groups: Sequence[SubjectTrials],
    rate: float,
    channels: Sequence[str],
    params: ModelParams | None = None,
    workers: int = 1,
) -> Evaluation:
    tuned = params or ModelParams()
    if len(groups) < 2:
        raise ValueError("LOSO needs at least two participants")
    jobs: list[tuple[str, SubjectTrials, np.ndarray, np.ndarray]] = []
    for held in groups:
        train_trials: list[np.ndarray] = []
        train_labels: list[str] = []
        for group in groups:
            if group.subject == held.subject:
                continue
            train_trials.extend(group.trials)
            train_labels.extend(group.labels)
        if len(set(train_labels)) < 2 or not held.labels:
            continue
        combined = SubjectTrials(
            "combined",
            train_trials + held.trials,
            train_labels + held.labels,
        )
        train_index = np.arange(len(train_labels))
        test_index = np.arange(len(train_labels), len(train_labels) + len(held.labels))
        jobs.append((held.subject, combined, train_index, test_index))

    def run(
        job: tuple[str, SubjectTrials, np.ndarray, np.ndarray],
    ) -> tuple[FoldScore, Prediction]:
        subject, combined, train_index, test_index = job
        predicted = _score_fold(
            classifier, feature, combined, train_index, test_index, rate, channels, tuned, workers
        )
        return FoldScore(subject, predicted.accuracy), predicted

    scored = _in_order(jobs, workers, run)
    scores = [item[0] for item in scored]
    predictions = [item[1] for item in scored]
    if not scores:
        raise ValueError("LOSO could not score any held-out participant")
    mean = float(np.mean([item.accuracy for item in scores]))
    detail = f"{len(scores)} participants · {mean:.0%}"
    return _with_metrics(mean, detail, scores, predictions)


