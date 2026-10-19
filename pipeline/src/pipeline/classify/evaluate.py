"""Train, validation, and test, or cross-validation, or leave-one-subject-out.

Supervised features are fit inside each training split. The test split never
enters that fit.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline.classify.metrics import summarize
from pipeline.classify.models import CLASSIFIERS, build_model, scales_features
from pipeline.params import ModelParams
from pipeline.classify.riemann import TangentEncoder
from pipeline.features.encode import FeatureEncoder, make_encoder

SCHEMES = ("split", "cross_validation", "loso", "nested", "nested_loso")
_SEED = 42


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
    if name != "riemann" and feature == "none":
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


_Item = TypeVar("_Item")
_Out = TypeVar("_Out")


def _in_order(
    items: Sequence[_Item],
    workers: int,
    function: Callable[[_Item], _Out],
) -> list[_Out]:
    """Run ``function`` on each item. More than one worker keeps the input order."""
    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(int(workers), len(items))) as pool:
        return list(pool.map(function, items))


def _score_fold(
    classifier: str,
    feature: str,
    group: SubjectTrials,
    train_index: np.ndarray,
    test_index: np.ndarray,
    rate: float,
    channels: Sequence[str],
    params: ModelParams,
    workers: int,
) -> Prediction:
    """One fold. Parallel folds each keep a single BLAS thread so cores do not pile up."""
    if workers <= 1:
        return _score(classifier, feature, group, train_index, test_index, rate, channels, params)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _score(classifier, feature, group, train_index, test_index, rate, channels, params)


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


def _score(
    classifier: str,
    feature: str,
    group: SubjectTrials,
    train_index: np.ndarray,
    test_index: np.ndarray,
    rate: float,
    channels: Sequence[str],
    params: ModelParams | None = None,
) -> Prediction:
    tuned = params or ModelParams()
    train_trials = [group.trials[int(index)] for index in train_index]
    train_labels = [group.labels[int(index)] for index in train_index]
    test_trials = [group.trials[int(index)] for index in test_index]
    test_labels = [group.labels[int(index)] for index in test_index]
    if len(set(train_labels)) < 2:
        raise ValueError("Training fold has only one class")
    encoder: FeatureEncoder | TangentEncoder
    if classifier == "riemann":
        encoder = TangentEncoder()
    else:
        encoder = make_encoder(feature, rate, channels, tuned)
    encoder.fit(train_trials, train_labels)
    learned = encoder.transform(train_trials)
    held = encoder.transform(test_trials)
    if scales_features(classifier):
        scaler = StandardScaler()
        learned = scaler.fit_transform(learned)
        held = scaler.transform(held)
    model = build_model(classifier, len(train_labels), tuned)
    try:
        model.fit(learned, np.asarray(train_labels))
    except (IndexError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"Could not fit {classifier} on this training fold") from exc
    predict_started = time.perf_counter()
    predicted = model.predict(held)
    predict_seconds = time.perf_counter() - predict_started
    true = np.asarray(test_labels)
    class_scores, score_labels = _class_scores(model, held)
    return Prediction(
        float(accuracy_score(true, predicted)),
        true,
        np.asarray(predicted),
        class_scores,
        score_labels,
        predict_seconds,
    )


def _class_scores(model: object, held: np.ndarray) -> tuple[np.ndarray | None, list[str]]:
    labels = [str(item) for item in getattr(model, "classes_", [])]
    proba = getattr(model, "predict_proba", None)
    if callable(proba):
        try:
            scores = np.asarray(proba(held), dtype=np.float64)
        except (ValueError, IndexError):
            scores = None
        if scores is not None and scores.ndim == 2 and scores.shape[1] == len(labels):
            return scores, labels
    decision = getattr(model, "decision_function", None)
    if not callable(decision):
        return None, []
    raw = np.asarray(decision(held), dtype=np.float64)
    if raw.ndim == 1 and len(labels) == 2:
        raw = np.column_stack([-raw, raw])
    if raw.ndim == 2 and raw.shape[1] == len(labels):
        return raw, labels
    return None, []


def _with_metrics(
    accuracy: float,
    detail: str,
    folds: list[FoldScore],
    predictions: list[Prediction],
) -> Evaluation:
    y_true = np.concatenate([item.y_true for item in predictions])
    y_pred = np.concatenate([item.y_pred for item in predictions])
    order = sorted({str(label) for label in y_true} | {str(label) for label in y_pred})
    scores = _stack_scores(predictions, order)
    labels, matrix, metrics = summarize(y_true, y_pred, scores)
    return Evaluation(accuracy, detail, folds, labels, matrix, metrics)


def _stack_scores(predictions: list[Prediction], order: list[str]) -> np.ndarray | None:
    if any(item.scores is None for item in predictions):
        return None
    blocks: list[np.ndarray] = []
    for item in predictions:
        assert item.scores is not None
        aligned = np.full((item.scores.shape[0], len(order)), -1.0e6)
        for column, label in enumerate(item.score_labels):
            if label not in order:
                continue
            aligned[:, order.index(label)] = item.scores[:, column]
        blocks.append(aligned)
    return np.vstack(blocks)
