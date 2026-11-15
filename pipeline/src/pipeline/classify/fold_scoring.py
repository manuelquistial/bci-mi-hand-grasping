"""Fit one fold and summarize pooled predictions."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from pipeline.classify.evaluate_types import Evaluation, FoldScore, Prediction, SubjectTrials
from pipeline.classify.metrics import summarize
from pipeline.classify.models import build_model, scales_features
from pipeline.features.encode import make_encoder
from pipeline.params import ModelParams


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
