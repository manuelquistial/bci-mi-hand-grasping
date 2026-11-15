"""Inner-fold search and subject-level refits for nested evaluation."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from pipeline.classify.evaluate_types import Prediction, SubjectTrials
from pipeline.classify.fold_scoring import _score
from pipeline.params import ModelParams


def _select(
    classifier: str,
    feature: str,
    by_subject: dict[str, SubjectTrials],
    fold: dict[str, Any],
    candidates: Sequence[ModelParams],
    rate: float,
    channels: Sequence[str],
    workers: int,
) -> tuple[ModelParams, dict[str, Any], float, list[dict[str, Any]]]:
    started = time.perf_counter()

    def score_candidate(
        candidate: ModelParams,
    ) -> tuple[float | None, dict[str, Any], dict[str, Any]]:
        updates = _diff(candidate)
        scores: list[float] = []
        errors: list[str] = []
        for inner in fold["inner"]:
            try:
                predicted, spans = _fit_subjects(
                    classifier,
                    feature,
                    by_subject,
                    inner["train_subjects"],
                    inner["validation_subjects"],
                    updates,
                    rate,
                    channels,
                    workers,
                )
            except (ValueError, RuntimeError) as exc:
                errors.append(str(exc))
                continue
            for subject_id, _start, _stop, balanced, _error in _span_scores(spans, predicted):
                if balanced is None:
                    errors.append(f"{subject_id} could not be scored")
                    continue
                scores.append(balanced)
        mean = float(np.mean(scores)) if scores else None
        row = {
            "hyperparameters": _public_updates(updates),
            "balanced_accuracy": None if mean is None else round(mean, 4),
            "subjects": len(scores),
            "errors": errors,
        }
        return mean, updates, row

    if workers <= 1 or len(candidates) <= 1:
        scored = [score_candidate(candidate) for candidate in candidates]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(int(workers), len(candidates))) as pool:
            scored = list(pool.map(score_candidate, candidates))
    best_score: float | None = None
    best = candidates[0]
    best_updates: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for candidate, (mean, updates, row) in zip(candidates, scored, strict=True):
        rows.append(row)
        if mean is not None and (best_score is None or mean > best_score):
            best_score = mean
            best = candidate
            best_updates = updates
    if best_score is None:
        raise ValueError("No configuration could be scored on the inner subject folds")
    return best, best_updates, time.perf_counter() - started, rows


def _fit_subjects(
    classifier: str,
    feature: str,
    by_subject: dict[str, SubjectTrials],
    train_subjects: Sequence[str],
    test_subjects: Sequence[str],
    updates: dict[str, Any],
    rate: float,
    channels: Sequence[str],
    workers: int = 1,
) -> tuple[Prediction, list[tuple[str, int, int]]]:
    train = _join(by_subject, train_subjects)
    test = _join(by_subject, test_subjects)
    if len(set(train.labels)) < 2:
        raise ValueError(
            "Training participants do not cover two classes: "
            + ", ".join(sorted(set(train.labels)) or {"none"})
        )
    combined = SubjectTrials(
        "combined",
        train.trials + test.trials,
        train.labels + test.labels,
        train.dataset_id,
        train.session_id,
    )
    train_index = np.arange(len(train.labels))
    test_index = np.arange(len(train.labels), len(train.labels) + len(test.labels))
    params = _apply(updates)
    if workers <= 1:
        predicted = _score(
            classifier,
            feature,
            combined,
            train_index,
            test_index,
            rate,
            channels,
            params,
        )
    else:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            predicted = _score(
                classifier,
                feature,
                combined,
                train_index,
                test_index,
                rate,
                channels,
                params,
            )
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for subject_id in test_subjects:
        width = len(by_subject[subject_id].labels)
        spans.append((subject_id, cursor, cursor + width))
        cursor += width
    return predicted, spans


def _join(by_subject: dict[str, SubjectTrials], names: Sequence[str]) -> SubjectTrials:
    trials: list[np.ndarray] = []
    labels: list[str] = []
    dataset_id = ""
    session_id = ""
    for name in names:
        group = by_subject[name]
        trials.extend(group.trials)
        labels.extend(group.labels)
        dataset_id = group.dataset_id
        session_id = group.session_id
    if not trials:
        raise ValueError("A subject split has no trials")
    return SubjectTrials("joined", trials, labels, dataset_id, session_id)


def _span_scores(
    spans: Sequence[tuple[str, int, int]],
    predicted: Prediction,
) -> list[tuple[str, int, int, float | None, str]]:
    rows = []
    for subject_id, start, stop in spans:
        true = predicted.y_true[start:stop]
        guess = predicted.y_pred[start:stop]
        if true.size == 0:
            rows.append((subject_id, start, stop, None, "no trials"))
            continue
        rows.append((subject_id, start, stop, float(balanced_accuracy_score(true, guess)), ""))
    return rows


def _diff(candidate: ModelParams) -> dict[str, Any]:
    base = ModelParams()
    updates = {}
    for name in ModelParams.__dataclass_fields__:
        value = getattr(candidate, name)
        if value != getattr(base, name):
            updates[name] = value
    return updates


def _apply(updates: dict[str, Any]) -> ModelParams:
    from pipeline.params import with_updates

    return with_updates(ModelParams(), updates)


def _public_updates(updates: dict[str, Any]) -> dict[str, Any]:
    public = {}
    for key, value in updates.items():
        public[key] = value
    return public
