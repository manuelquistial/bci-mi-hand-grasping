"""Per-participant decisions, scores, and the shared worker pool."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from pipeline.classify.evaluate import SubjectTrials
from pipeline.classify.paper_estimators import DiscriminantGeodesic, MinimumDistanceMean


def _cores() -> int:
    """One worker per core. Each worker keeps a single BLAS thread."""
    return max(1, os.cpu_count() or 1)


def _map_tasks(
    tasks: Sequence[object], function: Callable[[object], object]
) -> list[object]:
    if not tasks:
        return []
    workers = min(len(tasks), _cores())
    if workers == 1:
        return [function(task) for task in tasks]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, tasks))


def _decide_people(
    model: MinimumDistanceMean | DiscriminantGeodesic,
    gathered: tuple[np.ndarray, list[str], list[str]],
) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    covs, true, subjects = gathered
    predicted, scores = model.decide_covs(covs)
    return true, predicted, scores, subjects


def _collect(
    by_subject: dict[str, SubjectTrials],
    subject_ids: Sequence[str],
) -> tuple[list[np.ndarray], list[str]]:
    trials: list[np.ndarray] = []
    labels: list[str] = []
    for subject_id in subject_ids:
        group = by_subject[subject_id]
        trials.extend(group.trials)
        labels.extend(group.labels)
    if len(set(labels)) < 2:
        raise ValueError("A training split needs both hands")
    return trials, labels


def _predict_people(
    model: MinimumDistanceMean | DiscriminantGeodesic,
    by_subject: dict[str, SubjectTrials],
    subject_ids: Sequence[str],
) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    trials: list[np.ndarray] = []
    true: list[str] = []
    subjects: list[str] = []
    for subject_id in subject_ids:
        group = by_subject[subject_id]
        trials.extend(group.trials)
        true.extend(group.labels)
        subjects.extend([subject_id] * len(group.labels))
    predicted = model.predict(trials)
    return true, predicted, model.score_matrix(trials), subjects


def _per_subject(
    true: Sequence[str],
    predicted: Sequence[str],
    subjects: Sequence[str],
) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for subject_id in sorted(set(subjects)):
        mask = [subject == subject_id for subject in subjects]
        y_true = [label for label, keep in zip(true, mask, strict=True) if keep]
        y_pred = [label for label, keep in zip(predicted, mask, strict=True) if keep]
        if len(set(y_true)) < 2:
            continue
        rows.append((subject_id, float(balanced_accuracy_score(y_true, y_pred))))
    return rows
