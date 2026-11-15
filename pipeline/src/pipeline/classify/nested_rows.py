"""Per-subject and per-trial report rows for nested evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pipeline.classify.evaluate_types import Prediction, SubjectTrials
from pipeline.classify.nested_fit import _join, _span_scores
from pipeline.features.encode import make_encoder


def _subject_rows(
    people: Sequence[SubjectTrials],
    fold: int,
    spans: Sequence[tuple[str, int, int]],
    predicted: Prediction,
    dataset_id: str,
) -> list[dict[str, Any]]:
    lookup = {group.subject: group for group in people}
    rows = []
    for subject_id, start, stop, balanced, error in _span_scores(spans, predicted):
        group = lookup[subject_id]
        true = predicted.y_true[start:stop]
        guess = predicted.y_pred[start:stop]
        accuracy = None if error or true.size == 0 else float(np.mean(true == guess))
        rows.append(
            {
                "dataset_id": group.dataset_id or dataset_id,
                "subject_id": subject_id,
                "session_id": group.session_id,
                "fold": fold,
                "trials": int(true.size),
                "balanced_accuracy": None if balanced is None else round(balanced, 4),
                "accuracy": None if accuracy is None else round(accuracy, 4),
                "error": error,
            }
        )
    return rows


def _prediction_table(
    people: Sequence[SubjectTrials],
    fold: int,
    spans: Sequence[tuple[str, int, int]],
    predicted: Prediction,
    dataset_id: str,
) -> list[dict[str, Any]]:
    lookup = {group.subject: group for group in people}
    rows: list[dict[str, Any]] = []
    for subject_id, start, stop in spans:
        group = lookup[subject_id]
        for offset, trial_index in enumerate(range(start, stop)):
            trial_id = (
                group.trial_ids[offset]
                if offset < len(group.trial_ids)
                else f"{dataset_id}:{subject_id}:{group.session_id}:{offset}"
            )
            rows.append(
                {
                    "dataset_id": group.dataset_id or dataset_id,
                    "subject_id": subject_id,
                    "session_id": group.session_id,
                    "trial_id": trial_id,
                    "fold": fold,
                    "label": str(predicted.y_true[trial_index]),
                    "predicted": str(predicted.y_pred[trial_index]),
                }
            )
    return rows


def _tdp_channels(
    by_subject: dict[str, SubjectTrials],
    development: Sequence[str],
    channels: Sequence[str],
) -> list[str]:
    joined = _join(by_subject, development)
    encoder = make_encoder("tdp", 1.0, channels)
    encoder.fit(joined.trials, joined.labels)
    return list(getattr(encoder, "kept", []))


def _versions() -> dict[str, str]:
    import scipy
    import sklearn

    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _versions() -> dict[str, str]:
    import scipy
    import sklearn

    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }
