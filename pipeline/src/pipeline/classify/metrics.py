"""Metrics that all come from one confusion matrix, plus AUC when scores exist."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def summarize(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None,
) -> tuple[list[str], list[list[int]], dict[str, float | None]]:
    """Return ``(class labels, matrix, metrics)``. Rows of the matrix are the true class."""
    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    order = sorted({str(label) for label in true} | {str(label) for label in predicted})
    matrix = confusion_matrix(true, predicted, labels=order)
    recall = float(
        recall_score(true, predicted, labels=order, average="macro", zero_division=0)
    )
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(true, predicted)),
        "kappa": float(cohen_kappa_score(true, predicted, labels=order)),
        "precision": float(
            precision_score(true, predicted, labels=order, average="macro", zero_division=0)
        ),
        "recall": recall,
        "sensitivity": recall,
        "specificity": _specificity(matrix),
        "f1": float(f1_score(true, predicted, labels=order, average="macro", zero_division=0)),
        "auc": _auc(true, scores, order),
        "mcc": float(matthews_corrcoef(true, predicted)),
    }
    return order, matrix.astype(int).tolist(), metrics


def _specificity(matrix: np.ndarray) -> float:
    total = float(matrix.sum())
    values: list[float] = []
    for index in range(matrix.shape[0]):
        true_positive = float(matrix[index, index])
        false_positive = float(matrix[:, index].sum() - true_positive)
        false_negative = float(matrix[index, :].sum() - true_positive)
        true_negative = total - true_positive - false_positive - false_negative
        denominator = true_negative + false_positive
        values.append(0.0 if denominator == 0 else true_negative / denominator)
    return float(np.mean(values)) if values else 0.0


def _auc(y_true: np.ndarray, scores: np.ndarray | None, order: list[str]) -> float | None:
    if scores is None or scores.shape[1] != len(order):
        return None
    try:
        if len(order) == 2:
            return float(roc_auc_score(y_true, scores[:, 1], labels=order))
        return float(
            roc_auc_score(
                y_true,
                scores,
                labels=order,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return None
