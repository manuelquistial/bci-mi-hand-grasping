"""Two-class common spatial patterns for left versus right.

``Σ_left w = λ (Σ_left + Σ_right) w``. Features are the log variance of the
first and last ``m`` spatial filters.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.linalg import eigh

_M = 2
_EPS = 1e-12


def spatial_filters(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    *,
    pairs: int = _M,
) -> np.ndarray:
    """Columns are filters, largest eigenvalues first, then the smallest."""
    if not left or not right:
        raise ValueError("CSP needs left and right trials")
    cov_left = _mean_covariance(left)
    cov_right = _mean_covariance(right)
    if cov_left.shape != cov_right.shape:
        raise ValueError("Left and right trials must share a channel count")
    _eigenvalues, vectors = eigh(_shrink(cov_left), _shrink(cov_left + cov_right))
    count = int(vectors.shape[1])
    keep = min(pairs, count // 2)
    if keep < 1:
        raise ValueError("CSP needs at least two channels")
    chosen = list(range(count - keep, count)) + list(range(keep))
    return vectors[:, chosen]


def log_variances(trial: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """``log var(W^T X)`` for one trial shaped ``(channels, samples)``."""
    projected = np.asarray(filters, dtype=np.float64).T @ np.asarray(trial, dtype=np.float64)
    return np.log(np.var(projected, axis=1) + _EPS)


def assert_hand_pair(labels: Sequence[str]) -> None:
    """CSP and FBCSP compare left with right. Other classes are not dropped."""
    extra = sorted({str(label) for label in labels} - {"left", "right"})
    if extra:
        raise ValueError(
            "CSP and FBCSP are left versus right. Other classes are present: "
            + ", ".join(extra)
        )


def csp_features(
    trials: Sequence[np.ndarray],
    labels: Sequence[str],
    pairs: int = _M,
) -> tuple[list[str], np.ndarray]:
    """Return ``(feature names, values)`` with one row per trial."""
    assert_hand_pair(labels)
    left, right = _split(trials, labels)
    filters = spatial_filters(left, right, pairs=pairs)
    names = [f"left {index + 1}" for index in range(filters.shape[1] // 2)]
    names += [f"right {index + 1}" for index in range(filters.shape[1] - len(names))]
    values = np.stack([log_variances(trial, filters) for trial in trials], axis=0)
    return names, values


def _split(
    trials: Sequence[np.ndarray],
    labels: Sequence[str],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if len(labels) != len(trials):
        raise ValueError("Each trial needs a label")
    left = [trial for trial, label in zip(trials, labels) if label == "left"]
    right = [trial for trial, label in zip(trials, labels) if label == "right"]
    if not left or not right:
        raise ValueError("CSP needs left and right trials")
    return left, right


def _mean_covariance(trials: Sequence[np.ndarray]) -> np.ndarray:
    covariances = []
    for trial in trials:
        array = np.asarray(trial, dtype=np.float64)
        centered = array - array.mean(axis=1, keepdims=True)
        covariances.append((centered @ centered.T) / (centered.shape[1] - 1))
    return sum(covariances) / len(covariances)


def _shrink(cov: np.ndarray) -> np.ndarray:
    scale = float(np.trace(cov) / cov.shape[0])
    return cov + 1e-6 * max(scale, _EPS) * np.eye(cov.shape[0])
