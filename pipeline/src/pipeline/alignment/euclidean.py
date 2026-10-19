"""Euclidean alignment of one subject's band-filtered trials.

Each trial is shaped ``(channels, samples)``. The reference is the mean
spatial covariance. ``W = R^{-1/2}`` comes from the eigendecomposition, and
each aligned trial is ``W @ X``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_FLOOR = 1e-10


def trial_covariance(trial: np.ndarray) -> np.ndarray:
    """Centered spatial covariance. ``trial`` is ``(channels, samples)``."""
    array = np.asarray(trial, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError("A trial needs at least one channel and two samples")
    centered = array - array.mean(axis=1, keepdims=True)
    return (centered @ centered.T) / (centered.shape[1] - 1)


def alignment_matrix(trials: Sequence[np.ndarray]) -> np.ndarray:
    """``R^{-1/2}`` from the mean covariance of the trials."""
    if not trials:
        raise ValueError("Euclidean alignment needs at least one trial")
    covariances = [trial_covariance(trial) for trial in trials]
    reference = sum(covariances) / len(covariances)
    eigenvalues, eigenvectors = np.linalg.eigh(reference)
    eigenvalues = np.clip(eigenvalues, _FLOOR, None)
    scale = 1.0 / np.sqrt(eigenvalues)
    return eigenvectors @ np.diag(scale) @ eigenvectors.T


def align_trials(trials: Sequence[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    """Return ``(aligned trials, W)``. Alignment uses the original samples."""
    weights = alignment_matrix(trials)
    aligned = [weights @ np.asarray(trial, dtype=np.float64) for trial in trials]
    return aligned, weights


def apply_alignment(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply ``W`` to a signal shaped ``(samples, channels)``."""
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Alignment input must be a 2-D array")
    operator = np.asarray(weights, dtype=np.float64)
    if operator.shape != (array.shape[1], array.shape[1]):
        raise ValueError("Alignment matrix does not match the channel count")
    return array @ operator.T


def mean_covariance(trials: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of the centered covariances. After alignment this is near ``I``."""
    if not trials:
        raise ValueError("Euclidean alignment needs at least one trial")
    covariances = [trial_covariance(trial) for trial in trials]
    return sum(covariances) / len(covariances)
