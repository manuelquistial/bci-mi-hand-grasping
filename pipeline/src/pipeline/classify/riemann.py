"""Tangent-space map of trial covariances, then a linear SVM.

The reference is the mean covariance of the training trials. Each covariance
is whitened and mapped with the matrix logarithm. The classifier sees the
upper triangle of that tangent vector.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.linalg import logm

_EPS = 1e-12


class TangentEncoder:
    """Unsupervised map. ``fit`` uses training trials only."""

    def fit(self, trials: Sequence[np.ndarray], labels: Sequence[str]) -> None:
        del labels
        covariances = [_covariance(trial) for trial in trials]
        reference = sum(covariances) / len(covariances)
        self._whitener = _inverse_square_root(_shrink(reference))

    def transform(self, trials: Sequence[np.ndarray]) -> np.ndarray:
        rows = []
        for trial in trials:
            whitened = self._whitener @ _covariance(trial) @ self._whitener.T
            tangent = np.real(logm(_shrink(whitened)))
            rows.append(_upper(tangent))
        return np.stack(rows, axis=0)


def _covariance(trial: np.ndarray) -> np.ndarray:
    array = np.asarray(trial, dtype=np.float64)
    centered = array - array.mean(axis=1, keepdims=True)
    return (centered @ centered.T) / (centered.shape[1] - 1)


def _shrink(cov: np.ndarray) -> np.ndarray:
    scale = float(np.trace(cov) / cov.shape[0])
    return cov + 1e-6 * max(scale, _EPS) * np.eye(cov.shape[0])


def _inverse_square_root(cov: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, _EPS, None)
    return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T


def _upper(matrix: np.ndarray) -> np.ndarray:
    index = np.triu_indices(matrix.shape[0])
    values = matrix[index].astype(np.float64, copy=True)
    diagonal = index[0] == index[1]
    values[~diagonal] *= np.sqrt(2.0)
    return values
