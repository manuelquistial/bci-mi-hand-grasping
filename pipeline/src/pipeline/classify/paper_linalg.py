"""Riemannian linear algebra on batches of SPD matrices."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


_FLOOR = 1e-10


def _spd(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, _FLOOR)
    return vectors @ np.diag(values) @ vectors.T


def _inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(_spd(matrix))
    return vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T


def _sqrtm(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(_spd(matrix))
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


def _expm_sym(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    return vectors @ np.diag(np.exp(values)) @ vectors.T


def _spd_batch(covs: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (covs + np.swapaxes(covs, -1, -2))
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, _FLOOR)
    return np.einsum("nij,nj,nkj->nik", vectors, values, vectors, optimize=True)


def _batch_logm(covs: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (covs + np.swapaxes(covs, -1, -2)))
    values = np.maximum(values, _FLOOR)
    return np.einsum("nij,nj,nkj->nik", vectors, np.log(values), vectors, optimize=True)


def _batch_congruence(reference: np.ndarray, covs: np.ndarray) -> np.ndarray:
    whitener = _inv_sqrt(reference)
    return np.einsum("ij,njk,kl->nil", whitener, covs, whitener, optimize=True)


def _riemann_mean(covs: np.ndarray, iterations: int = 30) -> np.ndarray:
    current = _spd(np.mean(covs, axis=0))
    if len(covs) == 1:
        return current
    for _step in range(iterations):
        step = np.mean(_batch_logm(_batch_congruence(current, covs)), axis=0)
        half = _sqrtm(current)
        current = _spd(half @ _expm_sym(step) @ half)
        if float(np.linalg.norm(step, ord="fro")) < 1e-6:
            break
    return current


def _distance_many(covs: np.ndarray, mean: np.ndarray) -> np.ndarray:
    logged = _batch_logm(_batch_congruence(mean, covs))
    return np.sqrt(np.sum(logged * logged, axis=(1, 2)))


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(_distance_many(left.reshape(1, *left.shape), right)[0])


def _tangent_vectors(covs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    logged = _batch_logm(_batch_congruence(reference, covs))
    rows, cols = np.triu_indices(reference.shape[0])
    values = logged[:, rows, cols]
    scale = np.ones(rows.shape[0], dtype=np.float64)
    scale[rows != cols] = np.sqrt(2.0)
    return values * scale


def _from_tangent(vectors: np.ndarray, reference: np.ndarray) -> np.ndarray:
    half = _sqrtm(reference)
    size = reference.shape[0]
    rows, cols = np.triu_indices(size)
    values = np.asarray(vectors, dtype=np.float64).copy()
    values[:, rows != cols] /= np.sqrt(2.0)
    logged = np.zeros((len(vectors), size, size), dtype=np.float64)
    logged[:, rows, cols] = values
    logged[:, cols, rows] = values
    symmetric = 0.5 * (logged + np.swapaxes(logged, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    mapped_log = np.einsum(
        "nij,nj,nkj->nik", eigvecs, np.exp(eigvals), eigvecs, optimize=True
    )
    mapped = np.einsum("ij,njk,kl->nil", half, mapped_log, half, optimize=True)
    return _spd_batch(mapped)


def _class_names(labels: Sequence[str]) -> list[str]:
    return sorted(set(labels))


def _class_means(
    covs: np.ndarray, labels: Sequence[str]
) -> tuple[list[str], list[np.ndarray]]:
    names = _class_names(labels)
    coded = np.asarray(labels)
    means = [_riemann_mean(covs[coded == name]) for name in names]
    return names, means
