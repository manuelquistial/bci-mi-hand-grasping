"""Spherical spline current-source density on the unit sphere.

Perrin's spline: Legendre order ``m = 4``, 20 terms, and a small ridge on the
Gram matrix. The fit uses every electrode. Perimeter electrodes stay in that
fit and are dropped from the rows that leave this module.
"""

from __future__ import annotations

import numpy as np

from pipeline.spatial.montage import angles, positions, surrounded

_ORDER = 4
_TERMS = 20
_RIDGE = 1e-5
_LOCAL = np.deg2rad(36.0)


def matrix(
    channels: list[str],
    order: int = _ORDER,
    ridge: float = _RIDGE,
) -> tuple[np.ndarray, list[str]]:
    labels, points = positions(channels)
    count = len(labels)
    if count < 4:
        raise ValueError("Spherical spline needs at least 4 positioned electrodes")
    if order < 2:
        raise ValueError("Spline order must be at least 2")
    cosine = np.clip(points @ points.T, -1.0, 1.0)
    scale = 1.0 / (4.0 * np.pi)
    gram = scale * _series(cosine, power=order)
    kernel = scale * _series(cosine, power=order - 1)
    system = np.zeros((count + 1, count + 1), dtype=np.float64)
    system[:count, :count] = gram + ridge * np.eye(count)
    system[:count, count] = 1.0
    system[count, :count] = 1.0
    solved = np.linalg.solve(system, np.eye(count + 1, count))
    full = -(kernel @ solved[:count, :])
    keep = _interior(labels, points)
    if not keep:
        raise ValueError("Spherical spline found no interior electrodes in this montage")
    return full[keep], [labels[index] for index in keep]


def _series(cosine: np.ndarray, power: int) -> np.ndarray:
    """``sum (2n+1) / (n^p (n+1)^p) P_n(x)`` for ``n = 1..20``."""
    total = np.zeros_like(cosine)
    previous = np.ones_like(cosine)
    current = cosine.copy()
    for degree in range(1, _TERMS + 1):
        if degree > 1:
            nxt = ((2 * degree - 1) * cosine * current - (degree - 1) * previous) / degree
            previous, current = current, nxt
        factor = (2 * degree + 1) / (degree**power * (degree + 1) ** power)
        total += factor * current
    return total


def _interior(labels: list[str], points: np.ndarray) -> list[int]:
    separation = angles(points)
    kept: list[int] = []
    for index in range(len(labels)):
        neighbors = np.flatnonzero((separation[index] > 0) & (separation[index] <= _LOCAL))
        if neighbors.size >= 3 and surrounded(points[index], points[neighbors]):
            kept.append(index)
    return kept
