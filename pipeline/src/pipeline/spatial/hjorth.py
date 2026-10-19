"""Local Hjorth Laplacian. Each kept channel subtracts the mean of its neighbors.

Dense caps use electrodes within 36°. A sparse cap (8 channels or fewer) that
has no pair that close falls back to the nearest available electrodes around
C3, Cz, and C4. Perimeter rows, which are not surrounded, are left out.
"""

from __future__ import annotations

import numpy as np

from pipeline.spatial.montage import angles, positions, surrounded

_SPARSE = 8
_MOTOR = {"C3", "CZ", "C4"}


def matrix(channels: list[str], degrees: float = 36.0) -> tuple[np.ndarray, list[str]]:
    labels, points = positions(channels)
    separation = angles(points)
    local = _rows(labels, points, separation, np.deg2rad(degrees), require_surround=True)
    if local:
        return _stack(local, len(labels)), [labels[index] for index, _ in local]
    if len(labels) > _SPARSE:
        raise ValueError("Hjorth found no surrounded electrodes in this montage")
    motor = [index for index, name in enumerate(labels) if name.upper() in _MOTOR]
    if not motor:
        raise ValueError("Hjorth on a sparse montage needs C3, Cz, or C4")
    fallback = _nearest(motor, separation)
    if not fallback:
        raise ValueError("Hjorth on a sparse montage needs C3, Cz, or C4")
    return _stack(fallback, len(labels)), [labels[index] for index, _ in fallback]


def _rows(
    labels: list[str],
    points: np.ndarray,
    separation: np.ndarray,
    limit: float,
    *,
    require_surround: bool,
) -> list[tuple[int, np.ndarray]]:
    chosen: list[tuple[int, np.ndarray]] = []
    for index in range(len(labels)):
        neighbors = np.flatnonzero((separation[index] > 0) & (separation[index] <= limit))
        if neighbors.size < 2:
            continue
        if require_surround and not surrounded(points[index], points[neighbors]):
            continue
        chosen.append((index, neighbors))
    return chosen


def _nearest(indexes: list[int], separation: np.ndarray) -> list[tuple[int, np.ndarray]]:
    chosen: list[tuple[int, np.ndarray]] = []
    for index in indexes:
        order = np.argsort(separation[index])
        neighbors = [int(other) for other in order if other != index][:4]
        if neighbors:
            chosen.append((index, np.asarray(neighbors, dtype=int)))
    return chosen


def _stack(rows: list[tuple[int, np.ndarray]], width: int) -> np.ndarray:
    matrix_ = np.zeros((len(rows), width), dtype=np.float64)
    for row, (index, neighbors) in enumerate(rows):
        matrix_[row, index] = 1.0
        matrix_[row, neighbors] = -1.0 / neighbors.size
    return matrix_
