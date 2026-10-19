"""Distance-weighted Laplacian. Neighbors inside 70° use a Gaussian kernel."""

from __future__ import annotations

import numpy as np

from pipeline.spatial.montage import angles, positions, surrounded

def matrix(
    channels: list[str],
    radius_degrees: float = 70.0,
    sigma_degrees: float = 30.0,
) -> tuple[np.ndarray, list[str]]:
    labels, points = positions(channels)
    separation = angles(points)
    radius = np.deg2rad(radius_degrees)
    sigma = np.deg2rad(sigma_degrees)
    chosen: list[tuple[int, np.ndarray, np.ndarray]] = []
    for index in range(len(labels)):
        neighbors = np.flatnonzero((separation[index] > 0) & (separation[index] <= radius))
        if neighbors.size < 3 or not surrounded(points[index], points[neighbors]):
            continue
        distance = separation[index, neighbors]
        weights = np.exp(-(distance**2) / (2 * sigma**2))
        weights = weights / weights.sum()
        chosen.append((index, neighbors, weights))
    if not chosen:
        raise ValueError("Large Laplacian found no surrounded electrodes in this montage")
    operator = np.zeros((len(chosen), len(labels)), dtype=np.float64)
    kept: list[str] = []
    for row, (index, neighbors, weights) in enumerate(chosen):
        operator[row, index] = 1.0
        operator[row, neighbors] = -weights
        kept.append(labels[index])
    return operator, kept
