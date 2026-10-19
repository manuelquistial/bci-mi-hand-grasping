"""Time-domain parameters and a Fisher-ratio channel prune.

Each channel contributes the log variance and the log power of the first and
second derivatives, normalized by the previous one. Channels whose Fisher
ratio is below the median, comparing left with right, are left out.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_EPS = 1e-12


def channel_parameters(signal: np.ndarray) -> np.ndarray:
    """``[log var, log var(d1)/var, log var(d2)/var(d1)]``."""
    series = np.asarray(signal, dtype=np.float64)
    if series.ndim != 1 or series.size < 4:
        raise ValueError("A time-domain channel needs at least four samples")
    first = np.diff(series)
    second = np.diff(first)
    variance = float(np.var(series))
    first_power = float(np.var(first))
    second_power = float(np.var(second))
    return np.array(
        [
            np.log(variance + _EPS),
            np.log((first_power + _EPS) / (variance + _EPS)),
            np.log((second_power + _EPS) / (first_power + _EPS)),
        ],
        dtype=np.float64,
    )


def time_domain_parameters(
    trials: Sequence[np.ndarray],
    labels: Sequence[str],
    channels: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    """Return ``(feature names, values)`` with one row per trial."""
    if not trials:
        raise ValueError("Time-domain parameters need at least one trial")
    if len(labels) != len(trials):
        raise ValueError("Each trial needs a label")
    per_trial = [_trial_matrix(trial, channels) for trial in trials]
    kept = _keep_channels(per_trial, labels)
    names = [
        f"{channels[index]} {kind}"
        for index in kept
        for kind in ("var", "d1/var", "d2/d1")
    ]
    values = np.stack(
        [
            np.array([matrix[index, offset] for index in kept for offset in range(3)])
            for matrix in per_trial
        ],
        axis=0,
    )
    return names, values


def _trial_matrix(trial: np.ndarray, channels: Sequence[str]) -> np.ndarray:
    array = np.asarray(trial, dtype=np.float64)
    if array.shape[0] != len(channels):
        raise ValueError("Channel list does not match the trial")
    return np.stack([channel_parameters(array[index]) for index in range(array.shape[0])])


def _keep_channels(matrices: list[np.ndarray], labels: Sequence[str]) -> list[int]:
    count = matrices[0].shape[0]
    left = [matrix for matrix, label in zip(matrices, labels) if label == "left"]
    right = [matrix for matrix, label in zip(matrices, labels) if label == "right"]
    if not left or not right:
        return list(range(count))
    scores = [_fisher(np.stack(left)[:, index], np.stack(right)[:, index]) for index in range(count)]
    cutoff = float(np.median(scores))
    kept = [index for index, score in enumerate(scores) if score >= cutoff]
    return kept or [int(np.argmax(scores))]


def _fisher(left: np.ndarray, right: np.ndarray) -> float:
    gap = (left.mean(axis=0) - right.mean(axis=0)) ** 2
    spread = left.var(axis=0) + right.var(axis=0) + _EPS
    return float(np.mean(gap / spread))
