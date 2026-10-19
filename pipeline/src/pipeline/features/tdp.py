"""Time-domain parameters and a Fisher-ratio channel prune.

Each channel contributes the log variance and the log power of the first and
second derivatives, normalized by the previous one. Channels whose Fisher
ratio is below the median are left out. With more than two classes the score
is the mean of each class against the rest.
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
    """Keep channels at or above the median class-versus-rest Fisher score.

    Two classes reduce to the original left-versus-right score. Extra classes
    stay in the comparison instead of being dropped.
    """
    count = matrices[0].shape[0]
    classes = sorted({str(label) for label in labels})
    if len(classes) < 2:
        return list(range(count))
    scores = [_channel_score(matrices, labels, index, classes) for index in range(count)]
    finite = [score for score in scores if np.isfinite(score)]
    if not finite:
        return list(range(count))
    cutoff = float(np.median(finite))
    kept = [index for index, score in enumerate(scores) if np.isfinite(score) and score >= cutoff]
    if kept:
        return kept
    best = max(range(count), key=lambda index: scores[index] if np.isfinite(scores[index]) else -1.0)
    return [best]


def _channel_score(
    matrices: list[np.ndarray],
    labels: Sequence[str],
    index: int,
    classes: Sequence[str],
) -> float:
    parts: list[float] = []
    for name in classes:
        one = [matrix for matrix, label in zip(matrices, labels) if label == name]
        rest = [matrix for matrix, label in zip(matrices, labels) if label != name]
        if not one or not rest:
            continue
        score = _fisher(np.stack(one)[:, index], np.stack(rest)[:, index])
        if np.isfinite(score):
            parts.append(score)
    if not parts:
        return 0.0
    return float(np.mean(parts))


def _fisher(left: np.ndarray, right: np.ndarray) -> float:
    gap = (left.mean(axis=0) - right.mean(axis=0)) ** 2
    spread = left.var(axis=0) + right.var(axis=0) + _EPS
    return float(np.mean(gap / spread))
