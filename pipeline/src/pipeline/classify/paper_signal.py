"""Spatial method and Euclidean alignment applied before the paper models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pipeline.alignment.euclidean import align_trials
from pipeline.classify.evaluate import SubjectTrials
from pipeline.params import ModelParams
from pipeline.spatial import METHODS as SPATIAL_METHODS
from pipeline.spatial import apply_spatial


SIGNAL_ALIGNMENTS = ("none", "euclidean")


def apply_signal(
    groups: Sequence[SubjectTrials],
    channels: Sequence[str],
    spatial: str,
    alignment: str,
) -> tuple[list[SubjectTrials], list[str]]:
    """Same spatial step and per-participant Euclidean alignment as the batch."""
    spatial_name = spatial.strip().lower()
    alignment_name = alignment.strip().lower()
    if spatial_name not in SPATIAL_METHODS:
        raise ValueError(f"Unknown spatial method {spatial!r}")
    if alignment_name not in SIGNAL_ALIGNMENTS:
        raise ValueError(f"Unknown alignment {alignment!r}")
    labels = [str(channel) for channel in channels]
    if spatial_name == "none" and alignment_name == "none":
        _check_width(groups, len(labels))
        return list(groups), labels
    tuned = ModelParams()
    transformed: list[SubjectTrials] = []
    kept = list(labels)
    for group in groups:
        trials: list[np.ndarray] = []
        for trial in group.trials:
            signal = np.ascontiguousarray(np.asarray(trial, dtype=np.float64).T)
            if signal.shape[1] != len(labels):
                raise ValueError("Channel list does not match the signal width")
            if spatial_name != "none":
                signal, kept = apply_spatial(signal, labels, spatial_name, tuned)
            trials.append(np.ascontiguousarray(signal.T))
        transformed.append(_replace_trials(group, trials))
    if alignment_name != "euclidean":
        return transformed, [str(name) for name in kept]
    aligned = []
    for group in transformed:
        trials, _weights = align_trials(group.trials)
        aligned.append(_replace_trials(group, trials))
    return aligned, [str(name) for name in kept]


def _replace_trials(group: SubjectTrials, trials: list[np.ndarray]) -> SubjectTrials:
    return SubjectTrials(
        group.subject,
        trials,
        list(group.labels),
        group.dataset_id,
        group.session_id,
        group.trial_ids,
    )


def _check_width(groups: Sequence[SubjectTrials], width: int) -> None:
    for group in groups:
        for trial in group.trials:
            if np.asarray(trial).shape[0] != width:
                raise ValueError("Channel list does not match the signal width")
