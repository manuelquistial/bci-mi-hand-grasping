"""Derive spatial and aligned cohorts and store participant covariances."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pipeline.alignment import align_trials, mean_covariance
from pipeline.classify import SubjectTrials
from pipeline.params import ModelParams
from pipeline.spatial import apply_spatial


def derive_cohort(
    cohort: dict[str, Any],
    spatial: str,
    alignment: str,
    params: ModelParams | None,
) -> dict[str, Any]:
    """Apply one spatial method and one alignment without reading the files again."""
    groups = list(cohort["groups"])
    channels = [str(channel) for channel in cohort["channels"]]
    if spatial != "none" or alignment != "none":
        groups, channels = transform_groups(
            groups, channels, spatial, alignment, params
        )
    return {
        "dataset": cohort["dataset"],
        "groups": groups,
        "rate": cohort["rate"],
        "channels": channels,
        "spatial": spatial,
        "alignment": alignment,
        "participants": len(groups),
    }


def transform_groups(
    groups: Sequence[SubjectTrials],
    channels: list[str],
    spatial: str,
    alignment: str,
    params: ModelParams | None,
) -> tuple[list[SubjectTrials], list[str]]:
    tuned = params or ModelParams()
    kept = list(channels)
    transformed: list[SubjectTrials] = []
    for group in groups:
        trials: list[np.ndarray] = []
        for trial in group.trials:
            signal = np.ascontiguousarray(np.asarray(trial, dtype=np.float64).T)
            if spatial != "none":
                signal, kept = apply_spatial(signal, channels, spatial, tuned)
            trials.append(np.ascontiguousarray(signal.T))
        transformed.append(
            SubjectTrials(
                group.subject,
                trials,
                list(group.labels),
                group.dataset_id,
                group.session_id,
                group.trial_ids,
            )
        )
    if alignment != "euclidean":
        return transformed, [str(name) for name in kept]
    aligned: list[SubjectTrials] = []
    for group in transformed:
        trials, _weights = align_trials(group.trials)
        aligned.append(
            SubjectTrials(
                group.subject,
                trials,
                list(group.labels),
                group.dataset_id,
                group.session_id,
                group.trial_ids,
            )
        )
    return aligned, [str(name) for name in kept]


def covariances(
    groups: Sequence[SubjectTrials], channels: Sequence[str]
) -> list[dict[str, Any]]:
    """Mean spatial covariance of each participant, one matrix per person."""
    saved: list[dict[str, Any]] = []
    for group in groups:
        matrix = mean_covariance(group.trials)
        saved.append(
            {
                "subject": group.subject,
                "channels": list(channels),
                "trials": len(group.trials),
                "matrix": round_matrix(matrix),
            }
        )
    return saved


def round_matrix(values: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 6) for value in row] for row in np.asarray(values)]
