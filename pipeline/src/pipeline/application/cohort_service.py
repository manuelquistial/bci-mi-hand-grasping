"""Filtered imagery trials for every participant, with a per-process trial cache."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pipeline.adapters.simulator.catalog import query_simulator_trials
from pipeline.alignment import align_trials
from pipeline.classify import SubjectTrials
from pipeline.epochs import ACTIVITY_SECONDS, is_activity_onset
from pipeline.params import ModelParams
from pipeline.preprocess import FilterSpec, apply_notch_then_bandpass
from pipeline.spatial import apply_spatial

TrialLoader = Callable[..., dict[str, Any]]
SubjectsFor = Callable[[str], dict[str, Any]]
WindowPreparer = Callable[..., dict[str, Any]]


def imagery_trials(
    channels: list[str],
    *,
    dataset: str,
    subject: str | None,
    session_dir: str | None,
    sample_rate_hz: float,
    spec: FilterSpec,
    spatial: str,
    trial_loader: TrialLoader,
    params: ModelParams | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Imagery trials after the same filter and spatial step as the preview."""
    bundle = trial_loader(
        dataset=dataset,
        subject=subject,
        session_dir=session_dir,
        trial_seconds=ACTIVITY_SECONDS,
    )
    source_channels = [str(name) for name in bundle.get("channels") or []]
    rate = float(bundle.get("sample_rate_hz") or sample_rate_hz)
    prepared: list[np.ndarray] = []
    labels: list[str] = []
    for trial in bundle.get("trials") or []:
        if not isinstance(trial, dict):
            continue
        kind = str(trial.get("kind") or "")
        label = str(trial.get("label") or "")
        if not is_activity_onset(dataset, kind, label):
            continue
        array = np.asarray(trial.get("samples"), dtype=np.float64)
        filtered = apply_notch_then_bandpass(array, rate, spec)
        names = source_channels or channels
        if spatial != "none":
            filtered, names = apply_spatial(filtered, names, spatial, params)
        if names != channels:
            raise ValueError("Trial channels do not match the preview")
        prepared.append(filtered.T)
        labels.append(label)
    if not prepared:
        raise ValueError("This step needs at least one imagery trial")
    return prepared, labels


def align_groups(
    groups: list[SubjectTrials],
    example: str,
) -> tuple[list[SubjectTrials], np.ndarray]:
    """Each participant is aligned to their own mean covariance."""
    aligned: list[SubjectTrials] = []
    weights: np.ndarray | None = None
    for group in groups:
        trials, matrix = align_trials(group.trials)
        if group.subject == example:
            weights = matrix
        aligned.append(
            SubjectTrials(
                group.subject,
                trials,
                group.labels,
                group.dataset_id,
                group.session_id,
                group.trial_ids,
            )
        )
    if weights is None:
        raise ValueError("This step needs at least one imagery trial")
    return aligned, weights


def group_named(groups: list[SubjectTrials], example: str) -> SubjectTrials:
    for group in groups:
        if group.subject == example:
            return group
    raise ValueError("This step needs at least one imagery trial")


class CohortService:
    """Load and cache imagery trials for one participant or the whole dataset."""

    def __init__(
        self,
        *,
        lock: threading.Lock,
        simulator_project: Path,
        trial_loader: TrialLoader | None,
        subjects_for: SubjectsFor,
        prepare_window: WindowPreparer,
    ) -> None:
        self._lock = lock
        self._simulator_project = simulator_project
        self._trial_loader = trial_loader
        self._subjects_for = subjects_for
        self._prepare_window = prepare_window
        self._trial_cache: dict[tuple[str, str, str, float], dict[str, Any]] = {}

    def load_cohort(
        self,
        *,
        dataset: str,
        session_dir: str | None = None,
        notch_hz: float = 50.0,
        band_low_hz: float = 8.0,
        band_high_hz: float = 30.0,
        order: int = 4,
        spatial: str = "none",
        alignment: str = "none",
        params: ModelParams | None = None,
        workers: int = 1,
    ) -> dict[str, Any]:
        """Filtered imagery trials for every participant.

        The pre-model request still draws one person for its charts. This load
        is the shared cohort a batch scores with every classifier.
        """
        tuned = params or ModelParams()
        listed = list(self._subjects_for(dataset)["subjects"])
        example = str(listed[0]["id"]) if listed else None
        prepared = self._prepare_window(
            dataset=dataset,
            subject=example,
            session_dir=session_dir,
            notch_hz=notch_hz,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            order=order,
            spatial=spatial,
            params=tuned,
        )
        alignment_name = (alignment or "none").strip().lower()
        groups = self.cohort(
            dataset=dataset,
            session_dir=session_dir,
            sample_rate_hz=prepared["rate"],
            spec=prepared["spec"],
            spatial=prepared["spatial_name"],
            channels=prepared["current_channels"],
            only_subject=None,
            params=tuned,
            workers=workers,
        )
        if alignment_name == "euclidean":
            groups, _weights = align_groups(groups, groups[0].subject)
        elif alignment_name != "none":
            raise ValueError(f"Unknown alignment {alignment!r}")
        return {
            "dataset": dataset,
            "groups": groups,
            "rate": prepared["rate"],
            "channels": prepared["current_channels"],
            "spatial": prepared["spatial_name"],
            "alignment": alignment_name,
            "participants": len(groups),
        }

    def cohort(
        self,
        *,
        dataset: str,
        session_dir: str | None,
        sample_rate_hz: float,
        spec: FilterSpec,
        spatial: str,
        channels: list[str],
        only_subject: str | None,
        params: ModelParams | None = None,
        workers: int = 1,
    ) -> list[SubjectTrials]:
        """Imagery trials for every participant, or for the chart example alone."""
        listed = [str(item["id"]) for item in self._subjects_for(dataset)["subjects"]]
        if only_subject is not None and listed:
            targets: list[str | None] = [only_subject]
        elif listed:
            targets = list(listed)
        else:
            targets = [None]

        def load(subject_id: str | None) -> SubjectTrials | None:
            try:
                try:
                    trials, labels = imagery_trials(
                        channels,
                        dataset=dataset,
                        subject=subject_id,
                        session_dir=session_dir,
                        sample_rate_hz=sample_rate_hz,
                        spec=spec,
                        spatial=spatial,
                        trial_loader=self.cached_trials,
                        params=params,
                    )
                except ValueError as exc:
                    if "imagery trial" not in str(exc):
                        raise
                    return None
                session_id = {"shuqfa2024": "02", "liu2024_stroke": "imagery"}.get(dataset, "")
                subject_name = subject_id or "participant"
                trial_ids = tuple(
                    f"{dataset}:{subject_name}:{session_id}:{index}" for index in range(len(labels))
                )
                return SubjectTrials(
                    subject_name,
                    trials,
                    labels,
                    dataset,
                    session_id,
                    trial_ids,
                )
            finally:
                # The numpy trials are copies. The raw window lists are not needed again.
                self.forget_trials(
                    dataset=dataset,
                    subject=subject_id,
                    session_dir=session_dir,
                    trial_seconds=ACTIVITY_SECONDS,
                )

        if workers <= 1 or len(targets) <= 1:
            loaded = [load(subject_id) for subject_id in targets]
        else:
            with ThreadPoolExecutor(max_workers=min(int(workers), len(targets))) as pool:
                loaded = list(pool.map(load, targets))
        groups = [group for group in loaded if group is not None]
        if not groups:
            raise ValueError("This step needs at least one imagery trial")
        return groups

    def trial_key(self, **kwargs: Any) -> tuple[str, str, str, float]:
        return (
            str(kwargs.get("dataset") or ""),
            str(kwargs.get("subject") or ""),
            str(kwargs.get("session_dir") or ""),
            float(kwargs.get("trial_seconds") or 0),
        )

    def cached_trials(self, **kwargs: Any) -> dict[str, Any]:
        key = self.trial_key(**kwargs)
        with self._lock:
            cached = self._trial_cache.get(key)
        if cached is not None:
            return cached
        loader = self._trial_loader or (
            lambda **load_kwargs: query_simulator_trials(self._simulator_project, **load_kwargs)
        )
        bundle = loader(**kwargs)
        with self._lock:
            self._trial_cache[key] = bundle
        return bundle

    def forget_trials(self, **kwargs: Any) -> None:
        with self._lock:
            self._trial_cache.pop(self.trial_key(**kwargs), None)
