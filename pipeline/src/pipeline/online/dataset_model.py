"""Train a frozen online model from a simulator dataset (offline dump)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from pipeline.epochs import ACTIVITY_SECONDS, is_activity_onset
from pipeline.online.model import (
    FrozenOnlineModel,
    parse_classifier_names,
    train_frozen_model,
)
from pipeline.preprocess import FilterSpec
from pipeline.service import (
    default_simulator_project,
    query_simulator_subjects,
    query_simulator_trials,
)

TrialLoader = Callable[..., dict[str, Any]]
_HANDS = frozenset({"left", "right"})


def train_frozen_from_dataset(
    *,
    dataset: str,
    subject: str | None = None,
    session_dir: str | None = None,
    classifier_name: str | None = None,
    classifier_names: Sequence[str] | str | None = None,
    filter: FilterSpec | None = None,
    csp_pairs: int = 2,
    simulator_project: Path | None = None,
    trial_loader: TrialLoader | None = None,
    regularization_c: float = 1.0,
    control_classifier: str | None = None,
) -> FrozenOnlineModel:
    """Fit shared CSP + N classifier heads on left/right imagery trials.

    Uses the simulator ``--dump-trials`` path (no LSL). Filtering here is the
    offline ``filtfilt`` path; the live session applies the causal twin.
    """
    names = parse_classifier_names(classifier_names, classifier_name=classifier_name)
    project = Path(simulator_project or default_simulator_project())
    chosen = subject or _default_subject(project, dataset, session_dir)
    loader = trial_loader or (
        lambda **kwargs: query_simulator_trials(project, **kwargs)
    )
    bundle = loader(
        dataset=dataset,
        subject=chosen,
        session_dir=session_dir,
        trial_seconds=ACTIVITY_SECONDS,
    )
    rate = float(bundle.get("sample_rate_hz") or 250.0)
    trials, labels = _hand_trials(bundle, dataset=dataset)
    if len(set(labels)) < 2:
        raise ValueError("Need left and right imagery trials to freeze the model")
    model = train_frozen_model(
        trials,
        labels,
        sample_rate_hz=rate,
        classifier_names=names,
        filter=filter or FilterSpec(),
        csp_pairs=csp_pairs,
        apply_filter=True,
        apply_training_ea=True,
        regularization_c=regularization_c,
        control_classifier=control_classifier,
    )
    meta = dict(model.meta or {})
    meta.update(
        {
            "dataset": dataset,
            "subject": chosen or "",
            "session_dir": session_dir or "",
            "n_hand_trials": len(labels),
        }
    )
    return FrozenOnlineModel(
        sample_rate_hz=model.sample_rate_hz,
        filter=model.filter,
        csp_filters=model.csp_filters,
        heads=model.heads,
        channel_count=model.channel_count,
        csp_pairs=model.csp_pairs,
        meta=meta,
        control_classifier=model.control_name,
    )


def _hand_trials(
    bundle: dict[str, Any],
    *,
    dataset: str,
) -> tuple[list[np.ndarray], list[str]]:
    prepared: list[np.ndarray] = []
    labels: list[str] = []
    for trial in bundle.get("trials") or []:
        if not isinstance(trial, dict):
            continue
        kind = str(trial.get("kind") or "")
        label = str(trial.get("label") or "")
        if not is_activity_onset(dataset, kind, label):
            continue
        if label not in _HANDS:
            continue
        array = np.asarray(trial.get("samples"), dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 16:
            continue
        # dump-trials are (samples, channels); CSP expects (channels, samples).
        prepared.append(array.T)
        labels.append(label)
    if not prepared:
        raise ValueError("No left/right imagery trials found in this participant")
    return prepared, labels


def _default_subject(
    project: Path,
    dataset: str,
    session_dir: str | None,
) -> str | None:
    if dataset == "recorded_session":
        return None
    subjects = query_simulator_subjects(project, dataset)
    if not subjects:
        raise ValueError(f"No participants listed for dataset {dataset!r}")
    return str(subjects[0]["id"])
