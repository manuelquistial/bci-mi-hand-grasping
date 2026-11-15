"""Train a frozen online model from a simulator dataset (offline dump).

Cross-subject by default: every participant's left/right imagery trials are
filtered, Euclidean-aligned **within that participant**, then pooled to fit
one shared CSP and the classifier heads. Online verification uses a single
subject against that frozen pack.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from pipeline.alignment.euclidean import align_trials
from pipeline.epochs import ACTIVITY_SECONDS, is_activity_onset
from pipeline.online.model import (
    FrozenOnlineModel,
    parse_classifier_names,
    train_frozen_model,
)
from pipeline.preprocess import FilterSpec, apply_notch_then_bandpass
from pipeline.adapters.simulator.catalog import (
    query_simulator_subjects,
    query_simulator_trials,
)
from pipeline.config import default_simulator_project

TrialLoader = Callable[..., dict[str, Any]]
SubjectLister = Callable[[Path, str], list[dict[str, str]]]
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
    subject_lister: SubjectLister | None = None,
    regularization_c: float = 1.0,
    control_classifier: str | None = None,
    cross_subject: bool = True,
    workers: int = 4,
    online_subject: str | None = None,
    hold_out_online_subject: bool = True,
) -> FrozenOnlineModel:
    """Fit shared CSP + N classifier heads on left/right imagery trials.

    When ``cross_subject`` is true (default), every listed participant is
    loaded. Euclidean alignment is applied per participant, then CSP and the
    heads are fit on the pooled aligned trials. ``online_subject`` (or
    ``subject``) is the verification subject for online replay. By default that
    participant is held out of training (LOSO-style pack).

    When ``cross_subject`` is false, only ``subject`` is used (legacy smoke).
    """
    names = parse_classifier_names(classifier_names, classifier_name=classifier_name)
    project = Path(simulator_project or default_simulator_project())
    spec = filter or FilterSpec()
    loader = trial_loader or (
        lambda **kwargs: query_simulator_trials(project, **kwargs)
    )
    lister = subject_lister or (
        lambda root, dataset_id: query_simulator_subjects(root, dataset_id)
    )
    verify = online_subject or subject
    held_out = False
    if cross_subject and dataset != "recorded_session":
        targets = [str(item["id"]) for item in lister(project, dataset)]
        if not targets:
            raise ValueError(f"No participants listed for dataset {dataset!r}")
        if verify is None:
            verify = targets[0]
        if hold_out_online_subject and verify is not None:
            kept = [sid for sid in targets if not _same_subject(sid, verify)]
            if len(kept) == len(targets):
                raise ValueError(
                    f"online_subject {verify!r} not found among listed participants "
                    f"(cannot hold out for LOSO training)"
                )
            if not kept:
                raise ValueError("Holding out the online subject left an empty training cohort")
            targets = kept
            held_out = True
    else:
        chosen = subject or _default_subject(project, dataset, session_dir)
        targets = [chosen] if chosen is not None else [None]
        verify = chosen

    def load_one(subject_id: str | None) -> tuple[str, list[np.ndarray], list[str], float]:
        bundle = loader(
            dataset=dataset,
            subject=subject_id,
            session_dir=session_dir,
            trial_seconds=ACTIVITY_SECONDS,
        )
        rate = float(bundle.get("sample_rate_hz") or 250.0)
        trials, labels = _hand_trials(bundle, dataset=dataset)
        filtered: list[np.ndarray] = []
        for trial in trials:
            filtered.append(apply_notch_then_bandpass(trial.T, rate, spec).T)
        aligned, _weights = align_trials(filtered)
        return str(subject_id or "participant"), aligned, labels, rate

    if workers <= 1 or len(targets) <= 1:
        loaded = [load_one(subject_id) for subject_id in targets]
    else:
        with ThreadPoolExecutor(max_workers=min(int(workers), len(targets))) as pool:
            loaded = list(pool.map(load_one, targets))

    all_trials: list[np.ndarray] = []
    all_labels: list[str] = []
    used_subjects: list[str] = []
    rate = 250.0
    for subject_name, trials, labels, subject_rate in loaded:
        if not trials:
            continue
        rate = subject_rate
        used_subjects.append(subject_name)
        all_trials.extend(trials)
        all_labels.extend(labels)
    if len(set(all_labels)) < 2:
        raise ValueError("Need left and right imagery trials across the training cohort")
    model = train_frozen_model(
        all_trials,
        all_labels,
        sample_rate_hz=rate,
        classifier_names=names,
        filter=spec,
        csp_pairs=csp_pairs,
        apply_filter=False,
        apply_training_ea=False,
        regularization_c=regularization_c,
        control_classifier=control_classifier,
    )
    meta = dict(model.meta or {})
    meta.update(
        {
            "dataset": dataset,
            "cross_subject": cross_subject,
            "train_subjects": used_subjects,
            "n_train_subjects": len(used_subjects),
            "online_subject": verify or "",
            "held_out_online_subject": held_out,
            "session_dir": session_dir or "",
            "n_hand_trials": len(all_labels),
            "alignment": "euclidean",
            "features": "csp",
            "training_ea": "per_subject",
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
    return prepared, labels


def _default_subject(
    project: Path,
    dataset: str,
    session_dir: str | None,
) -> str | None:
    del session_dir
    if dataset == "recorded_session":
        return None
    subjects = query_simulator_subjects(project, dataset)
    if not subjects:
        raise ValueError(f"No participants listed for dataset {dataset!r}")
    return str(subjects[0]["id"])


def _subject_key(subject_id: str | None) -> str | None:
    """Normalize ids like ``1``, ``S001``, ``sub_1`` to a comparable key."""
    if subject_id is None:
        return None
    text = str(subject_id).strip()
    if not text:
        return None
    match = re.fullmatch(r"(?:sub[_-]?|s)?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return str(int(match.group(1)))
    return text.casefold()


def _same_subject(left: str | None, right: str | None) -> bool:
    a = _subject_key(left)
    b = _subject_key(right)
    return a is not None and a == b
