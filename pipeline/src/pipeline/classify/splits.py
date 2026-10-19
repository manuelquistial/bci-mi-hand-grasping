"""Cross-subject folds. Every trial of a person stays in one side of a split."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pipeline.classify.evaluate import SubjectTrials

_SEED = 42


def session_for(dataset_id: str) -> str:
    """Run the loader actually opens. Shuqfa defaults to run 02. Liu has one imagery file."""
    if dataset_id == "shuqfa2024":
        return "02"
    if dataset_id == "liu2024_stroke":
        return "imagery"
    return ""


def identify(group: SubjectTrials, dataset_id: str) -> SubjectTrials:
    """Fill dataset, session, and trial ids when the cohort did not store them."""
    session = group.session_id or session_for(dataset_id or group.dataset_id)
    dataset = group.dataset_id or dataset_id
    if group.trial_ids and len(group.trial_ids) == len(group.labels):
        return SubjectTrials(
            group.subject,
            group.trials,
            group.labels,
            dataset,
            session,
            group.trial_ids,
        )
    trial_ids = tuple(
        f"{dataset}:{group.subject}:{session}:{index}" for index in range(len(group.labels))
    )
    return SubjectTrials(group.subject, group.trials, group.labels, dataset, session, trial_ids)


def fold_counts(n_subjects: int, requested_outer: int, requested_inner: int) -> dict[str, Any]:
    """Reduce fold counts when there are not enough people. Never split trials instead."""
    if n_subjects < 2:
        raise ValueError(
            "Cross-subject evaluation needs at least two participants. "
            "Trials were not split at random."
        )
    outer = min(max(2, int(requested_outer)), n_subjects)
    notes: list[str] = []
    if outer != int(requested_outer):
        notes.append(
            f"Outer folds reduced from {int(requested_outer)} to {outer} "
            f"because there are {n_subjects} participants"
        )
    smallest_dev = n_subjects - int(np.ceil(n_subjects / outer))
    if smallest_dev < 2:
        raise ValueError(
            "The development side of an outer fold would have fewer than two participants, "
            "so an inner subject split is not possible"
        )
    inner = min(max(2, int(requested_inner)), smallest_dev)
    if inner != int(requested_inner):
        notes.append(
            f"Inner folds reduced from {int(requested_inner)} to {inner} "
            f"because an outer development set can have {smallest_dev} participants"
        )
    return {
        "participants": n_subjects,
        "outer_folds": outer,
        "inner_folds": inner,
        "adjustment": "; ".join(notes),
    }


def build_manifest(
    groups: Sequence[SubjectTrials],
    *,
    dataset_id: str,
    outer: str,
    outer_folds: int,
    inner_folds: int,
    seed: int = _SEED,
) -> dict[str, Any]:
    """One split reused by every pipeline. Subjects are the only grouping key."""
    people = [identify(group, dataset_id) for group in groups]
    names = [group.subject for group in people]
    if len(names) != len(set(names)):
        raise ValueError("A dataset has two groups with the same participant id")
    kind = "loso" if outer == "loso" else "group"
    plan = fold_counts(len(names), len(names) if kind == "loso" else outer_folds, inner_folds)
    outer_map = (
        {name: index for index, name in enumerate(sorted(names))}
        if kind == "loso"
        else _assign(names, int(plan["outer_folds"]), seed)
    )
    folds: list[dict[str, Any]] = []
    for number in sorted(set(outer_map.values())):
        test = sorted(name for name, fold in outer_map.items() if fold == number)
        development = sorted(name for name in names if name not in set(test))
        inner_map = _assign(development, int(plan["inner_folds"]), seed + 1 + number)
        inner = []
        for inner_number in sorted(set(inner_map.values())):
            held = sorted(name for name, fold in inner_map.items() if fold == inner_number)
            train = sorted(name for name in development if name not in set(held))
            inner.append(
                {
                    "fold": inner_number + 1,
                    "train_subjects": train,
                    "validation_subjects": held,
                }
            )
        folds.append(
            {
                "fold": number + 1,
                "test_subjects": test,
                "development_subjects": development,
                "inner": inner,
            }
        )
    manifest = {
        "dataset_id": dataset_id,
        "seed": int(seed),
        "outer": kind,
        "requested_outer_folds": len(names) if kind == "loso" else int(outer_folds),
        "outer_folds": len(folds),
        "requested_inner_folds": int(inner_folds),
        "inner_folds": int(plan["inner_folds"]),
        "adjustment": plan["adjustment"],
        "subjects": sorted(names),
        "trials": [
            {
                "dataset_id": group.dataset_id,
                "subject_id": group.subject,
                "session_id": group.session_id,
                "trial_id": trial_id,
                "label": label,
            }
            for group in people
            for trial_id, label in zip(group.trial_ids, group.labels, strict=True)
        ],
        "folds": folds,
    }
    check_manifest(manifest)
    return manifest


def check_manifest(manifest: dict[str, Any]) -> None:
    """Reject a split that reuses a person or leaves someone out of the outer test."""
    seen: list[str] = []
    for fold in manifest["folds"]:
        test = set(fold["test_subjects"])
        development = set(fold["development_subjects"])
        if test & development:
            raise ValueError("A participant is in both development and outer test")
        if not test:
            raise ValueError("An outer fold has no test participant")
        seen.extend(test)
        for inner in fold["inner"]:
            train = set(inner["train_subjects"])
            validation = set(inner["validation_subjects"])
            if train & validation:
                raise ValueError("A participant is in both inner train and validation")
            if validation & test or train & test:
                raise ValueError("An outer test participant entered an inner fold")
            if not validation.issubset(development) or not train.issubset(development):
                raise ValueError("An inner fold left the outer development set")
    if len(seen) != len(set(seen)):
        raise ValueError("A participant is the outer test more than once")
    if set(seen) != set(manifest["subjects"]):
        raise ValueError("A participant never appears as the outer test")


def _assign(subjects: Sequence[str], folds: int, seed: int) -> dict[str, int]:
    ordered = sorted(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)
    return {name: index % folds for index, name in enumerate(ordered)}
