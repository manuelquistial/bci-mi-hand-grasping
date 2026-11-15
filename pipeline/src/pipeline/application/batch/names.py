"""Dataset, feature, and classifier name lists for the batch."""

from __future__ import annotations

from collections.abc import Sequence

from pipeline.classify import CLASSIFIERS, SCHEMES
from pipeline.features import METHODS as FEATURE_METHODS


DATASETS = ("shuqfa2024", "liu2024_stroke")
DEFAULT_FEATURES = ("ts", "tdp", "psd", "csp", "fbcsp")
DEFAULT_EVALUATIONS = ("nested",)
ALIGNMENTS = ("none", "euclidean")
METRIC_COLUMNS = (
    "balanced_accuracy",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1",
    "kappa",
    "auc",
    "mcc",
)


def pick_names(
    requested: Sequence[str], allowed: Sequence[str], label: str
) -> list[str]:
    names: list[str] = []
    for name in requested:
        cleaned = name.strip().lower()
        if cleaned not in allowed:
            raise ValueError(f"Unknown {label} {name!r}")
        if cleaned not in names:
            names.append(cleaned)
    return names


def choice_list(
    value: str | Sequence[str],
    allowed: Sequence[str],
    label: str,
) -> list[str]:
    if isinstance(value, str):
        requested = [item.strip() for item in value.split(",") if item.strip()]
    else:
        requested = [str(item).strip() for item in value if str(item).strip()]
    if not requested:
        raise ValueError(f"The batch needs at least one {label}")
    return pick_names(requested, allowed, label)


def classifier_names(requested: Sequence[str] | None) -> list[str]:
    """Every classifier except ``none``, or the names the caller listed."""
    if not requested:
        return [name for name in CLASSIFIERS if name != "none"]
    names: list[str] = []
    for name in requested:
        cleaned = name.strip().lower()
        if cleaned not in CLASSIFIERS or cleaned == "none":
            raise ValueError(f"Unknown classifier {name!r}")
        if cleaned not in names:
            names.append(cleaned)
    return names


def iter_jobs(
    classifiers: Sequence[str],
    features: Sequence[str],
    evaluations: Sequence[str],
) -> list[tuple[str, str, str]]:
    """``(feature, evaluation, classifier)`` pairs."""
    feature_names = pick_names(features, FEATURE_METHODS, "feature method")
    scheme_names = pick_names(evaluations, SCHEMES, "evaluation")
    if classifiers and any(name == "none" for name in feature_names):
        raise ValueError("Choose a feature method before the classifier")
    jobs: list[tuple[str, str, str]] = []
    for feature in feature_names:
        if feature == "none":
            continue
        for scheme in scheme_names:
            for classifier in classifiers:
                jobs.append((feature, scheme, classifier))
    if not jobs:
        raise ValueError("The batch needs at least one classifier")
    return jobs
