"""Score every classifier on each dataset from one shared cohort.

The pre-model request still trains one classifier. This batch loads a dataset
once, then runs the classifiers side by side in memory. Datasets follow one
another so the trials of both are not resident together.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pipeline.classify import CLASSIFIERS, SCHEMES, evaluate_classifier
from pipeline.features import METHODS as FEATURE_METHODS
from pipeline.params import ModelParams

DATASETS = ("shuqfa2024", "liu2024_stroke")
DEFAULT_FEATURES = ("csp", "fbcsp")
DEFAULT_EVALUATIONS = ("cross_validation", "loso")


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
    """``(feature, evaluation, classifier)`` pairs. Riemannian runs once per scheme."""
    feature_names = _names(features, FEATURE_METHODS, "feature method")
    scheme_names = _names(evaluations, SCHEMES, "evaluation")
    vector = [name for name in classifiers if name != "riemann"]
    if vector and any(name == "none" for name in feature_names):
        raise ValueError("Choose a feature method before the classifier")
    jobs: list[tuple[str, str, str]] = []
    for feature in feature_names:
        if feature == "none":
            continue
        for scheme in scheme_names:
            for classifier in vector:
                jobs.append((feature, scheme, classifier))
    if "riemann" in classifiers:
        for scheme in scheme_names:
            jobs.append(("none", scheme, "riemann"))
    if not jobs:
        raise ValueError("The batch needs at least one classifier")
    return jobs


def run_batch(
    service: Any,
    *,
    datasets: Sequence[str],
    classifiers: Sequence[str] | None = None,
    features: Sequence[str] = DEFAULT_FEATURES,
    evaluations: Sequence[str] = DEFAULT_EVALUATIONS,
    workers: int = 1,
    spatial: str = "none",
    alignment: str = "none",
    notch_hz: float = 50.0,
    band_low_hz: float = 8.0,
    band_high_hz: float = 30.0,
    order: int = 4,
    params: ModelParams | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Load each dataset once and score every job on that cohort."""
    chosen = [str(dataset).strip() for dataset in datasets if str(dataset).strip()]
    if not chosen:
        raise ValueError("The batch needs at least one dataset")
    names = classifier_names(classifiers)
    jobs = iter_jobs(names, features, evaluations)
    pool = max(1, int(workers))
    results: list[dict[str, Any]] = []
    for dataset in chosen:
        cohort = service.load_cohort(
            dataset=dataset,
            notch_hz=notch_hz,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            order=order,
            spatial=spatial,
            alignment=alignment,
            params=params,
            workers=pool,
        )
        for feature, scheme, classifier in jobs:
            row = _score_job(cohort, feature, scheme, classifier, pool, params)
            results.append(row)
            if on_result is not None:
                on_result(row)
    return results


def write_batch(path: Path, results: Sequence[dict[str, Any]]) -> None:
    """Write the batch table. One row per dataset, feature, scheme, and classifier."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"results": list(results)}, indent=2),
        encoding="utf-8",
    )


def _score_job(
    cohort: dict[str, Any],
    feature: str,
    scheme: str,
    classifier: str,
    workers: int,
    params: ModelParams | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    base = {
        "dataset": cohort["dataset"],
        "participants": cohort["participants"],
        "spatial": cohort["spatial"],
        "alignment": cohort["alignment"],
        "features": feature,
        "evaluation": scheme,
        "classifier": classifier,
    }
    try:
        scored = evaluate_classifier(
            classifier,
            scheme,
            feature,
            cohort["groups"],
            sample_rate_hz=float(cohort["rate"]),
            channels=cohort["channels"],
            params=params,
            workers=workers,
        )
    except (ValueError, RuntimeError) as exc:
        base.update(
            {
                "accuracy": None,
                "detail": "",
                "folds": [],
                "metrics": {},
                "confusion_labels": [],
                "confusion_matrix": [],
                "seconds": round(time.perf_counter() - started, 3),
                "error": str(exc),
            }
        )
        return base
    base.update(
        {
            "accuracy": round(scored.accuracy, 4),
            "detail": scored.detail,
            "folds": [
                {"name": item.name, "accuracy": round(item.accuracy, 4)} for item in scored.folds
            ],
            "metrics": {
                key: None if value is None else round(float(value), 4)
                for key, value in scored.metrics.items()
            },
            "confusion_labels": scored.labels,
            "confusion_matrix": scored.matrix,
            "seconds": round(time.perf_counter() - started, 3),
            "error": "",
        }
    )
    return base


def _names(requested: Sequence[str], allowed: Sequence[str], label: str) -> list[str]:
    names: list[str] = []
    for name in requested:
        cleaned = name.strip().lower()
        if cleaned not in allowed:
            raise ValueError(f"Unknown {label} {name!r}")
        if cleaned not in names:
            names.append(cleaned)
    return names
