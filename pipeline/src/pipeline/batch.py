"""Score every classifier on each dataset from one shared cohort.

The pre-model request still trains one classifier. This batch loads a dataset
once, then runs the folds of each classifier on those trials. A row is written
as soon as it finishes, and a second launch skips rows that already succeeded.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.alignment import mean_covariance
from pipeline.classify import CLASSIFIERS, SCHEMES, SubjectTrials, evaluate_classifier
from pipeline.features import METHODS as FEATURE_METHODS
from pipeline.params import ModelParams

DATASETS = ("shuqfa2024", "liu2024_stroke")
DEFAULT_FEATURES = ("ts", "tdp", "psd", "csp", "fbcsp")
DEFAULT_EVALUATIONS = ("nested",)
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
            jobs.append(("covariance", scheme, "riemann"))
    if not jobs:
        raise ValueError("The batch needs at least one classifier")
    return jobs


def comparison_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One block per dataset, feature, and scheme. The highest accuracy is first."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in results:
        key = (str(row["dataset"]), str(row["features"]), str(row["evaluation"]))
        groups.setdefault(key, []).append(row)
    ordered: list[dict[str, Any]] = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=_rank_key)
        for place, row in enumerate(ranked, start=1):
            metrics = row.get("metrics") or {}
            ordered.append(
                {
                    "rank": place,
                    "dataset": row["dataset"],
                    "features": row["features"],
                    "evaluation": row["evaluation"],
                    "classifier": row["classifier"],
                    "participants": row.get("participants"),
                    "seconds": row.get("seconds"),
                    "confusion_labels": row.get("confusion_labels") or [],
                    "confusion_matrix": row.get("confusion_matrix") or [],
                    **{name: metrics.get(name) for name in METRIC_COLUMNS},
                    "error": row.get("error") or "",
                }
            )
    return ordered


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
    output: Path | None = None,
) -> list[dict[str, Any]]:
    """Load each dataset once and score every job on that cohort.

    When ``output`` is set, each finished job replaces the file. A later run
    with the same path skips jobs that already have a row.
    """
    chosen = [str(dataset).strip() for dataset in datasets if str(dataset).strip()]
    if not chosen:
        raise ValueError("The batch needs at least one dataset")
    names = classifier_names(classifiers)
    jobs = iter_jobs(names, features, evaluations)
    pool = max(1, int(workers))
    snapshot = _load_snapshot(output)
    prior = float(snapshot.get("elapsed_seconds") or 0)
    started = time.perf_counter()
    for dataset in chosen:
        if _dataset_done(dataset, jobs, snapshot["results"]):
            continue
        try:
            loaded = time.perf_counter()
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
            load_seconds = round(time.perf_counter() - loaded, 3)
        except (ValueError, RuntimeError, OSError) as exc:
            _remember_dataset(snapshot, dataset, None, 0, str(exc))
            _checkpoint(output, snapshot, prior, started)
            continue
        _remember_dataset(snapshot, dataset, cohort, load_seconds, "")
        _checkpoint(output, snapshot, prior, started)
        manifests: dict[str, dict[str, Any]] = {}
        for feature, scheme, classifier in jobs:
            if _row_exists(snapshot["results"], dataset, feature, scheme, classifier):
                continue
            manifest = _shared_manifest(
                snapshot,
                manifests,
                cohort,
                dataset,
                scheme,
                params,
            )
            row = _score_job(
                cohort,
                feature,
                scheme,
                classifier,
                pool,
                params,
                manifest,
            )
            _upsert(snapshot["results"], row)
            _checkpoint(output, snapshot, prior, started)
            if on_result is not None:
                on_result(row)
    snapshot["finished"] = _all_done(chosen, jobs, snapshot["results"])
    if snapshot["finished"]:
        snapshot["finished_at"] = _now()
    _checkpoint(output, snapshot, prior, started)
    return list(snapshot["results"])


def write_batch(path: Path, results: Sequence[dict[str, Any]]) -> None:
    """Write results, the comparison table, and any covariances already stored."""
    snapshot = _empty_snapshot()
    snapshot["results"] = list(results)
    snapshot["finished"] = True
    snapshot["finished_at"] = _now()
    _checkpoint(path, snapshot, 0, time.perf_counter())


def _shared_manifest(
    snapshot: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    cohort: dict[str, Any],
    dataset: str,
    scheme: str,
    params: ModelParams | None,
) -> dict[str, Any] | None:
    if scheme not in {"nested", "nested_loso"}:
        return None
    if scheme not in manifests:
        from pipeline.classify.splits import build_manifest

        tuned = params or ModelParams()
        manifests[scheme] = build_manifest(
            cohort["groups"],
            dataset_id=dataset,
            outer="loso" if scheme == "nested_loso" else "group",
            outer_folds=tuned.folds,
            inner_folds=tuned.inner_folds,
        )
        snapshot.setdefault("manifests", {})
        snapshot["manifests"][f"{dataset}:{scheme}"] = manifests[scheme]
    return manifests[scheme]


def dry_run_estimate(
    service: Any,
    *,
    datasets: Sequence[str],
    classifiers: Sequence[str] | None = None,
    features: Sequence[str] = DEFAULT_FEATURES,
    evaluations: Sequence[str] = DEFAULT_EVALUATIONS,
    params: ModelParams | None = None,
) -> dict[str, Any]:
    """Count model fits from the subject list. Does not load trials or train."""
    from pipeline.classify.search import search_space, select_configs
    from pipeline.classify.splits import fold_counts

    tuned = params or ModelParams()
    names = classifier_names(classifiers)
    jobs = iter_jobs(names, features, evaluations)
    plans = []
    total = 0
    for dataset in datasets:
        listed = service.list_subjects(str(dataset))["subjects"]
        count = len(listed)
        for scheme in evaluations:
            if scheme not in {"nested", "nested_loso"}:
                continue
            try:
                plan = fold_counts(
                    count,
                    count if scheme == "nested_loso" else tuned.folds,
                    tuned.inner_folds,
                )
            except ValueError as exc:
                plans.append({"dataset": dataset, "evaluation": scheme, "error": str(exc)})
                continue
            fits = 0
            searched = {"grid": 0, "random": 0}
            for feature, _scheme, classifier in jobs:
                if _scheme != scheme:
                    continue
                space_feature = "covariance" if classifier == "riemann" else feature
                space = search_space(classifier, space_feature)
                _chosen, method, evaluated = select_configs(space, tuned.search_budget, 42)
                searched[method] = searched.get(method, 0) + 1
                pipeline_fits = int(plan["outer_folds"]) * (
                    int(plan["inner_folds"]) * evaluated + 1
                )
                fits += pipeline_fits
            total += fits
            plans.append(
                {
                    "dataset": dataset,
                    "participants": count,
                    "evaluation": scheme,
                    "outer_folds": plan["outer_folds"],
                    "inner_folds": plan["inner_folds"],
                    "adjustment": plan["adjustment"],
                    "fits": fits,
                    "exhaustive_pipelines": searched.get("grid", 0),
                    "random_pipelines": searched.get("random", 0),
                }
            )
    return {
        "dry_run": True,
        "search_budget": tuned.search_budget,
        "pipelines": len(jobs),
        "total_fits": total,
        "plans": plans,
        "note": (
            "Each fit is one training of one pipeline on one subject split. "
            "Pass --run to train. Looking at the outer test to pick a winner "
            "describes these datasets; a new held-out sample would be required "
            "to claim that winner in general."
        ),
    }


def _score_job(
    cohort: dict[str, Any],
    feature: str,
    scheme: str,
    classifier: str,
    workers: int,
    params: ModelParams | None,
    manifest: dict[str, Any] | None = None,
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
            "none" if classifier == "riemann" else feature,
            cohort["groups"],
            sample_rate_hz=float(cohort["rate"]),
            channels=cohort["channels"],
            params=params,
            workers=workers,
            manifest=manifest,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        base.update(_empty_score(started, str(exc)))
        return base
    except Exception as exc:
        base.update(_empty_score(started, str(exc)))
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
    report = getattr(scored, "report", None)
    if isinstance(report, dict):
        stored = dict(report)
        stored.pop("manifest", None)
        base["search_seconds"] = stored.get("search_seconds")
        base["refit_seconds"] = stored.get("refit_seconds")
        base["inference_seconds"] = stored.get("inference_seconds")
        base["subjects_scored"] = stored.get("subjects_scored")
        base["subjects_failed"] = stored.get("subjects_failed")
        base["balanced_accuracy_std"] = stored.get("balanced_accuracy_std")
        base["subject_scores"] = stored.get("subjects")
        base["failures"] = stored.get("failures")
        base["predictions"] = stored.get("predictions")
        base["tdp_channels"] = stored.get("tdp_channels")
        base["chosen"] = stored.get("chosen")
        base["search"] = stored.get("search")
    return base


def _empty_score(started: float, message: str) -> dict[str, Any]:
    return {
        "accuracy": None,
        "detail": "",
        "folds": [],
        "metrics": {},
        "confusion_labels": [],
        "confusion_matrix": [],
        "seconds": round(time.perf_counter() - started, 3),
        "error": message,
    }


def _remember_dataset(
    snapshot: dict[str, Any],
    dataset: str,
    cohort: dict[str, Any] | None,
    load_seconds: float,
    error: str,
) -> None:
    stored = next((item for item in snapshot["datasets"] if item["dataset"] == dataset), None)
    if stored is None:
        stored = {"dataset": dataset}
        snapshot["datasets"].append(stored)
    stored["load_seconds"] = load_seconds
    stored["error"] = error
    if cohort is None:
        return
    stored["participants"] = cohort["participants"]
    stored["channels"] = list(cohort["channels"])
    stored["spatial"] = cohort["spatial"]
    stored["alignment"] = cohort["alignment"]
    if not stored.get("covariances"):
        stored["covariances"] = _covariances(cohort["groups"], cohort["channels"])


def _covariances(groups: Sequence[SubjectTrials], channels: Sequence[str]) -> list[dict[str, Any]]:
    """Mean spatial covariance of each participant, one matrix per person."""
    saved: list[dict[str, Any]] = []
    for group in groups:
        matrix = mean_covariance(group.trials)
        saved.append(
            {
                "subject": group.subject,
                "channels": list(channels),
                "trials": len(group.trials),
                "matrix": _matrix(matrix),
            }
        )
    return saved


def _matrix(values: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 6) for value in row] for row in np.asarray(values)]


def _dataset_done(dataset: str, jobs: Sequence[tuple[str, str, str]], results: Sequence[dict[str, Any]]) -> bool:
    return all(
        _row_exists(results, dataset, feature, scheme, classifier)
        for feature, scheme, classifier in jobs
    )


def _all_done(
    datasets: Sequence[str],
    jobs: Sequence[tuple[str, str, str]],
    results: Sequence[dict[str, Any]],
) -> bool:
    return all(_dataset_done(dataset, jobs, results) for dataset in datasets)


def _row_exists(
    results: Sequence[dict[str, Any]],
    dataset: str,
    feature: str,
    scheme: str,
    classifier: str,
) -> bool:
    return any(_same_job(row, dataset, feature, scheme, classifier) for row in results)


def _upsert(results: list[dict[str, Any]], row: dict[str, Any]) -> None:
    for index, existing in enumerate(results):
        if _same_job(
            existing,
            str(row["dataset"]),
            str(row["features"]),
            str(row["evaluation"]),
            str(row["classifier"]),
        ):
            results[index] = row
            return
    results.append(row)


def _same_job(
    row: dict[str, Any],
    dataset: str,
    feature: str,
    scheme: str,
    classifier: str,
) -> bool:
    return (
        row.get("dataset") == dataset
        and row.get("features") == feature
        and row.get("evaluation") == scheme
        and row.get("classifier") == classifier
    )


def _rank_key(row: dict[str, Any]) -> tuple[bool, float, str]:
    metrics = row.get("metrics") or {}
    balanced = metrics.get("balanced_accuracy")
    accuracy = balanced if balanced is not None else row.get("accuracy")
    if accuracy is None:
        return (True, 0.0, str(row.get("classifier") or ""))
    return (False, -float(accuracy), str(row.get("classifier") or ""))


def _checkpoint(
    path: Path | None,
    snapshot: dict[str, Any],
    prior_seconds: float,
    started: float,
) -> None:
    snapshot["updated_at"] = _now()
    snapshot["elapsed_seconds"] = round(prior_seconds + (time.perf_counter() - started), 3)
    snapshot["ranking"] = comparison_rows(snapshot["results"])
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    temporary.replace(destination)
    _write_table(destination.with_name(f"{destination.stem}-comparison.csv"), snapshot["ranking"])


def _write_table(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "dataset",
        "features",
        "evaluation",
        "classifier",
        "participants",
        "seconds",
        *METRIC_COLUMNS,
        "confusion_labels",
        "confusion_matrix",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "confusion_labels": json.dumps(row.get("confusion_labels") or []),
                    "confusion_matrix": json.dumps(row.get("confusion_matrix") or []),
                }
            )


def _load_snapshot(path: Path | None) -> dict[str, Any]:
    snapshot = _empty_snapshot()
    if path is None or not Path(path).is_file():
        return snapshot
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return snapshot
    snapshot["started_at"] = str(loaded.get("started_at") or snapshot["started_at"])
    snapshot["elapsed_seconds"] = float(loaded.get("elapsed_seconds") or 0)
    snapshot["datasets"] = list(loaded.get("datasets") or [])
    snapshot["results"] = list(loaded.get("results") or [])
    return snapshot


def _empty_snapshot() -> dict[str, Any]:
    return {
        "started_at": _now(),
        "updated_at": "",
        "finished_at": "",
        "elapsed_seconds": 0,
        "finished": False,
        "datasets": [],
        "results": [],
        "ranking": [],
    }


def _names(requested: Sequence[str], allowed: Sequence[str], label: str) -> list[str]:
    names: list[str] = []
    for name in requested:
        cleaned = name.strip().lower()
        if cleaned not in allowed:
            raise ValueError(f"Unknown {label} {name!r}")
        if cleaned not in names:
            names.append(cleaned)
    return names


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
