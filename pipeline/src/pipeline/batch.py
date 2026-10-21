"""Score every spatial, alignment, feature, and classifier combination.

The pre-model request still trains one classifier. This batch loads a dataset
once, prepares each spatial method and alignment from those trials, then runs
the classifiers together. A row is written as soon as it finishes. A later
launch keeps rows that succeeded and retries rows that failed.
"""

from __future__ import annotations

import contextvars
import csv
import ctypes
import gc
import json
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.alignment import align_trials, mean_covariance
from pipeline.classify import CLASSIFIERS, SCHEMES, SubjectTrials, evaluate_classifier
from pipeline.features import METHODS as FEATURE_METHODS
from pipeline.params import ModelParams
from pipeline.spatial import METHODS as SPATIAL_METHODS
from pipeline.spatial import apply_spatial

DATASETS = ("shuqfa2024", "liu2024_stroke")
DEFAULT_FEATURES = ("ts", "tdp", "psd", "csp", "fbcsp")
DEFAULT_EVALUATIONS = ("nested",)
ALIGNMENTS = ("none", "euclidean")
# Fresh workers attach this cohort from shared memory. It is not inherited.
_ACTIVE: dict[str, Any] = {}
_WORKER_BYTES = 512 * 1024 * 1024
_SPILL: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "batch_row_spill",
    default=None,
)
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
    """``(feature, evaluation, classifier)`` pairs."""
    feature_names = _names(features, FEATURE_METHODS, "feature method")
    scheme_names = _names(evaluations, SCHEMES, "evaluation")
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


def comparison_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One block per dataset, spatial method, alignment, feature, and scheme."""
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in results:
        key = (
            str(row["dataset"]),
            str(row.get("spatial") or "none"),
            str(row.get("alignment") or "none"),
            str(row["features"]),
            str(row["evaluation"]),
        )
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
                    "spatial": row.get("spatial") or "none",
                    "alignment": row.get("alignment") or "none",
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
    spatial: str | Sequence[str] = "none",
    alignment: str | Sequence[str] = "none",
    notch_hz: float = 50.0,
    band_low_hz: float = 8.0,
    band_high_hz: float = 30.0,
    order: int = 4,
    params: ModelParams | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
    output: Path | None = None,
) -> list[dict[str, Any]]:
    """Load each dataset once and score every spatial and alignment variant.

    When ``output`` is set, each finished job replaces the file. A later run
    with the same path skips jobs that already succeeded and retries failures.
    """
    chosen = [str(dataset).strip() for dataset in datasets if str(dataset).strip()]
    if not chosen:
        raise ValueError("The batch needs at least one dataset")
    spatials = _choice_list(spatial, SPATIAL_METHODS, "spatial method")
    alignments = _choice_list(alignment, ALIGNMENTS, "alignment")
    names = classifier_names(classifiers)
    jobs = iter_jobs(names, features, evaluations)
    pool = max(1, int(workers))
    snapshot = _load_snapshot(output)
    prior = float(snapshot.get("elapsed_seconds") or 0)
    started = time.perf_counter()
    for dataset in chosen:
        if _dataset_done(dataset, spatials, alignments, jobs, snapshot["results"]):
            continue
        try:
            loaded = time.perf_counter()
            cohort = service.load_cohort(
                dataset=dataset,
                notch_hz=notch_hz,
                band_low_hz=band_low_hz,
                band_high_hz=band_high_hz,
                order=order,
                spatial="none",
                alignment="none",
                params=params,
                workers=pool,
            )
            load_seconds = round(time.perf_counter() - loaded, 3)
            _release_ram()
        except (ValueError, RuntimeError, OSError) as exc:
            _remember_dataset(snapshot, dataset, None, 0, str(exc))
            _checkpoint(output, snapshot, prior, started)
            continue
        _remember_dataset(snapshot, dataset, cohort, load_seconds, "")
        _checkpoint(output, snapshot, prior, started)
        manifests = _stored_manifests(snapshot, dataset)
        for spatial_name in spatials:
            for alignment_name in alignments:
                if _variant_done(
                    dataset,
                    spatial_name,
                    alignment_name,
                    jobs,
                    snapshot["results"],
                ):
                    continue
                try:
                    derived = _derive_cohort(cohort, spatial_name, alignment_name, params)
                except (ValueError, RuntimeError, OSError) as exc:
                    _fail_variant(
                        snapshot,
                        cohort,
                        jobs,
                        dataset,
                        spatial_name,
                        alignment_name,
                        str(exc),
                        output,
                        prior,
                        started,
                        on_result,
                    )
                    continue
                pending = [
                    (feature, scheme, classifier)
                    for feature, scheme, classifier in jobs
                    if not _row_exists(
                        snapshot["results"],
                        dataset,
                        spatial_name,
                        alignment_name,
                        feature,
                        scheme,
                        classifier,
                    )
                ]
                if not pending:
                    derived = None
                    _release_ram()
                    continue
                from pipeline.features.cache import clear, prepare

                prepare(
                    derived["groups"],
                    float(derived["rate"]),
                    derived["channels"],
                    params,
                )
                try:
                    _score_pending(
                        pending,
                        derived,
                        snapshot,
                        manifests,
                        dataset,
                        _fit_workers(pool, derived["groups"]),
                        params,
                        output,
                        prior,
                        started,
                        on_result,
                    )
                finally:
                    clear()
                    derived = None
                    _release_ram()
    snapshot["finished"] = _all_done(chosen, spatials, alignments, jobs, snapshot["results"])
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
    spatial: str | Sequence[str] = "none",
    alignment: str | Sequence[str] = "none",
    params: ModelParams | None = None,
) -> dict[str, Any]:
    """Count model fits from the subject list. Does not load trials or train."""
    from pipeline.classify.search import search_space, select_configs
    from pipeline.classify.splits import fold_counts

    tuned = params or ModelParams()
    spatials = _choice_list(spatial, SPATIAL_METHODS, "spatial method")
    alignments = _choice_list(alignment, ALIGNMENTS, "alignment")
    variants = len(spatials) * len(alignments)
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
                space = search_space(classifier, feature)
                _chosen, method, evaluated = select_configs(space, tuned.search_budget, 42)
                searched[method] = searched.get(method, 0) + 1
                pipeline_fits = int(plan["outer_folds"]) * (
                    int(plan["inner_folds"]) * evaluated + 1
                )
                fits += pipeline_fits
            fits *= variants
            total += fits
            plans.append(
                {
                    "dataset": dataset,
                    "participants": count,
                    "evaluation": scheme,
                    "spatial_methods": list(spatials),
                    "alignments": list(alignments),
                    "variants": variants,
                    "outer_folds": plan["outer_folds"],
                    "inner_folds": plan["inner_folds"],
                    "adjustment": plan["adjustment"],
                    "fits": fits,
                    "exhaustive_pipelines": searched.get("grid", 0) * variants,
                    "random_pipelines": searched.get("random", 0) * variants,
                }
            )
    return {
        "dry_run": True,
        "search_budget": tuned.search_budget,
        "pipelines": len(jobs) * variants,
        "total_fits": total,
        "plans": plans,
        "note": (
            "Each fit is one training of one pipeline on one subject split. "
            "Pass --run to train. Looking at the outer test to pick a winner "
            "describes these datasets; a new held-out sample would be required "
            "to claim that winner in general."
        ),
    }


def _score_pending(
    pending: Sequence[tuple[str, str, str]],
    cohort: dict[str, Any],
    snapshot: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    dataset: str,
    workers: int,
    params: ModelParams | None,
    output: Path | None,
    prior: float,
    started: float,
    on_result: Callable[[dict[str, Any]], None] | None,
) -> None:
    """Score the remaining pipelines. On Linux each one gets its own process."""
    payloads = [
        (
            feature,
            scheme,
            classifier,
            params,
            _shared_manifest(snapshot, manifests, cohort, dataset, scheme, params),
        )
        for feature, scheme, classifier in pending
    ]
    if len(payloads) <= 1 or not _use_processes(workers):
        fit_workers = workers if len(payloads) <= 1 else 1
        for feature, scheme, classifier, job_params, manifest in payloads:
            row = _score_job(
                cohort,
                feature,
                scheme,
                classifier,
                fit_workers,
                job_params,
                manifest,
            )
            _store_row(snapshot, row, output, prior, started, on_result)
        return
    from pipeline.batch_share import publish_cohort
    from pipeline.features.cache import clear

    published = publish_cohort(cohort)
    clear()
    # The private spatial copy can go. Workers read the shared block.
    old_groups = cohort["groups"]
    cohort["groups"] = published.groups
    published.groups = []
    del old_groups
    _release_ram()
    finished_jobs: set[tuple[str, str, str]] = set()
    try:
        with _linux_executor(min(workers, len(payloads)), published.spec) as executor:
            waiting = {executor.submit(_run_forked_job, payload): payload for payload in payloads}
            for finished in as_completed(waiting):
                feature, scheme, classifier, _params, _manifest = waiting[finished]
                try:
                    row = finished.result()
                except Exception as exc:
                    row = _failed_row(cohort, feature, scheme, classifier, str(exc))
                finished_jobs.add((feature, scheme, classifier))
                _store_row(snapshot, row, output, prior, started, on_result)
    except Exception as exc:
        for feature, scheme, classifier, _params, _manifest in payloads:
            if (feature, scheme, classifier) in finished_jobs:
                continue
            _store_row(
                snapshot,
                _failed_row(cohort, feature, scheme, classifier, str(exc)),
                output,
                prior,
                started,
                on_result,
            )
    finally:
        cohort["groups"] = []
        published.close()
        _ACTIVE.clear()
        _release_ram()


def _attach_shared_cohort(spec: dict[str, Any]) -> None:
    """Map the shared trials into one fresh worker."""
    from pipeline.batch_share import attach_cohort

    _ACTIVE["cohort"] = attach_cohort(spec)


def _run_forked_job(
    payload: tuple[str, str, str, ModelParams | None, dict[str, Any] | None],
) -> dict[str, Any]:
    """One pipeline in a fresh process. Algebra stays on a single thread."""
    feature, scheme, classifier, params, manifest = payload
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _score_job(
            _ACTIVE["cohort"],
            feature,
            scheme,
            classifier,
            1,
            params,
            manifest,
        )


def _store_row(
    snapshot: dict[str, Any],
    row: dict[str, Any],
    output: Path | None,
    prior: float,
    started: float,
    on_result: Callable[[dict[str, Any]], None] | None,
) -> None:
    stored = row
    if output is not None and "predictions" in row:
        _append_full_row(output, snapshot, row)
        stored = _slim(row)
    _upsert(snapshot["results"], stored)
    _checkpoint(output, snapshot, prior, started)
    if on_result is not None:
        on_result(stored)


def _failed_row(
    cohort: dict[str, Any],
    feature: str,
    scheme: str,
    classifier: str,
    error: str,
) -> dict[str, Any]:
    return {
        "dataset": cohort["dataset"],
        "participants": cohort["participants"],
        "spatial": cohort["spatial"],
        "alignment": cohort["alignment"],
        "features": feature,
        "evaluation": scheme,
        "classifier": classifier,
        **_empty_score(time.perf_counter(), error),
    }


def _use_processes(workers: int) -> bool:
    """Fresh workers share the trials. macOS keeps this stack in one process."""
    return workers > 1 and sys.platform == "linux"


def _linux_executor(workers: int, spec: dict[str, Any] | None = None) -> ProcessPoolExecutor:
    """One process per pipeline. Each worker is a fresh interpreter."""
    from multiprocessing import get_context

    kwargs: dict[str, Any] = {"max_workers": workers, "mp_context": get_context("forkserver")}
    if spec is not None:
        kwargs["initializer"] = _attach_shared_cohort
        kwargs["initargs"] = (spec,)
    return ProcessPoolExecutor(**kwargs)


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
            feature,
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


def _dataset_done(
    dataset: str,
    spatials: Sequence[str],
    alignments: Sequence[str],
    jobs: Sequence[tuple[str, str, str]],
    results: Sequence[dict[str, Any]],
) -> bool:
    return all(
        _variant_done(dataset, spatial_name, alignment_name, jobs, results)
        for spatial_name in spatials
        for alignment_name in alignments
    )


def _variant_done(
    dataset: str,
    spatial: str,
    alignment: str,
    jobs: Sequence[tuple[str, str, str]],
    results: Sequence[dict[str, Any]],
) -> bool:
    return all(
        _row_exists(results, dataset, spatial, alignment, feature, scheme, classifier)
        for feature, scheme, classifier in jobs
    )


def _all_done(
    datasets: Sequence[str],
    spatials: Sequence[str],
    alignments: Sequence[str],
    jobs: Sequence[tuple[str, str, str]],
    results: Sequence[dict[str, Any]],
) -> bool:
    return all(
        _dataset_done(dataset, spatials, alignments, jobs, results) for dataset in datasets
    )


def _row_exists(
    results: Sequence[dict[str, Any]],
    dataset: str,
    spatial: str,
    alignment: str,
    feature: str,
    scheme: str,
    classifier: str,
) -> bool:
    return any(
        _same_job(row, dataset, spatial, alignment, feature, scheme, classifier)
        and not str(row.get("error") or "")
        for row in results
    )


def _upsert(results: list[dict[str, Any]], row: dict[str, Any]) -> None:
    for index, existing in enumerate(results):
        if _same_job(
            existing,
            str(row["dataset"]),
            str(row.get("spatial") or "none"),
            str(row.get("alignment") or "none"),
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
    spatial: str,
    alignment: str,
    feature: str,
    scheme: str,
    classifier: str,
) -> bool:
    return (
        row.get("dataset") == dataset
        and str(row.get("spatial") or "none") == spatial
        and str(row.get("alignment") or "none") == alignment
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
    _write_checkpoint_file(destination, snapshot)
    _write_table(destination.with_name(f"{destination.stem}-comparison.csv"), snapshot["ranking"])
    if snapshot.get("finished"):
        _rows_path(destination).unlink(missing_ok=True)
        snapshot["_row_offsets"] = {}


def _write_checkpoint_file(path: Path, snapshot: dict[str, Any]) -> None:
    """Write the snapshot. Prediction rows are copied from the spill file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    offsets = snapshot.get("_row_offsets") or {}
    rows_path = _rows_path(path)
    header_keys = (
        "finished",
        "started_at",
        "updated_at",
        "finished_at",
        "elapsed_seconds",
        "datasets",
        "manifests",
    )
    results = list(snapshot.get("results") or [])
    with temporary.open("w", encoding="utf-8") as out:
        out.write("{\n")
        for key in header_keys:
            out.write(f"  {json.dumps(key)}: {json.dumps(snapshot.get(key))},\n")
        out.write('  "results": [\n')
        if rows_path.is_file() and offsets:
            with rows_path.open("rb") as src:
                for index, row in enumerate(results):
                    _write_result_line(out, src, offsets, row, last=index == len(results) - 1)
        else:
            for index, row in enumerate(results):
                suffix = "\n" if index == len(results) - 1 else ",\n"
                out.write("    " + json.dumps(row) + suffix)
        out.write("  ],\n")
        out.write(f'  "ranking": {json.dumps(snapshot.get("ranking") or [])}\n')
        out.write("}\n")
    temporary.replace(path)


def _write_result_line(
    out: Any,
    src: Any,
    offsets: dict[tuple[str, ...], int],
    row: dict[str, Any],
    *,
    last: bool,
) -> None:
    offset = offsets.get(_job_key(row))
    if offset is None:
        text = json.dumps(row)
    else:
        src.seek(offset)
        text = src.readline().decode("utf-8").strip()
    out.write("    " + text + ("\n" if last else ",\n"))


def _rows_path(path: Path) -> Path:
    return path.with_name(path.name + ".rows.jsonl")


def _job_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("dataset") or ""),
        str(row.get("spatial") or "none"),
        str(row.get("alignment") or "none"),
        str(row.get("features") or ""),
        str(row.get("evaluation") or ""),
        str(row.get("classifier") or ""),
    )


def _slim(row: dict[str, Any]) -> dict[str, Any]:
    """Drop per-trial predictions. The spill file keeps them for the checkpoint."""
    if "predictions" not in row:
        return row
    slim = dict(row)
    slim.pop("predictions", None)
    return slim


def _append_full_row(path: Path, snapshot: dict[str, Any], row: dict[str, Any]) -> None:
    destination = _rows_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n"
    with destination.open("ab") as handle:
        offset = handle.tell()
        handle.write(payload)
    snapshot.setdefault("_row_offsets", {})[_job_key(row)] = offset


def _spill_hook(data: dict[str, Any]) -> dict[str, Any]:
    state = _SPILL.get()
    if state is None or "predictions" not in data or "classifier" not in data or "features" not in data:
        return data
    handle = state["handle"]
    offset = handle.tell()
    handle.write(json.dumps(data, separators=(",", ":")).encode("utf-8") + b"\n")
    state["offsets"][_job_key(data)] = offset
    data.pop("predictions", None)
    return data


def _write_table(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "dataset",
        "spatial",
        "alignment",
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
    snapshot["_row_offsets"] = {}
    if path is None or not Path(path).is_file():
        return snapshot
    destination = Path(path)
    rows_path = _rows_path(destination)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    offsets: dict[tuple[str, str, str, str, str, str], int] = {}
    with rows_path.open("wb") as handle:
        state = {"handle": handle, "offsets": offsets}
        token = _SPILL.set(state)
        try:
            with destination.open(encoding="utf-8") as source:
                loaded = json.load(source, object_hook=_spill_hook)
        finally:
            _SPILL.reset(token)
    if not isinstance(loaded, dict):
        return snapshot
    snapshot["started_at"] = str(loaded.get("started_at") or snapshot["started_at"])
    snapshot["elapsed_seconds"] = float(loaded.get("elapsed_seconds") or 0)
    snapshot["finished"] = bool(loaded.get("finished") or False)
    snapshot["finished_at"] = str(loaded.get("finished_at") or "")
    snapshot["datasets"] = list(loaded.get("datasets") or [])
    snapshot["results"] = list(loaded.get("results") or [])
    snapshot["manifests"] = dict(loaded.get("manifests") or {})
    snapshot["_row_offsets"] = offsets
    del loaded
    _release_ram()
    return snapshot


def _release_ram() -> None:
    """Return free allocator pages so the next worker budget sees the real resident size."""
    gc.collect()
    if not sys.platform.startswith("linux"):
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        return


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
        "manifests": {},
    }


def _stored_manifests(snapshot: dict[str, Any], dataset: str) -> dict[str, dict[str, Any]]:
    """Reuse the fold assignment already written for this dataset."""
    saved = snapshot.get("manifests") or {}
    prefix = f"{dataset}:"
    return {
        key[len(prefix) :]: value
        for key, value in saved.items()
        if key.startswith(prefix) and isinstance(value, dict)
    }


def _choice_list(
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
    return _names(requested, allowed, label)


def _derive_cohort(
    cohort: dict[str, Any],
    spatial: str,
    alignment: str,
    params: ModelParams | None,
) -> dict[str, Any]:
    """Apply one spatial method and one alignment without reading the files again."""
    groups = list(cohort["groups"])
    channels = [str(channel) for channel in cohort["channels"]]
    if spatial != "none" or alignment != "none":
        groups, channels = _transform_groups(groups, channels, spatial, alignment, params)
    return {
        "dataset": cohort["dataset"],
        "groups": groups,
        "rate": cohort["rate"],
        "channels": channels,
        "spatial": spatial,
        "alignment": alignment,
        "participants": len(groups),
    }


def _transform_groups(
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


def _fail_variant(
    snapshot: dict[str, Any],
    cohort: dict[str, Any],
    jobs: Sequence[tuple[str, str, str]],
    dataset: str,
    spatial: str,
    alignment: str,
    error: str,
    output: Path | None,
    prior: float,
    started: float,
    on_result: Callable[[dict[str, Any]], None] | None,
) -> None:
    failed = {**cohort, "dataset": dataset, "spatial": spatial, "alignment": alignment}
    for feature, scheme, classifier in jobs:
        if _row_exists(snapshot["results"], dataset, spatial, alignment, feature, scheme, classifier):
            continue
        _store_row(
            snapshot,
            _failed_row(failed, feature, scheme, classifier, error),
            output,
            prior,
            started,
            on_result,
        )


def _fit_workers(requested: int, groups: Sequence[SubjectTrials]) -> int:
    """How many fresh workers fit beside one shared copy of the trials.

    Each worker is a new interpreter. It attaches the shared block instead of
    copying this process, and keeps a private working set for one model.
    """
    count = max(1, int(requested))
    available = _available_ram()
    if available is None:
        return count
    trial_bytes = sum(int(np.asarray(trial).nbytes) for group in groups for trial in group.trials)
    room = int(available * 0.5)
    if trial_bytes > room:
        return 1
    copies = room // _WORKER_BYTES
    return max(1, min(count, copies))


def _available_ram() -> int | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


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
