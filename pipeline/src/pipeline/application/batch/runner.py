"""Run the batch: load each dataset once and score every variant."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pipeline.adapters.storage.batch_files import load_snapshot
from pipeline.application.batch.cohort import derive_cohort
from pipeline.application.batch.jobs import (
    all_done,
    dataset_done,
    row_exists,
    variant_done,
)
from pipeline.application.batch.names import (
    ALIGNMENTS,
    DEFAULT_EVALUATIONS,
    DEFAULT_FEATURES,
    choice_list,
    classifier_names,
    iter_jobs,
)
from pipeline.application.batch.pool import score_pending
from pipeline.application.batch.progress import checkpoint, fail_variant
from pipeline.application.batch.ram import available_ram, fit_workers, release_ram
from pipeline.application.batch.snapshot import (
    empty_snapshot,
    now,
    remember_dataset,
    stored_manifests,
)
from pipeline.params import ModelParams
from pipeline.spatial import METHODS as SPATIAL_METHODS


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
    spatials = choice_list(spatial, SPATIAL_METHODS, "spatial method")
    alignments = choice_list(alignment, ALIGNMENTS, "alignment")
    names = classifier_names(classifiers)
    jobs = iter_jobs(names, features, evaluations)
    pool = max(1, int(workers))
    snapshot = load_snapshot(output)
    prior = float(snapshot.get("elapsed_seconds") or 0)
    started = time.perf_counter()
    for dataset in chosen:
        if dataset_done(dataset, spatials, alignments, jobs, snapshot["results"]):
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
            release_ram()
        except (ValueError, RuntimeError, OSError) as exc:
            remember_dataset(snapshot, dataset, None, 0, str(exc))
            checkpoint(output, snapshot, prior, started)
            continue
        remember_dataset(snapshot, dataset, cohort, load_seconds, "")
        checkpoint(output, snapshot, prior, started)
        manifests = stored_manifests(snapshot, dataset)
        for spatial_name in spatials:
            for alignment_name in alignments:
                if variant_done(
                    dataset,
                    spatial_name,
                    alignment_name,
                    jobs,
                    snapshot["results"],
                ):
                    continue
                try:
                    derived = derive_cohort(
                        cohort, spatial_name, alignment_name, params
                    )
                except (ValueError, RuntimeError, OSError) as exc:
                    fail_variant(
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
                    if not row_exists(
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
                    release_ram()
                    continue
                from pipeline.features.cache import clear, prepare

                prepare(
                    derived["groups"],
                    float(derived["rate"]),
                    derived["channels"],
                    params,
                )
                try:
                    score_pending(
                        pending,
                        derived,
                        snapshot,
                        manifests,
                        dataset,
                        fit_workers(pool, derived["groups"], available_ram()),
                        params,
                        output,
                        prior,
                        started,
                        on_result,
                    )
                finally:
                    clear()
                    derived = None
                    release_ram()
    snapshot["finished"] = all_done(
        chosen, spatials, alignments, jobs, snapshot["results"]
    )
    if snapshot["finished"]:
        snapshot["finished_at"] = now()
    checkpoint(output, snapshot, prior, started)
    return list(snapshot["results"])


def write_batch(path: Path, results: Sequence[dict[str, Any]]) -> None:
    """Write results, the comparison table, and any covariances already stored."""
    snapshot = empty_snapshot()
    snapshot["results"] = list(results)
    snapshot["finished"] = True
    snapshot["finished_at"] = now()
    checkpoint(path, snapshot, 0, time.perf_counter())
