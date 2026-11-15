"""Score pending pipelines in-process or in forkserver workers sharing the trials."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline.application.batch.progress import store_row
from pipeline.application.batch.ram import release_ram
from pipeline.application.batch.scoring import failed_row, score_job
from pipeline.application.batch.snapshot import shared_manifest
from pipeline.params import ModelParams

# Fresh workers attach this cohort from shared memory. It is not inherited.
ACTIVE: dict[str, Any] = {}


def use_processes(workers: int) -> bool:
    """Fresh workers share the trials. macOS keeps this stack in one process."""
    return workers > 1 and sys.platform == "linux"


def linux_executor(
    workers: int, spec: dict[str, Any] | None = None
) -> ProcessPoolExecutor:
    """One process per pipeline. Each worker is a fresh interpreter."""
    from multiprocessing import get_context

    kwargs: dict[str, Any] = {
        "max_workers": workers,
        "mp_context": get_context("forkserver"),
    }
    if spec is not None:
        kwargs["initializer"] = attach_shared_cohort
        kwargs["initargs"] = (spec,)
    return ProcessPoolExecutor(**kwargs)


def attach_shared_cohort(spec: dict[str, Any]) -> None:
    """Map the shared trials into one fresh worker."""
    from pipeline.batch_share import attach_cohort

    ACTIVE["cohort"] = attach_cohort(spec)


def run_forked_job(
    payload: tuple[str, str, str, ModelParams | None, dict[str, Any] | None],
) -> dict[str, Any]:
    """One pipeline in a fresh process. Algebra stays on a single thread."""
    feature, scheme, classifier, params, manifest = payload
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return score_job(
            ACTIVE["cohort"],
            feature,
            scheme,
            classifier,
            1,
            params,
            manifest,
        )


def score_pending(
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
            shared_manifest(snapshot, manifests, cohort, dataset, scheme, params),
        )
        for feature, scheme, classifier in pending
    ]
    if len(payloads) <= 1 or not use_processes(workers):
        fit_workers = workers if len(payloads) <= 1 else 1
        for feature, scheme, classifier, job_params, manifest in payloads:
            row = score_job(
                cohort,
                feature,
                scheme,
                classifier,
                fit_workers,
                job_params,
                manifest,
            )
            store_row(snapshot, row, output, prior, started, on_result)
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
    release_ram()
    finished_jobs: set[tuple[str, str, str]] = set()
    try:
        with linux_executor(min(workers, len(payloads)), published.spec) as executor:
            waiting = {
                executor.submit(run_forked_job, payload): payload
                for payload in payloads
            }
            for finished in as_completed(waiting):
                feature, scheme, classifier, _params, _manifest = waiting[finished]
                try:
                    row = finished.result()
                except Exception as exc:
                    row = failed_row(cohort, feature, scheme, classifier, str(exc))
                finished_jobs.add((feature, scheme, classifier))
                store_row(snapshot, row, output, prior, started, on_result)
    except Exception as exc:
        for feature, scheme, classifier, _params, _manifest in payloads:
            if (feature, scheme, classifier) in finished_jobs:
                continue
            store_row(
                snapshot,
                failed_row(cohort, feature, scheme, classifier, str(exc)),
                output,
                prior,
                started,
                on_result,
            )
    finally:
        cohort["groups"] = []
        published.close()
        ACTIVE.clear()
        release_ram()
