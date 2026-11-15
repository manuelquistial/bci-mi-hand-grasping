"""Checkpoint the snapshot and store finished rows."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pipeline.adapters.storage.batch_files import (
    append_full_row,
    rows_file,
    slim_row,
    write_checkpoint_file,
    write_table,
)
from pipeline.application.batch.jobs import comparison_rows, row_exists, upsert
from pipeline.application.batch.scoring import failed_row
from pipeline.application.batch.snapshot import now


def checkpoint(
    path: Path | None,
    snapshot: dict[str, Any],
    prior_seconds: float,
    started: float,
) -> None:
    snapshot["updated_at"] = now()
    snapshot["elapsed_seconds"] = round(
        prior_seconds + (time.perf_counter() - started), 3
    )
    snapshot["ranking"] = comparison_rows(snapshot["results"])
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_checkpoint_file(destination, snapshot)
    write_table(
        destination.with_name(f"{destination.stem}-comparison.csv"), snapshot["ranking"]
    )
    if snapshot.get("finished"):
        rows_file(destination).unlink(missing_ok=True)
        snapshot["_row_offsets"] = {}


def store_row(
    snapshot: dict[str, Any],
    row: dict[str, Any],
    output: Path | None,
    prior: float,
    started: float,
    on_result: Callable[[dict[str, Any]], None] | None,
) -> None:
    stored = row
    if output is not None and "predictions" in row:
        append_full_row(output, snapshot, row)
        stored = slim_row(row)
    upsert(snapshot["results"], stored)
    checkpoint(output, snapshot, prior, started)
    if on_result is not None:
        on_result(stored)


def fail_variant(
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
        if row_exists(
            snapshot["results"],
            dataset,
            spatial,
            alignment,
            feature,
            scheme,
            classifier,
        ):
            continue
        store_row(
            snapshot,
            failed_row(failed, feature, scheme, classifier, error),
            output,
            prior,
            started,
            on_result,
        )
