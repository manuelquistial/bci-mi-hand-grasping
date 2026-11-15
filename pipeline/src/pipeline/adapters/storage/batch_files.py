"""Batch checkpoint files: JSON snapshot, per-row spill file, comparison CSV."""

from __future__ import annotations

import contextvars
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pipeline.application.batch.jobs import job_key
from pipeline.application.batch.names import METRIC_COLUMNS
from pipeline.application.batch.ram import release_ram
from pipeline.application.batch.snapshot import empty_snapshot


SPILL: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "batch_row_spill",
    default=None,
)


def rows_file(path: Path) -> Path:
    return path.with_name(path.name + ".rows.jsonl")


def slim_row(row: dict[str, Any]) -> dict[str, Any]:
    """Drop per-trial predictions. The spill file keeps them for the checkpoint."""
    if "predictions" not in row:
        return row
    slim = dict(row)
    slim.pop("predictions", None)
    return slim


def append_full_row(path: Path, snapshot: dict[str, Any], row: dict[str, Any]) -> None:
    destination = rows_file(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n"
    with destination.open("ab") as handle:
        offset = handle.tell()
        handle.write(payload)
    snapshot.setdefault("_row_offsets", {})[job_key(row)] = offset


def spill_hook(data: dict[str, Any]) -> dict[str, Any]:
    state = SPILL.get()
    if (
        state is None
        or "predictions" not in data
        or "classifier" not in data
        or "features" not in data
    ):
        return data
    handle = state["handle"]
    offset = handle.tell()
    handle.write(json.dumps(data, separators=(",", ":")).encode("utf-8") + b"\n")
    state["offsets"][job_key(data)] = offset
    data.pop("predictions", None)
    return data


def write_table(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def write_checkpoint_file(path: Path, snapshot: dict[str, Any]) -> None:
    """Write the snapshot. Prediction rows are copied from the spill file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    offsets = snapshot.get("_row_offsets") or {}
    rows_path = rows_file(path)
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
                    write_result_line(
                        out, src, offsets, row, last=index == len(results) - 1
                    )
        else:
            for index, row in enumerate(results):
                suffix = "\n" if index == len(results) - 1 else ",\n"
                out.write("    " + json.dumps(row) + suffix)
        out.write("  ],\n")
        out.write(f'  "ranking": {json.dumps(snapshot.get("ranking") or [])}\n')
        out.write("}\n")
    temporary.replace(path)


def write_result_line(
    out: Any,
    src: Any,
    offsets: dict[tuple[str, ...], int],
    row: dict[str, Any],
    *,
    last: bool,
) -> None:
    offset = offsets.get(job_key(row))
    if offset is None:
        text = json.dumps(row)
    else:
        src.seek(offset)
        text = src.readline().decode("utf-8").strip()
    out.write("    " + text + ("\n" if last else ",\n"))


def load_snapshot(path: Path | None) -> dict[str, Any]:
    snapshot = empty_snapshot()
    snapshot["_row_offsets"] = {}
    if path is None or not Path(path).is_file():
        return snapshot
    destination = Path(path)
    rows_path = rows_file(destination)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    offsets: dict[tuple[str, str, str, str, str, str], int] = {}
    with rows_path.open("wb") as handle:
        state = {"handle": handle, "offsets": offsets}
        token = SPILL.set(state)
        try:
            with destination.open(encoding="utf-8") as source:
                loaded = json.load(source, object_hook=spill_hook)
        finally:
            SPILL.reset(token)
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
    release_ram()
    return snapshot
