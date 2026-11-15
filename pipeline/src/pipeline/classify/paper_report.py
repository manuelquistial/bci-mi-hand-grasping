"""Result rows, checkpoint snapshots, and the comparison table."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pipeline.classify.evaluate import Evaluation


_COMPARISON_COLUMNS = (
    "dataset",
    "spatial",
    "alignment",
    "features",
    "evaluation",
    "classifier",
    "participants",
    "seconds",
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
    "balanced_accuracy_std",
    "chosen",
    "confusion_labels",
    "confusion_matrix",
    "error",
)


def comparison_row(
    result: Evaluation,
    *,
    dataset: str,
    participants: int,
    seconds: float,
    error: str = "",
) -> dict[str, object]:
    """One JSON row with the same fields the nested batch stores for a flow."""
    report = result.report or {}
    metrics = {
        key: None if value is None else round(float(value), 4)
        for key, value in result.metrics.items()
    }
    return {
        "dataset": dataset,
        "participants": participants,
        "spatial": report.get("spatial") or "none",
        "alignment": report.get("alignment") or "none",
        "features": report.get("feature") or "covariance",
        "evaluation": "nested",
        "classifier": report.get("model") or "",
        "accuracy": round(float(result.accuracy), 4),
        "detail": result.detail,
        "folds": [
            {"name": item.name, "accuracy": round(item.accuracy, 4)}
            for item in result.folds
        ],
        "metrics": metrics,
        "confusion_labels": list(result.labels),
        "confusion_matrix": result.matrix,
        "seconds": round(seconds, 3),
        "error": error,
        "balanced_accuracy_std": report.get("balanced_accuracy_std"),
        "chosen": report.get("chosen") or [],
        "search": report.get("search"),
        "search_seconds": report.get("search_seconds"),
        "refit_seconds": report.get("refit_seconds"),
        "inference_seconds": report.get("inference_seconds"),
        "subjects_scored": report.get("subjects_scored"),
        "subject_scores": report.get("subjects") or [],
    }


def _paper_key(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("dataset") or ""),
        str(row.get("spatial") or "none"),
        str(row.get("alignment") or "none"),
        str(row.get("features") or ""),
        str(row.get("classifier") or ""),
    )


def _paper_row_done(
    rows: Sequence[dict[str, object]],
    dataset: str,
    spatial: str,
    alignment: str,
    model: str,
) -> bool:
    key = (dataset, spatial, alignment, "csp", model)
    return any(_paper_key(row) == key and not row.get("error") for row in rows)


def _upsert_paper_row(rows: list[dict[str, object]], row: dict[str, object]) -> None:
    key = _paper_key(row)
    for index, stored in enumerate(rows):
        if _paper_key(stored) == key:
            rows[index] = row
            return
    rows.append(row)


def _failed_paper_row(
    dataset: str,
    spatial: str,
    alignment: str,
    model: str,
    participants: int,
    seconds: float,
    error: str,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "participants": participants,
        "spatial": spatial,
        "alignment": alignment,
        "features": "csp",
        "evaluation": "nested",
        "classifier": model,
        "accuracy": None,
        "detail": "",
        "folds": [],
        "metrics": {},
        "confusion_labels": [],
        "confusion_matrix": [],
        "seconds": round(seconds, 3),
        "error": error,
        "balanced_accuracy_std": None,
        "chosen": [],
        "search": None,
        "search_seconds": None,
        "refit_seconds": None,
        "inference_seconds": None,
        "subjects_scored": 0,
        "subject_scores": [],
    }


def _load_paper_snapshot(path: Path) -> dict[str, object]:
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("results"), list):
            loaded["finished"] = False
            return loaded
    return {"kind": "paper-csp", "finished": False, "results": []}


def _write_paper_snapshot(path: Path, snapshot: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(snapshot)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    _write_paper_table(
        path.with_name(f"{path.stem}-comparison.csv"), payload["results"]
    )


def _write_paper_table(path: Path, rows: Sequence[dict[str, object]]) -> None:
    metric_names = (
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_COMPARISON_COLUMNS))
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            flat = {column: row.get(column, "") for column in _COMPARISON_COLUMNS}
            for name in metric_names:
                flat[name] = "" if metrics.get(name) is None else metrics.get(name, "")
            flat["chosen"] = json.dumps(_jsonable(row.get("chosen") or []))
            flat["confusion_labels"] = json.dumps(row.get("confusion_labels") or [])
            flat["confusion_matrix"] = json.dumps(row.get("confusion_matrix") or [])
            flat["error"] = row.get("error") or ""
            writer.writerow(
                {column: flat.get(column, "") for column in _COMPARISON_COLUMNS}
            )


def _jsonable(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
