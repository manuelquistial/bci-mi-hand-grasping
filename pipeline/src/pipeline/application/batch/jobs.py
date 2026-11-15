"""Job identity, resume checks, and the comparison ranking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pipeline.application.batch.names import METRIC_COLUMNS


def job_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("dataset") or ""),
        str(row.get("spatial") or "none"),
        str(row.get("alignment") or "none"),
        str(row.get("features") or ""),
        str(row.get("evaluation") or ""),
        str(row.get("classifier") or ""),
    )


def same_job(
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


def row_exists(
    results: Sequence[dict[str, Any]],
    dataset: str,
    spatial: str,
    alignment: str,
    feature: str,
    scheme: str,
    classifier: str,
) -> bool:
    return any(
        same_job(row, dataset, spatial, alignment, feature, scheme, classifier)
        and not str(row.get("error") or "")
        for row in results
    )


def upsert(results: list[dict[str, Any]], row: dict[str, Any]) -> None:
    for index, existing in enumerate(results):
        if same_job(
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


def variant_done(
    dataset: str,
    spatial: str,
    alignment: str,
    jobs: Sequence[tuple[str, str, str]],
    results: Sequence[dict[str, Any]],
) -> bool:
    return all(
        row_exists(results, dataset, spatial, alignment, feature, scheme, classifier)
        for feature, scheme, classifier in jobs
    )


def dataset_done(
    dataset: str,
    spatials: Sequence[str],
    alignments: Sequence[str],
    jobs: Sequence[tuple[str, str, str]],
    results: Sequence[dict[str, Any]],
) -> bool:
    return all(
        variant_done(dataset, spatial_name, alignment_name, jobs, results)
        for spatial_name in spatials
        for alignment_name in alignments
    )


def all_done(
    datasets: Sequence[str],
    spatials: Sequence[str],
    alignments: Sequence[str],
    jobs: Sequence[tuple[str, str, str]],
    results: Sequence[dict[str, Any]],
) -> bool:
    return all(
        dataset_done(dataset, spatials, alignments, jobs, results)
        for dataset in datasets
    )


def rank_key(row: dict[str, Any]) -> tuple[bool, float, str]:
    metrics = row.get("metrics") or {}
    balanced = metrics.get("balanced_accuracy")
    accuracy = balanced if balanced is not None else row.get("accuracy")
    if accuracy is None:
        return (True, 0.0, str(row.get("classifier") or ""))
    return (False, -float(accuracy), str(row.get("classifier") or ""))


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
        ranked = sorted(groups[key], key=rank_key)
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
