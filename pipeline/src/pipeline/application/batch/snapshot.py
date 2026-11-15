"""In-memory batch snapshot: datasets, manifests, and fold assignments."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pipeline.application.batch.cohort import covariances
from pipeline.params import ModelParams


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_snapshot() -> dict[str, Any]:
    return {
        "started_at": now(),
        "updated_at": "",
        "finished_at": "",
        "elapsed_seconds": 0,
        "finished": False,
        "datasets": [],
        "results": [],
        "ranking": [],
        "manifests": {},
    }


def stored_manifests(
    snapshot: dict[str, Any], dataset: str
) -> dict[str, dict[str, Any]]:
    """Reuse the fold assignment already written for this dataset."""
    saved = snapshot.get("manifests") or {}
    prefix = f"{dataset}:"
    return {
        key[len(prefix) :]: value
        for key, value in saved.items()
        if key.startswith(prefix) and isinstance(value, dict)
    }


def remember_dataset(
    snapshot: dict[str, Any],
    dataset: str,
    cohort: dict[str, Any] | None,
    load_seconds: float,
    error: str,
) -> None:
    stored = next(
        (item for item in snapshot["datasets"] if item["dataset"] == dataset), None
    )
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
        stored["covariances"] = covariances(cohort["groups"], cohort["channels"])


def shared_manifest(
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
