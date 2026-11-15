"""Score one pipeline and build result or failure rows."""

from __future__ import annotations

import time
from typing import Any

from pipeline.classify import evaluate_classifier
from pipeline.params import ModelParams


def empty_score(started: float, message: str) -> dict[str, Any]:
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


def failed_row(
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
        **empty_score(time.perf_counter(), error),
    }


def score_job(
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
        base.update(empty_score(started, str(exc)))
        return base
    except Exception as exc:
        base.update(empty_score(started, str(exc)))
        return base
    base.update(
        {
            "accuracy": round(scored.accuracy, 4),
            "detail": scored.detail,
            "folds": [
                {"name": item.name, "accuracy": round(item.accuracy, 4)}
                for item in scored.folds
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
