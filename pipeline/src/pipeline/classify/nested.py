"""Nested cross-subject evaluation. Test participants never enter model selection."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from pipeline.classify.evaluate_types import Evaluation, FoldScore, Prediction, SubjectTrials
from pipeline.classify.fold_scoring import _with_metrics
from pipeline.classify.nested_fit import _public_updates, _select, _fit_subjects
from pipeline.classify.nested_rows import (
    _prediction_table,
    _subject_rows,
    _tdp_channels,
    _versions,
)
from pipeline.classify.search import configs_for
from pipeline.classify.splits import build_manifest, identify
from pipeline.params import ModelParams


def evaluate_nested(
    classifier: str,
    feature: str,
    groups: Sequence[SubjectTrials],
    *,
    sample_rate_hz: float,
    channels: Sequence[str],
    params: ModelParams | None = None,
    workers: int = 1,
    outer: str = "group",
    seed: int = 42,
    manifest: dict[str, Any] | None = None,
) -> Evaluation:
    """Outer subject folds, inner subject folds, then one refit on the development set."""
    tuned = params or ModelParams()
    dataset_ids = {group.dataset_id for group in groups if group.dataset_id}
    if len(dataset_ids) > 1:
        raise ValueError("Nested evaluation cannot mix datasets")
    dataset_id = next(iter(dataset_ids), "")
    people = [identify(group, dataset_id) for group in groups]
    by_subject = {group.subject: group for group in people}
    if manifest is None:
        manifest = build_manifest(
            people,
            dataset_id=dataset_id,
            outer=outer,
            outer_folds=tuned.folds,
            inner_folds=tuned.inner_folds,
            seed=seed,
        )
    else:
        from pipeline.classify.splits import check_manifest

        check_manifest(manifest)
    candidates, search_info = configs_for(
        classifier,
        feature,
        tuned,
        tuned.search_budget,
        seed,
    )
    predictions: list[Prediction] = []
    folds: list[FoldScore] = []
    subject_rows: list[dict[str, Any]] = []
    chosen_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    search_seconds = 0.0
    refit_seconds = 0.0
    predict_seconds = 0.0
    for fold in manifest["folds"]:
        try:
            winner, updates, elapsed, inner_rows = _select(
                classifier,
                feature,
                by_subject,
                fold,
                candidates,
                sample_rate_hz,
                channels,
                workers,
            )
        except (ValueError, RuntimeError) as exc:
            failures.append({"fold": fold["fold"], "stage": "search", "error": str(exc)})
            continue
        search_seconds += elapsed
        started = time.perf_counter()
        try:
            predicted, spans = _fit_subjects(
                classifier,
                feature,
                by_subject,
                fold["development_subjects"],
                fold["test_subjects"],
                updates,
                sample_rate_hz,
                channels,
            )
        except (ValueError, RuntimeError) as exc:
            failures.append({"fold": fold["fold"], "stage": "refit", "error": str(exc)})
            continue
        elapsed_fit = time.perf_counter() - started
        predict_seconds += predicted.predict_seconds
        refit_seconds += max(0.0, elapsed_fit - predicted.predict_seconds)
        predictions.append(predicted)
        scored = _subject_rows(
            people,
            fold["fold"],
            spans,
            predicted,
            dataset_id,
        )
        for row in scored:
            if row["error"]:
                failures.append(
                    {
                        "fold": fold["fold"],
                        "subject": row["subject_id"],
                        "stage": "test",
                        "error": row["error"],
                    }
                )
                continue
            folds.append(FoldScore(row["subject_id"], float(row["balanced_accuracy"])))
            subject_rows.append(row)
        prediction_rows.extend(
            _prediction_table(people, fold["fold"], spans, predicted, dataset_id)
        )
        chosen_rows.append(
            {
                "fold": fold["fold"],
                "hyperparameters": _public_updates(updates),
                "inner": inner_rows,
            }
        )
        if feature == "tdp":
            channel_rows.append(
                {
                    "fold": fold["fold"],
                    "channels": _tdp_channels(
                        by_subject,
                        fold["development_subjects"],
                        channels,
                    ),
                }
            )
    if not predictions:
        reason = failures[0]["error"] if failures else "no outer fold could be scored"
        raise ValueError(f"Nested evaluation failed: {reason}")
    pooled = _with_metrics(
        float(np.mean([item.accuracy for item in folds])) if folds else 0.0,
        "",
        folds,
        predictions,
    )
    scored_values = [float(row["balanced_accuracy"]) for row in subject_rows]
    scored_ids = {str(row["subject_id"]) for row in subject_rows}
    missing = sorted({group.subject for group in people} - scored_ids)
    mean = float(np.mean(scored_values)) if scored_values else 0.0
    spread = float(np.std(scored_values)) if scored_values else 0.0
    metrics = dict(pooled.metrics)
    metrics["balanced_accuracy"] = round(mean, 4)
    incomplete = len(missing)
    detail = (
        f"nested {manifest['outer']} · {manifest['outer_folds']} outer · "
        f"{manifest['inner_folds']} inner · {len(subject_rows)} participants · "
        f"balanced accuracy {mean:.0%} ± {spread:.0%}"
    )
    if incomplete:
        detail = f"{detail} · {incomplete} failed"
    if manifest["adjustment"]:
        detail = f"{detail} · {manifest['adjustment']}"
    report = {
        "manifest": manifest,
        "search": search_info,
        "chosen": chosen_rows,
        "subjects": subject_rows,
        "failures": failures,
        "predictions": prediction_rows,
        "tdp_channels": channel_rows,
        "search_seconds": round(search_seconds, 3),
        "refit_seconds": round(refit_seconds, 3),
        "inference_seconds": round(predict_seconds, 3),
        "subjects_scored": len(subject_rows),
        "subjects_failed": len(missing),
        "subjects_missing": missing,
        "versions": _versions(),
        "balanced_accuracy_mean": round(mean, 4),
        "balanced_accuracy_std": round(spread, 4),
        "pooled_accuracy": pooled.metrics.get("accuracy"),
    }
    return Evaluation(mean, detail, folds, pooled.labels, pooled.matrix, metrics, report)
