"""Nested cross-subject evaluation. Test participants never enter model selection."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from pipeline.classify.evaluate import (
    Evaluation,
    FoldScore,
    Prediction,
    SubjectTrials,
    _score,
    _with_metrics,
)
from pipeline.classify.search import configs_for
from pipeline.classify.splits import build_manifest, identify
from pipeline.features.encode import make_encoder
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
    feature_name = "covariance" if classifier == "riemann" else feature
    candidates, search_info = configs_for(
        classifier,
        feature_name,
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


def _select(
    classifier: str,
    feature: str,
    by_subject: dict[str, SubjectTrials],
    fold: dict[str, Any],
    candidates: Sequence[ModelParams],
    rate: float,
    channels: Sequence[str],
    workers: int,
) -> tuple[ModelParams, dict[str, Any], float, list[dict[str, Any]]]:
    started = time.perf_counter()

    def score_candidate(
        candidate: ModelParams,
    ) -> tuple[float | None, dict[str, Any], dict[str, Any]]:
        updates = _diff(candidate)
        scores: list[float] = []
        errors: list[str] = []
        for inner in fold["inner"]:
            try:
                predicted, spans = _fit_subjects(
                    classifier,
                    feature,
                    by_subject,
                    inner["train_subjects"],
                    inner["validation_subjects"],
                    updates,
                    rate,
                    channels,
                    workers,
                )
            except (ValueError, RuntimeError) as exc:
                errors.append(str(exc))
                continue
            for subject_id, _start, _stop, balanced, _error in _span_scores(spans, predicted):
                if balanced is None:
                    errors.append(f"{subject_id} could not be scored")
                    continue
                scores.append(balanced)
        mean = float(np.mean(scores)) if scores else None
        row = {
            "hyperparameters": _public_updates(updates),
            "balanced_accuracy": None if mean is None else round(mean, 4),
            "subjects": len(scores),
            "errors": errors,
        }
        return mean, updates, row

    if workers <= 1 or len(candidates) <= 1:
        scored = [score_candidate(candidate) for candidate in candidates]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(int(workers), len(candidates))) as pool:
            scored = list(pool.map(score_candidate, candidates))
    best_score: float | None = None
    best = candidates[0]
    best_updates: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for candidate, (mean, updates, row) in zip(candidates, scored, strict=True):
        rows.append(row)
        if mean is not None and (best_score is None or mean > best_score):
            best_score = mean
            best = candidate
            best_updates = updates
    if best_score is None:
        raise ValueError("No configuration could be scored on the inner subject folds")
    return best, best_updates, time.perf_counter() - started, rows


def _fit_subjects(
    classifier: str,
    feature: str,
    by_subject: dict[str, SubjectTrials],
    train_subjects: Sequence[str],
    test_subjects: Sequence[str],
    updates: dict[str, Any],
    rate: float,
    channels: Sequence[str],
    workers: int = 1,
) -> tuple[Prediction, list[tuple[str, int, int]]]:
    train = _join(by_subject, train_subjects)
    test = _join(by_subject, test_subjects)
    if len(set(train.labels)) < 2:
        raise ValueError(
            "Training participants do not cover two classes: "
            + ", ".join(sorted(set(train.labels)) or {"none"})
        )
    combined = SubjectTrials(
        "combined",
        train.trials + test.trials,
        train.labels + test.labels,
        train.dataset_id,
        train.session_id,
    )
    train_index = np.arange(len(train.labels))
    test_index = np.arange(len(train.labels), len(train.labels) + len(test.labels))
    params = _apply(updates)
    used_feature = "none" if classifier == "riemann" else feature
    if workers <= 1:
        predicted = _score(
            classifier,
            used_feature,
            combined,
            train_index,
            test_index,
            rate,
            channels,
            params,
        )
    else:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            predicted = _score(
                classifier,
                used_feature,
                combined,
                train_index,
                test_index,
                rate,
                channels,
                params,
            )
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for subject_id in test_subjects:
        width = len(by_subject[subject_id].labels)
        spans.append((subject_id, cursor, cursor + width))
        cursor += width
    return predicted, spans


def _join(by_subject: dict[str, SubjectTrials], names: Sequence[str]) -> SubjectTrials:
    trials: list[np.ndarray] = []
    labels: list[str] = []
    dataset_id = ""
    session_id = ""
    for name in names:
        group = by_subject[name]
        trials.extend(group.trials)
        labels.extend(group.labels)
        dataset_id = group.dataset_id
        session_id = group.session_id
    if not trials:
        raise ValueError("A subject split has no trials")
    return SubjectTrials("joined", trials, labels, dataset_id, session_id)


def _span_scores(
    spans: Sequence[tuple[str, int, int]],
    predicted: Prediction,
) -> list[tuple[str, int, int, float | None, str]]:
    rows = []
    for subject_id, start, stop in spans:
        true = predicted.y_true[start:stop]
        guess = predicted.y_pred[start:stop]
        if true.size == 0:
            rows.append((subject_id, start, stop, None, "no trials"))
            continue
        rows.append((subject_id, start, stop, float(balanced_accuracy_score(true, guess)), ""))
    return rows


def _subject_rows(
    people: Sequence[SubjectTrials],
    fold: int,
    spans: Sequence[tuple[str, int, int]],
    predicted: Prediction,
    dataset_id: str,
) -> list[dict[str, Any]]:
    lookup = {group.subject: group for group in people}
    rows = []
    for subject_id, start, stop, balanced, error in _span_scores(spans, predicted):
        group = lookup[subject_id]
        true = predicted.y_true[start:stop]
        guess = predicted.y_pred[start:stop]
        accuracy = None if error or true.size == 0 else float(np.mean(true == guess))
        rows.append(
            {
                "dataset_id": group.dataset_id or dataset_id,
                "subject_id": subject_id,
                "session_id": group.session_id,
                "fold": fold,
                "trials": int(true.size),
                "balanced_accuracy": None if balanced is None else round(balanced, 4),
                "accuracy": None if accuracy is None else round(accuracy, 4),
                "error": error,
            }
        )
    return rows


def _prediction_table(
    people: Sequence[SubjectTrials],
    fold: int,
    spans: Sequence[tuple[str, int, int]],
    predicted: Prediction,
    dataset_id: str,
) -> list[dict[str, Any]]:
    lookup = {group.subject: group for group in people}
    rows: list[dict[str, Any]] = []
    for subject_id, start, stop in spans:
        group = lookup[subject_id]
        for offset, trial_index in enumerate(range(start, stop)):
            trial_id = (
                group.trial_ids[offset]
                if offset < len(group.trial_ids)
                else f"{dataset_id}:{subject_id}:{group.session_id}:{offset}"
            )
            rows.append(
                {
                    "dataset_id": group.dataset_id or dataset_id,
                    "subject_id": subject_id,
                    "session_id": group.session_id,
                    "trial_id": trial_id,
                    "fold": fold,
                    "label": str(predicted.y_true[trial_index]),
                    "predicted": str(predicted.y_pred[trial_index]),
                }
            )
    return rows


def _tdp_channels(
    by_subject: dict[str, SubjectTrials],
    development: Sequence[str],
    channels: Sequence[str],
) -> list[str]:
    joined = _join(by_subject, development)
    encoder = make_encoder("tdp", 1.0, channels)
    encoder.fit(joined.trials, joined.labels)
    return list(getattr(encoder, "kept", []))


def _versions() -> dict[str, str]:
    import scipy
    import sklearn

    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _diff(candidate: ModelParams) -> dict[str, Any]:
    base = ModelParams()
    updates = {}
    for name in ModelParams.__dataclass_fields__:
        value = getattr(candidate, name)
        if value != getattr(base, name):
            updates[name] = value
    return updates


def _apply(updates: dict[str, Any]) -> ModelParams:
    from pipeline.params import with_updates

    return with_updates(ModelParams(), updates)


def _public_updates(updates: dict[str, Any]) -> dict[str, Any]:
    public = {}
    for key, value in updates.items():
        public[key] = value
    return public
