"""Cross-subject evaluation of the paper models."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from pipeline.classify.evaluate import (
    Evaluation,
    FoldScore,
    Prediction,
    SubjectTrials,
    _with_metrics,
)
from pipeline.classify.paper_banks import _CovBank, _TrialCovBank
from pipeline.classify.paper_csp import _csp_filters, _pair_values, _project
from pipeline.classify.paper_estimators import DiscriminantGeodesic, MinimumDistanceMean
from pipeline.classify.paper_people import (
    _collect,
    _decide_people,
    _per_subject,
    _predict_people,
)
from pipeline.classify.paper_search import TwfbSearch, _select_mdm_pairs, _select_window
from pipeline.classify.paper_signal import apply_signal
from pipeline.classify.splits import build_manifest, identify


def evaluate_cross_subject(
    name: str,
    groups: Sequence[SubjectTrials],
    sample_rate_hz: float,
    *,
    folds: int = 5,
    inner_folds: int = 3,
    seed: int = 42,
    search: TwfbSearch | None = None,
    channels: Sequence[str] | None = None,
    spatial: str | None = None,
    alignment: str | None = None,
    pairs: Sequence[int] | None = None,
) -> Evaluation:
    """Same outer and inner subject folds as the nested batch.

    ``name`` is ``mdm`` or ``twfb_dgfmdr``. Test participants never enter the
    fit or the window search. ``pairs`` turns the feature into CSP and chooses
    the pair count on the inner subject folds.
    """
    model_name = name.strip().lower()
    if model_name not in {"mdm", "twfb_dgfmdr"}:
        raise ValueError(f"Unknown paper model {name!r}")
    dataset_ids = {group.dataset_id for group in groups if group.dataset_id}
    if len(dataset_ids) > 1:
        raise ValueError("Cross-subject evaluation cannot mix datasets")
    dataset_id = next(iter(dataset_ids), "")
    spatial_name: str | None = None
    alignment_name: str | None = None
    source: Sequence[SubjectTrials] = groups
    if spatial is not None or alignment is not None:
        if channels is None:
            raise ValueError("A spatial method or an alignment needs the channel names")
        spatial_name = (spatial or "none").strip().lower()
        alignment_name = (alignment or "none").strip().lower()
        source, _kept = apply_signal(groups, channels, spatial_name, alignment_name)
    pair_values = None if pairs is None else _pair_values(pairs)
    people = [identify(group, dataset_id) for group in source]
    manifest = build_manifest(
        people,
        dataset_id=dataset_id,
        outer="group",
        outer_folds=folds,
        inner_folds=inner_folds,
        seed=seed,
    )
    by_subject = {group.subject: group for group in people}
    tuning = search or TwfbSearch(seed=seed)
    predictions: list[Prediction] = []
    fold_scores: list[FoldScore] = []
    chosen_rows: list[dict[str, object]] = []
    search_seconds = 0.0
    refit_seconds = 0.0
    inference_seconds = 0.0
    for fold in manifest["folds"]:
        train_ids = list(fold["development_subjects"])
        test_ids = list(fold["test_subjects"])
        if model_name == "mdm" and pair_values is None:
            started = time.perf_counter()
            train_trials, train_labels = _collect(by_subject, train_ids)
            model = MinimumDistanceMean().fit(train_trials, train_labels)
            refit_seconds += time.perf_counter() - started
            started = time.perf_counter()
            true, predicted, scores, subjects = _predict_people(
                model, by_subject, test_ids
            )
            inference_seconds += time.perf_counter() - started
            chosen = {"model": "mdm"}
        elif model_name == "mdm":
            trial_bank = _TrialCovBank(by_subject)
            started = time.perf_counter()
            pair, score = _select_mdm_pairs(trial_bank, fold["inner"], pair_values)
            search_seconds += time.perf_counter() - started
            started = time.perf_counter()
            train_covs, train_labels, _subjects = trial_bank.gather(train_ids)
            filters = _csp_filters(train_covs, train_labels, pair)
            model = MinimumDistanceMean().fit_covs(
                _project(train_covs, filters), train_labels
            )
            refit_seconds += time.perf_counter() - started
            started = time.perf_counter()
            test_covs, test_labels, test_subjects = trial_bank.gather(test_ids)
            true, predicted, scores, subjects = _decide_people(
                model,
                (_project(test_covs, filters), test_labels, test_subjects),
            )
            inference_seconds += time.perf_counter() - started
            chosen = {
                "model": "mdm",
                "feature": "csp",
                "csp_pairs": pair,
                "inner_balanced_accuracy": score,
            }
        else:
            bank = _CovBank(by_subject, float(sample_rate_hz))
            started = time.perf_counter()
            spec, score = _select_window(bank, fold["inner"], tuning, pairs=pair_values)
            search_seconds += time.perf_counter() - started
            window = spec["window"]
            band = spec["band"]
            started = time.perf_counter()
            train_covs, train_labels, _subjects = bank.gather(train_ids, window, band)
            if pair_values is None:
                model = DiscriminantGeodesic().fit_covs(train_covs, train_labels)
            else:
                filters = _csp_filters(train_covs, train_labels, int(spec["csp_pairs"]))
                train_covs = _project(train_covs, filters)
                model = DiscriminantGeodesic().fit_covs(train_covs, train_labels)
            refit_seconds += time.perf_counter() - started
            started = time.perf_counter()
            test_covs, test_labels, test_subjects = bank.gather(test_ids, window, band)
            if pair_values is not None:
                test_covs = _project(test_covs, filters)
            true, predicted, scores, subjects = _decide_people(
                model,
                (test_covs, test_labels, test_subjects),
            )
            inference_seconds += time.perf_counter() - started
            chosen = {**spec, "inner_balanced_accuracy": score}
            if pair_values is not None:
                chosen["feature"] = "csp"
        if spatial_name is not None:
            chosen = {"spatial": spatial_name, "alignment": alignment_name, **chosen}
        predictions.append(
            Prediction(
                float(balanced_accuracy_score(true, predicted)),
                np.asarray(true),
                np.asarray(predicted),
                scores,
                list(model.classes_),
            )
        )
        for subject_id, balanced in _per_subject(true, predicted, subjects):
            fold_scores.append(FoldScore(subject_id, balanced))
        chosen_rows.append({"fold": fold["fold"], **chosen})
    values = [item.accuracy for item in fold_scores]
    mean = float(np.mean(values)) if values else 0.0
    spread = float(np.std(values)) if values else 0.0
    pooled = _with_metrics(mean, "", fold_scores, predictions)
    metrics = dict(pooled.metrics)
    metrics["balanced_accuracy"] = round(mean, 4)
    detail = (
        f"{model_name} · {manifest['outer_folds']} outer · "
        f"{manifest['inner_folds']} inner · {len(values)} participants · "
        f"balanced accuracy {mean:.0%} ± {spread:.0%}"
    )
    report = {
        "model": model_name,
        "manifest": manifest,
        "chosen": chosen_rows,
        "balanced_accuracy_mean": round(mean, 4),
        "balanced_accuracy_std": round(spread, 4),
        "subjects": [
            {"subject_id": item.name, "balanced_accuracy": round(item.accuracy, 4)}
            for item in fold_scores
        ],
        "search_seconds": round(search_seconds, 3),
        "refit_seconds": round(refit_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "subjects_scored": len(fold_scores),
    }
    if spatial_name is not None:
        report["spatial"] = spatial_name
        report["alignment"] = alignment_name
    if pair_values is not None:
        report["feature"] = "csp"
        if model_name == "mdm":
            report["search"] = {
                "method": "grid",
                "evaluated": len(pair_values),
                "pairs": list(pair_values),
            }
        else:
            report["search"] = {
                "method": "bsa",
                "population": tuning.population,
                "generations": tuning.generations,
                "windows": len(tuning.windows),
                "bands": len(tuning.bands),
                "pairs": list(pair_values),
            }
    return Evaluation(
        mean, detail, fold_scores, pooled.labels, pooled.matrix, metrics, report
    )
