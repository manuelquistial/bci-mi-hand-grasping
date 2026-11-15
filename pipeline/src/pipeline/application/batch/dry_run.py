"""Count model fits without loading trials or training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pipeline.application.batch.names import (
    ALIGNMENTS,
    DEFAULT_EVALUATIONS,
    DEFAULT_FEATURES,
    choice_list,
    classifier_names,
    iter_jobs,
)
from pipeline.params import ModelParams
from pipeline.spatial import METHODS as SPATIAL_METHODS


def dry_run_estimate(
    service: Any,
    *,
    datasets: Sequence[str],
    classifiers: Sequence[str] | None = None,
    features: Sequence[str] = DEFAULT_FEATURES,
    evaluations: Sequence[str] = DEFAULT_EVALUATIONS,
    spatial: str | Sequence[str] = "none",
    alignment: str | Sequence[str] = "none",
    params: ModelParams | None = None,
) -> dict[str, Any]:
    """Count model fits from the subject list. Does not load trials or train."""
    from pipeline.classify.search import search_space, select_configs
    from pipeline.classify.splits import fold_counts

    tuned = params or ModelParams()
    spatials = choice_list(spatial, SPATIAL_METHODS, "spatial method")
    alignments = choice_list(alignment, ALIGNMENTS, "alignment")
    variants = len(spatials) * len(alignments)
    names = classifier_names(classifiers)
    jobs = iter_jobs(names, features, evaluations)
    plans = []
    total = 0
    for dataset in datasets:
        listed = service.list_subjects(str(dataset))["subjects"]
        count = len(listed)
        for scheme in evaluations:
            if scheme not in {"nested", "nested_loso"}:
                continue
            try:
                plan = fold_counts(
                    count,
                    count if scheme == "nested_loso" else tuned.folds,
                    tuned.inner_folds,
                )
            except ValueError as exc:
                plans.append(
                    {"dataset": dataset, "evaluation": scheme, "error": str(exc)}
                )
                continue
            fits = 0
            searched = {"grid": 0, "random": 0}
            for feature, _scheme, classifier in jobs:
                if _scheme != scheme:
                    continue
                space = search_space(classifier, feature)
                _chosen, method, evaluated = select_configs(
                    space, tuned.search_budget, 42
                )
                searched[method] = searched.get(method, 0) + 1
                pipeline_fits = int(plan["outer_folds"]) * (
                    int(plan["inner_folds"]) * evaluated + 1
                )
                fits += pipeline_fits
            fits *= variants
            total += fits
            plans.append(
                {
                    "dataset": dataset,
                    "participants": count,
                    "evaluation": scheme,
                    "spatial_methods": list(spatials),
                    "alignments": list(alignments),
                    "variants": variants,
                    "outer_folds": plan["outer_folds"],
                    "inner_folds": plan["inner_folds"],
                    "adjustment": plan["adjustment"],
                    "fits": fits,
                    "exhaustive_pipelines": searched.get("grid", 0) * variants,
                    "random_pipelines": searched.get("random", 0) * variants,
                }
            )
    return {
        "dry_run": True,
        "search_budget": tuned.search_budget,
        "pipelines": len(jobs) * variants,
        "total_fits": total,
        "plans": plans,
        "note": (
            "Each fit is one training of one pipeline on one subject split. "
            "Pass --run to train. Looking at the outer test to pick a winner "
            "describes these datasets; a new held-out sample would be required "
            "to claim that winner in general."
        ),
    }
