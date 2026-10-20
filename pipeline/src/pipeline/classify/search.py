"""Hyperparameter grids. TS, TDP, and PSD stay at the selected fixed settings."""

from __future__ import annotations

from typing import Any

import numpy as np

from pipeline.params import ModelParams, with_updates

VECTOR_FEATURES = ("ts", "tdp", "psd", "csp", "fbcsp")
VECTOR_CLASSIFIERS = (
    "lda",
    "slda",
    "logistic_regression",
    "linear_svm",
    "knn",
    "gaussian_nb",
    "decision_tree",
    "svm_rbf",
    "random_forest",
    "random_subspace_knn",
    "boosted_trees",
    "bagged_trees",
    "rusboosted_trees",
)


def pipeline_pairs(
    classifiers: list[str] | tuple[str, ...],
    features: list[str] | tuple[str, ...],
) -> list[tuple[str, str]]:
    """One job for each feature and classifier pair."""
    jobs: list[tuple[str, str]] = []
    vector = [name for name in classifiers if name in VECTOR_CLASSIFIERS]
    for feature in features:
        if feature not in VECTOR_FEATURES:
            continue
        for classifier in vector:
            jobs.append((feature, classifier))
    return jobs


def search_space(classifier: str, feature: str) -> list[dict[str, Any]]:
    """Cartesian product. LDA is the fixed reference."""
    grids = [_classifier_grid(classifier)]
    if feature in {"csp", "fbcsp"}:
        grids.append([{"csp_pairs": 1}, {"csp_pairs": 2}, {"csp_pairs": 3}])
    if feature == "fbcsp":
        grids.append([{"fbcsp_keep": 2}, {"fbcsp_keep": 4}, {"fbcsp_keep": 8}])
    space: list[dict[str, Any]] = [{}]
    for grid in grids:
        space = [{**left, **right} for left in space for right in grid]
    return space


def select_configs(
    space: list[dict[str, Any]],
    budget: int,
    seed: int,
) -> tuple[list[dict[str, Any]], str, int]:
    """Use every candidate when it fits in the budget. Otherwise sample with the seed."""
    limit = max(1, int(budget))
    if len(space) <= limit:
        return list(space), "grid", len(space)
    rng = np.random.default_rng(seed)
    picked = sorted(int(index) for index in rng.choice(len(space), size=limit, replace=False))
    return [space[index] for index in picked], "random", limit


def configs_for(
    classifier: str,
    feature: str,
    base: ModelParams,
    budget: int,
    seed: int,
) -> tuple[list[ModelParams], dict[str, Any]]:
    space = search_space(classifier, feature)
    chosen, method, evaluated = select_configs(space, budget, seed)
    return [with_updates(base, item) for item in chosen], {
        "method": method,
        "space_size": len(space),
        "evaluated": evaluated,
        "candidates": chosen,
    }


def _classifier_grid(classifier: str) -> list[dict[str, Any]]:
    if classifier == "lda":
        return [{}]
    if classifier == "slda":
        return [{"shrinkage": None}, {"shrinkage": 0.1}, {"shrinkage": 0.5}]
    if classifier in {"logistic_regression", "linear_svm"}:
        return [{"regularization_c": value} for value in (0.1, 1.0, 10.0)]
    if classifier == "svm_rbf":
        return [
            {"regularization_c": penalty, "gamma": gamma}
            for penalty in (0.1, 1.0, 10.0)
            for gamma in ("scale", 0.1)
        ]
    if classifier == "knn":
        return [
            {"neighbors": neighbors, "knn_weights": weights}
            for neighbors in (3, 5, 11)
            for weights in ("uniform", "distance")
        ]
    if classifier == "gaussian_nb":
        return [{"var_smoothing": value} for value in (1e-9, 1e-7)]
    if classifier == "decision_tree":
        return [
            {"max_depth": depth, "min_samples_leaf": leaf}
            for depth in (3, None)
            for leaf in (1, 5)
        ]
    if classifier == "random_forest":
        return [
            {"n_estimators": trees, "max_depth": depth, "min_samples_leaf": 1}
            for trees in (40, 80)
            for depth in (8, None)
        ]
    if classifier == "random_subspace_knn":
        return [
            {"n_estimators": trees, "max_features": features, "neighbors": neighbors}
            for trees in (10, 15)
            for features in (0.3, 0.5)
            for neighbors in (3, 5)
        ]
    if classifier == "boosted_trees":
        return [
            {"n_estimators": trees, "learning_rate": rate, "max_depth": 2}
            for trees in (20, 40)
            for rate in (0.5, 1.0)
        ]
    if classifier == "bagged_trees":
        return [
            {"n_estimators": trees, "max_depth": depth, "min_samples_leaf": 1}
            for trees in (15, 25)
            for depth in (4, None)
        ]
    if classifier == "rusboosted_trees":
        return [
            {"n_estimators": trees, "learning_rate": rate, "max_depth": 2}
            for trees in (20, 30)
            for rate in (0.5, 1.0)
        ]
    raise ValueError(f"Unknown classifier {classifier!r}")
