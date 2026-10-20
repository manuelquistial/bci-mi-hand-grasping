"""Classical classifiers used after the feature step.

Tree ensembles keep their own scaling. Margin and distance models are
standardized inside the evaluation fold, not here.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pipeline.params import ModelParams

_ESTIMATORS = {
    "random_forest": 80,
    "random_subspace_knn": 15,
    "boosted_trees": 40,
    "bagged_trees": 25,
    "rusboosted_trees": 30,
}

CLASSIFIERS = (
    "none",
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

_TREE = frozenset(
    {"decision_tree", "random_forest", "boosted_trees", "bagged_trees", "rusboosted_trees"}
)


def scales_features(name: str) -> bool:
    """Distance and margin models use a training-fold standardizer."""
    return name not in _TREE


def build_model(
    name: str,
    n_train: int,
    params: ModelParams | None = None,
) -> BaseEstimator:
    """One estimator for a feature vector already fit on the training fold."""
    tuned = params or ModelParams()
    neighbors = min(max(1, tuned.neighbors), max(1, n_train - 1))
    depth = _depth(name, tuned)
    trees = tuned.n_estimators if tuned.n_estimators and tuned.n_estimators > 0 else _ESTIMATORS.get(name, 25)
    penalty = float(tuned.regularization_c)
    if name in {"lda"}:
        return LinearDiscriminantAnalysis(solver="svd")
    if name == "slda":
        shrinkage = "auto" if tuned.shrinkage is None else float(tuned.shrinkage)
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
    if name == "logistic_regression":
        return LogisticRegression(C=penalty, max_iter=1000, random_state=42)
    if name == "linear_svm":
        return SVC(kernel="linear", C=penalty, random_state=42)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=neighbors, weights=tuned.knn_weights or "uniform")
    if name == "gaussian_nb":
        if tuned.var_smoothing is None:
            return GaussianNB()
        return GaussianNB(var_smoothing=float(tuned.var_smoothing))
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=_leaf(tuned),
            random_state=42,
        )
    if name == "svm_rbf":
        return SVC(kernel="rbf", C=penalty, gamma=tuned.gamma, random_state=42)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trees,
            max_depth=depth,
            min_samples_leaf=_leaf(tuned),
            random_state=42,
        )
    if name == "random_subspace_knn":
        return BaggingClassifier(
            estimator=KNeighborsClassifier(n_neighbors=neighbors),
            n_estimators=trees,
            max_features=tuned.max_features,
            bootstrap=False,
            random_state=42,
        )
    if name == "boosted_trees":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=_leaf(tuned),
                random_state=42,
            ),
            n_estimators=trees,
            learning_rate=1.0 if tuned.learning_rate is None else float(tuned.learning_rate),
            random_state=42,
        )
    if name == "bagged_trees":
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=_leaf(tuned),
                random_state=42,
            ),
            n_estimators=trees,
            random_state=42,
        )
    if name == "rusboosted_trees":
        return RusBoost(
            n_estimators=trees,
            max_depth=depth or 2,
            learning_rate=1.0 if tuned.learning_rate is None else float(tuned.learning_rate),
            random_state=42,
        )
    raise ValueError(f"Unknown classifier {name!r}")


def _leaf(params: ModelParams) -> int:
    if params.min_samples_leaf is None or params.min_samples_leaf < 1:
        return 1
    return int(params.min_samples_leaf)


def _depth(name: str, params: ModelParams) -> int | None:
    if params.max_depth is not None and params.max_depth > 0:
        return int(params.max_depth)
    if name in {"boosted_trees", "rusboosted_trees"}:
        return 2
    return None


class RusBoost(BaseEstimator, ClassifierMixin):
    """AdaBoost of shallow trees, each fit on a class-balanced draw."""

    def __init__(
        self,
        n_estimators: int = 30,
        max_depth: int = 2,
        learning_rate: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, samples: np.ndarray, labels: np.ndarray) -> "RusBoost":
        array = np.asarray(samples, dtype=np.float64)
        y = np.asarray(labels)
        classes, counts = np.unique(y, return_counts=True)
        if classes.size < 2:
            raise ValueError("Training fold has only one class")
        self.classes_ = classes
        minority = int(counts.min())
        rng = np.random.default_rng(self.random_state)
        weight = np.full(y.shape[0], 1.0 / y.shape[0])
        self._trees: list[DecisionTreeClassifier] = []
        self._alphas: list[float] = []
        for estimate in range(self.n_estimators):
            chosen = _balanced_draw(y, classes, minority, rng)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state + estimate,
            )
            tree.fit(array[chosen], y[chosen], sample_weight=weight[chosen])
            predicted = tree.predict(array)
            error = float(np.average(predicted != y, weights=weight))
            error = min(max(error, 1e-6), 1.0 - 1e-6)
            alpha = float(self.learning_rate) * float(
                np.log((1.0 - error) / error) + np.log(classes.size - 1)
            )
            if alpha <= 0:
                continue
            weight *= np.exp(alpha * (predicted != y))
            weight /= weight.sum()
            self._trees.append(tree)
            self._alphas.append(alpha)
        if not self._trees:
            raise ValueError("RUSBoost could not fit a tree on this fold")
        return self

    def predict(self, samples: np.ndarray) -> np.ndarray:
        labels = list(self.classes_)
        stacked = self._votes(np.asarray(samples, dtype=np.float64))
        return np.asarray(labels)[np.argmax(stacked, axis=1)]

    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        stacked = self._votes(np.asarray(samples, dtype=np.float64))
        shifted = stacked - stacked.max(axis=1, keepdims=True)
        exponential = np.exp(shifted)
        return exponential / exponential.sum(axis=1, keepdims=True)

    def _votes(self, array: np.ndarray) -> np.ndarray:
        scores = {str(label): np.zeros(array.shape[0]) for label in self.classes_}
        for tree, alpha in zip(self._trees, self._alphas):
            predicted = tree.predict(array)
            for label in self.classes_:
                scores[str(label)] += alpha * (predicted == label)
        return np.column_stack([scores[str(label)] for label in self.classes_])


def _balanced_draw(
    labels: np.ndarray,
    classes: np.ndarray,
    minority: int,
    rng: np.random.Generator,
) -> np.ndarray:
    chosen: list[np.ndarray] = []
    for label in classes:
        pool = np.flatnonzero(labels == label)
        pick = rng.choice(pool, size=minority, replace=pool.size < minority)
        chosen.append(pick)
    return np.concatenate(chosen)
