"""MDM and DGFMDRM estimators on covariance matrices."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from pipeline.classify.paper_covariance import _covariances, _window_covs
from pipeline.classify.paper_linalg import (
    _class_means,
    _class_names,
    _distance_many,
    _from_tangent,
    _riemann_mean,
    _tangent_vectors,
)


class MinimumDistanceMean:
    """One Riemannian mean per class. A trial takes the nearest mean."""

    def fit(
        self, trials: Sequence[np.ndarray], labels: Sequence[str]
    ) -> MinimumDistanceMean:
        return self.fit_covs(_covariances(trials), labels)

    def fit_covs(self, covs: np.ndarray, labels: Sequence[str]) -> MinimumDistanceMean:
        self.classes_, self.means_ = _class_means(covs, labels)
        return self

    def predict(self, trials: Sequence[np.ndarray]) -> list[str]:
        labels, _scores = self.decide_covs(_covariances(trials))
        return labels

    def score_matrix(self, trials: Sequence[np.ndarray]) -> np.ndarray:
        """Negative distance, one column per class. Higher is closer."""
        _labels, scores = self.decide_covs(_covariances(trials))
        return scores

    def decide_covs(self, covs: np.ndarray) -> tuple[list[str], np.ndarray]:
        distances = self._distances(covs)
        labels = [self.classes_[int(index)] for index in distances.argmin(axis=1)]
        return labels, -distances

    def _distances(self, covs: np.ndarray) -> np.ndarray:
        return np.column_stack([_distance_many(covs, mean) for mean in self.means_])


class DiscriminantGeodesic:
    """LTSA, Fisher geodesic filter, then minimum distance.

    With enough trials the tangent vectors are reduced by local tangent-space
    alignment and the decision is the nearest class mean in that filtered
    space. Otherwise the filter is mapped back to covariance matrices and the
    decision is the Riemannian minimum distance.
    """

    def fit(
        self,
        trials: Sequence[np.ndarray],
        labels: Sequence[str],
        *,
        sample_rate_hz: float | None = None,
        window: tuple[float, float] | None = None,
        band: tuple[float, float] | None = None,
    ) -> DiscriminantGeodesic:
        self._rate = sample_rate_hz
        self._window = window
        self._band = band
        if window is None and band is None:
            covs = _covariances(trials)
        else:
            covs = _window_covs(trials, float(sample_rate_hz), window, band)
        return self.fit_covs(covs, labels)

    def fit_covs(self, covs: np.ndarray, labels: Sequence[str]) -> DiscriminantGeodesic:
        self.classes_ = _class_names(labels)
        self._reference = _riemann_mean(covs)
        vectors = _tangent_vectors(covs, self._reference)
        reduced, self._ltsa = _fit_ltsa(vectors)
        if self._ltsa is not None:
            self._mode = "tangent"
            self._lda = _fisher(reduced, labels)
            coded = _decision_columns(self._lda, reduced)
            self._centers = {
                label: coded[np.asarray(labels) == label].mean(axis=0)
                for label in self.classes_
            }
            return self
        self._mode = "manifold"
        self._filter = _geodesic_filter(vectors, labels)
        filtered = _from_tangent(vectors @ self._filter, self._reference)
        self._means = dict(
            zip(self.classes_, _class_means(filtered, labels)[1], strict=True)
        )
        return self

    def predict(self, trials: Sequence[np.ndarray]) -> list[str]:
        return self.predict_covs(self._covs_for(trials))

    def predict_covs(self, covs: np.ndarray) -> list[str]:
        labels, _scores = self.decide_covs(covs)
        return labels

    def score_matrix(self, trials: Sequence[np.ndarray]) -> np.ndarray:
        return self.score_covs(self._covs_for(trials))

    def score_covs(self, covs: np.ndarray) -> np.ndarray:
        _labels, scores = self.decide_covs(covs)
        return scores

    def decide_covs(self, covs: np.ndarray) -> tuple[list[str], np.ndarray]:
        if self._mode == "tangent":
            coded = self._codes(covs)
            centers = np.stack([self._centers[label] for label in self.classes_])
            distances = np.linalg.norm(coded[:, None, :] - centers[None, :, :], axis=2)
        else:
            distances = self._manifold_distances(covs)
        labels = [self.classes_[int(index)] for index in distances.argmin(axis=1)]
        return labels, -distances

    def _covs_for(self, trials: Sequence[np.ndarray]) -> np.ndarray:
        if self._window is None and self._band is None:
            return _covariances(trials)
        return _window_covs(trials, float(self._rate), self._window, self._band)

    def _codes(self, covs: np.ndarray) -> np.ndarray:
        vectors = _tangent_vectors(covs, self._reference)
        return _decision_columns(self._lda, self._ltsa.transform(vectors))

    def _manifold_distances(self, covs: np.ndarray) -> np.ndarray:
        filtered = self._apply_filter(covs)
        return np.column_stack(
            [_distance_many(filtered, self._means[label]) for label in self.classes_]
        )

    def _apply_filter(self, covs: np.ndarray) -> np.ndarray:
        vectors = _tangent_vectors(covs, self._reference)
        return _from_tangent(vectors @ self._filter, self._reference)


def _fit_ltsa(vectors: np.ndarray) -> tuple[np.ndarray, object | None]:
    count, width = vectors.shape
    components = min(10, count // 4, width - 1)
    neighbors = min(count - 1, max(components + 1, 8))
    if components < 2 or count <= neighbors + 1 or width <= 12:
        return vectors, None
    from sklearn.manifold import LocallyLinearEmbedding

    try:
        embed = LocallyLinearEmbedding(
            n_neighbors=neighbors,
            n_components=components,
            method="ltsa",
            eigen_solver="dense",
            n_jobs=1,
            random_state=42,
        )
        return embed.fit_transform(vectors), embed
    except (ValueError, np.linalg.LinAlgError):
        return vectors, None


def _decision_columns(
    model: LinearDiscriminantAnalysis, features: np.ndarray
) -> np.ndarray:
    coded = np.asarray(model.decision_function(features), dtype=np.float64)
    if coded.ndim == 1:
        return coded.reshape(-1, 1)
    return coded


def _fisher(features: np.ndarray, labels: Sequence[str]) -> LinearDiscriminantAnalysis:
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(features, np.asarray(labels))
    return model


def _geodesic_filter(vectors: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    model = _fisher(vectors, labels)
    coef = np.atleast_2d(model.coef_)
    return coef.T @ np.linalg.pinv(coef @ coef.T) @ coef
