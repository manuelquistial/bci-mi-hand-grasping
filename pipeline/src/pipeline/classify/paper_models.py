"""MDM and TWFB + DGFMDRM, scored with this project's cross-subject split.

The trials stay as the batch already cuts them: 4 s from the imagery marker,
notch then 8–30 Hz. These estimators are not part of ``CLASSIFIERS`` and the
nested batch does not call them. Each participant's covariance is built once
per window and band. The Riemannian mean, tangent map, and distances are one
batched eigendecomposition, and a generation's new candidates run together
with one BLAS thread each.

MDM is the minimum Riemannian distance to the class mean (Barachant 2012).
TWFB + DGFMDRM follows Liu 2024: a 1 s window and a 4 Hz band chosen on the
training participants, local tangent-space alignment, a Fisher geodesic
filter, then the minimum distance.

``evaluate_csp`` runs either model after one spatial method and one alignment.
The feature is CSP: the same filter bank as the batch, with 1, 2, or 3 pairs
chosen on the inner subject folds. The classifier still sees a covariance,
``Wᵀ C W``, the covariance of the CSP time series. The log-variance vector
used by the other classifiers is not an input here. A generation keeps Liu's
population of 8; each candidate is scored for every CSP pair at once, one
job per core and one BLAS thread per job. ``score_csp_grid`` writes one row
per combination with the same metrics as the nested batch.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from pipeline.alignment.euclidean import align_trials
from pipeline.classify.evaluate import (
    Evaluation,
    FoldScore,
    Prediction,
    SubjectTrials,
    _with_metrics,
)
from pipeline.classify.splits import build_manifest, identify
from pipeline.features.csp import assert_hand_pair, filters_from_covariances
from pipeline.params import ModelParams
from pipeline.spatial import METHODS as SPATIAL_METHODS
from pipeline.spatial import apply_spatial

_FLOOR = 1e-10
_WINDOWS = (
    (0.0, 1.0),
    (0.5, 1.5),
    (1.0, 2.0),
    (1.5, 2.5),
    (2.0, 3.0),
    (2.5, 3.5),
    (3.0, 4.0),
)
_BANDS = tuple((float(low), float(low + 4)) for low in range(8, 27))
CSP_PAIRS = (1, 2, 3)
SIGNAL_ALIGNMENTS = ("none", "euclidean")
PAPER_MODELS = ("mdm", "twfb_dgfmdr")
_COMPARISON_COLUMNS = (
    "dataset",
    "spatial",
    "alignment",
    "features",
    "evaluation",
    "classifier",
    "participants",
    "seconds",
    "balanced_accuracy",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1",
    "kappa",
    "auc",
    "mcc",
    "balanced_accuracy_std",
    "chosen",
    "confusion_labels",
    "confusion_matrix",
    "error",
)


@dataclass(frozen=True)
class TwfbSearch:
    """Backtracking search over Liu's window and band grid."""

    population: int = 8
    generations: int = 6
    seed: int = 42
    windows: tuple[tuple[float, float], ...] = _WINDOWS
    bands: tuple[tuple[float, float], ...] = _BANDS


def csp_variants() -> tuple[tuple[str, str], ...]:
    """Every spatial method with Euclidean alignment and without it."""
    return tuple((spatial, alignment) for spatial in SPATIAL_METHODS for alignment in SIGNAL_ALIGNMENTS)


def evaluate_csp_grid(
    name: str,
    groups: Sequence[SubjectTrials],
    sample_rate_hz: float,
    channels: Sequence[str],
    *,
    variants: Sequence[tuple[str, str]] | None = None,
    folds: int = 5,
    inner_folds: int = 3,
    seed: int = 42,
    search: TwfbSearch | None = None,
    pairs: Sequence[int] = CSP_PAIRS,
) -> list[Evaluation]:
    """One cross-subject score for each spatial method and alignment."""
    selected = tuple(variants) if variants is not None else csp_variants()
    return [
        evaluate_csp(
            name,
            groups,
            sample_rate_hz,
            channels,
            spatial=spatial,
            alignment=alignment,
            pairs=pairs,
            folds=folds,
            inner_folds=inner_folds,
            seed=seed,
            search=search,
        )
        for spatial, alignment in selected
    ]


def evaluate_csp(
    name: str,
    groups: Sequence[SubjectTrials],
    sample_rate_hz: float,
    channels: Sequence[str],
    *,
    spatial: str = "none",
    alignment: str = "none",
    pairs: Sequence[int] = CSP_PAIRS,
    folds: int = 5,
    inner_folds: int = 3,
    seed: int = 42,
    search: TwfbSearch | None = None,
) -> Evaluation:
    """Paper model on one spatial method, one alignment, and CSP."""
    return evaluate_cross_subject(
        name,
        groups,
        sample_rate_hz,
        folds=folds,
        inner_folds=inner_folds,
        seed=seed,
        search=search,
        channels=channels,
        spatial=spatial,
        alignment=alignment,
        pairs=pairs,
    )


def comparison_row(
    result: Evaluation,
    *,
    dataset: str,
    participants: int,
    seconds: float,
    error: str = "",
) -> dict[str, object]:
    """One JSON row with the same fields the nested batch stores for a flow."""
    report = result.report or {}
    metrics = {
        key: None if value is None else round(float(value), 4)
        for key, value in result.metrics.items()
    }
    return {
        "dataset": dataset,
        "participants": participants,
        "spatial": report.get("spatial") or "none",
        "alignment": report.get("alignment") or "none",
        "features": report.get("feature") or "covariance",
        "evaluation": "nested",
        "classifier": report.get("model") or "",
        "accuracy": round(float(result.accuracy), 4),
        "detail": result.detail,
        "folds": [
            {"name": item.name, "accuracy": round(item.accuracy, 4)} for item in result.folds
        ],
        "metrics": metrics,
        "confusion_labels": list(result.labels),
        "confusion_matrix": result.matrix,
        "seconds": round(seconds, 3),
        "error": error,
        "balanced_accuracy_std": report.get("balanced_accuracy_std"),
        "chosen": report.get("chosen") or [],
        "search": report.get("search"),
        "search_seconds": report.get("search_seconds"),
        "refit_seconds": report.get("refit_seconds"),
        "inference_seconds": report.get("inference_seconds"),
        "subjects_scored": report.get("subjects_scored"),
        "subject_scores": report.get("subjects") or [],
    }


def score_csp_grid(
    groups: Sequence[SubjectTrials],
    sample_rate_hz: float,
    channels: Sequence[str],
    output: Path,
    *,
    dataset: str,
    models: Sequence[str] = PAPER_MODELS,
    variants: Sequence[tuple[str, str]] | None = None,
    pairs: Sequence[int] = CSP_PAIRS,
    folds: int = 5,
    inner_folds: int = 3,
    seed: int = 42,
    search: TwfbSearch | None = None,
) -> list[dict[str, object]]:
    """Score each model, spatial method, and alignment, checkpointing every row.

    A later call with the same file skips combinations that already succeeded.
    """
    destination = Path(output)
    snapshot = _load_paper_snapshot(destination)
    selected = tuple(variants) if variants is not None else csp_variants()
    names = tuple(dict.fromkeys(name.strip().lower() for name in models))
    for name in names:
        if name not in PAPER_MODELS:
            raise ValueError(f"Unknown paper model {name!r}")
    for name in names:
        for spatial, alignment in selected:
            if _paper_row_done(snapshot["results"], dataset, spatial, alignment, name):
                continue
            started = time.perf_counter()
            try:
                result = evaluate_csp(
                    name,
                    groups,
                    sample_rate_hz,
                    channels,
                    spatial=spatial,
                    alignment=alignment,
                    pairs=pairs,
                    folds=folds,
                    inner_folds=inner_folds,
                    seed=seed,
                    search=search,
                )
                row = comparison_row(
                    result,
                    dataset=dataset,
                    participants=len(groups),
                    seconds=time.perf_counter() - started,
                )
            except (ValueError, RuntimeError, OSError) as exc:
                row = _failed_paper_row(
                    dataset,
                    spatial,
                    alignment,
                    name,
                    len(groups),
                    time.perf_counter() - started,
                    str(exc),
                )
            _upsert_paper_row(snapshot["results"], row)
            _write_paper_snapshot(destination, snapshot)
    return list(snapshot["results"])


def _paper_key(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("dataset") or ""),
        str(row.get("spatial") or "none"),
        str(row.get("alignment") or "none"),
        str(row.get("features") or ""),
        str(row.get("classifier") or ""),
    )


def _paper_row_done(
    rows: Sequence[dict[str, object]],
    dataset: str,
    spatial: str,
    alignment: str,
    model: str,
) -> bool:
    key = (dataset, spatial, alignment, "csp", model)
    return any(_paper_key(row) == key and not row.get("error") for row in rows)


def _upsert_paper_row(rows: list[dict[str, object]], row: dict[str, object]) -> None:
    key = _paper_key(row)
    for index, stored in enumerate(rows):
        if _paper_key(stored) == key:
            rows[index] = row
            return
    rows.append(row)


def _failed_paper_row(
    dataset: str,
    spatial: str,
    alignment: str,
    model: str,
    participants: int,
    seconds: float,
    error: str,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "participants": participants,
        "spatial": spatial,
        "alignment": alignment,
        "features": "csp",
        "evaluation": "nested",
        "classifier": model,
        "accuracy": None,
        "detail": "",
        "folds": [],
        "metrics": {},
        "confusion_labels": [],
        "confusion_matrix": [],
        "seconds": round(seconds, 3),
        "error": error,
        "balanced_accuracy_std": None,
        "chosen": [],
        "search": None,
        "search_seconds": None,
        "refit_seconds": None,
        "inference_seconds": None,
        "subjects_scored": 0,
        "subject_scores": [],
    }


def _load_paper_snapshot(path: Path) -> dict[str, object]:
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("results"), list):
            loaded["finished"] = False
            return loaded
    return {"kind": "paper-csp", "finished": False, "results": []}


def _write_paper_snapshot(path: Path, snapshot: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(snapshot)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    _write_paper_table(path.with_name(f"{path.stem}-comparison.csv"), payload["results"])


def _write_paper_table(path: Path, rows: Sequence[dict[str, object]]) -> None:
    metric_names = (
        "balanced_accuracy",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
        "kappa",
        "auc",
        "mcc",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_COMPARISON_COLUMNS))
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            flat = {column: row.get(column, "") for column in _COMPARISON_COLUMNS}
            for name in metric_names:
                flat[name] = "" if metrics.get(name) is None else metrics.get(name, "")
            flat["chosen"] = json.dumps(_jsonable(row.get("chosen") or []))
            flat["confusion_labels"] = json.dumps(row.get("confusion_labels") or [])
            flat["confusion_matrix"] = json.dumps(row.get("confusion_matrix") or [])
            flat["error"] = row.get("error") or ""
            writer.writerow({column: flat.get(column, "") for column in _COMPARISON_COLUMNS})


def _jsonable(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


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
            true, predicted, scores, subjects = _predict_people(model, by_subject, test_ids)
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
            model = MinimumDistanceMean().fit_covs(_project(train_covs, filters), train_labels)
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
            report["search"] = {"method": "grid", "evaluated": len(pair_values), "pairs": list(pair_values)}
        else:
            report["search"] = {
                "method": "bsa",
                "population": tuning.population,
                "generations": tuning.generations,
                "windows": len(tuning.windows),
                "bands": len(tuning.bands),
                "pairs": list(pair_values),
            }
    return Evaluation(mean, detail, fold_scores, pooled.labels, pooled.matrix, metrics, report)


class MinimumDistanceMean:
    """One Riemannian mean per class. A trial takes the nearest mean."""

    def fit(self, trials: Sequence[np.ndarray], labels: Sequence[str]) -> MinimumDistanceMean:
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
        self._means = dict(zip(self.classes_, _class_means(filtered, labels)[1], strict=True))
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


def _decision_columns(model: LinearDiscriminantAnalysis, features: np.ndarray) -> np.ndarray:
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


def _select_window(
    bank: _CovBank,
    inner: Sequence[dict[str, list[str]]],
    search: TwfbSearch,
    pairs: Sequence[int] | None = None,
) -> tuple[dict[str, object], float]:
    specs = [
        {"window": window, "band": band}
        for window in search.windows
        for band in search.bands
    ]
    cache: dict[int, float] = {}
    chosen_pairs: dict[int, int] = {}
    lock = Lock()
    pair_list: tuple[int | None, ...] = (None,) if pairs is None else tuple(int(pair) for pair in pairs)

    def score_spec(
        window: tuple[float, float],
        band: tuple[float, float],
        pair: int | None,
    ) -> float:
        scores: list[float] = []
        for fold in inner:
            covs, labels, _subjects = bank.gather(fold["train_subjects"], window, band)
            val_covs, val_labels, val_subjects = bank.gather(fold["validation_subjects"], window, band)
            if pair is not None:
                filters = _csp_filters(covs, labels, pair)
                covs = _project(covs, filters)
                val_covs = _project(val_covs, filters)
            model = DiscriminantGeodesic().fit_covs(covs, labels)
            true, predicted, _matrix, subjects = _decide_people(
                model,
                (val_covs, val_labels, val_subjects),
            )
            per_person = [item[1] for item in _per_subject(true, predicted, subjects)]
            if per_person:
                scores.append(float(np.mean(per_person)))
        return float(np.mean(scores)) if scores else 0.0

    def score_task(task: tuple[int, int | None]) -> tuple[int, int | None, float]:
        index, pair = task
        spec = specs[index]
        with threadpool_limits(limits=1):
            value = score_spec(spec["window"], spec["band"], pair)
        return index, pair, value

    def score_indexes(indexes: Sequence[int]) -> None:
        pending: list[int] = []
        with lock:
            for index in indexes:
                if index not in cache and index not in pending:
                    pending.append(index)
        if not pending:
            return
        tasks = [(index, pair) for index in pending for pair in pair_list]
        grouped: dict[int, dict[int | None, float]] = {}
        for index, pair, value in _map_tasks(tasks, score_task):
            grouped.setdefault(index, {})[pair] = value
        with lock:
            for index, scores_by_pair in grouped.items():
                if index in cache:
                    continue
                best_pair = pair_list[0]
                best_value = -1.0
                for pair in pair_list:
                    value = scores_by_pair.get(pair, 0.0)
                    if value > best_value:
                        best_value = value
                        best_pair = pair
                cache[index] = best_value
                if best_pair is not None:
                    chosen_pairs[index] = int(best_pair)

    if len(specs) == 1:
        score_indexes([0])
        chosen = dict(specs[0])
        if pairs is not None:
            chosen["csp_pairs"] = int(chosen_pairs[0])
        return chosen, cache[0]
    rng = np.random.default_rng(search.seed)
    population = min(len(specs), max(2, search.population))
    current = rng.choice(len(specs), size=population, replace=False)
    historical = current.copy()
    score_indexes(current)
    scores = np.asarray([cache[int(index)] for index in current])
    best_at = int(current[int(scores.argmax())])
    best_score = float(scores.max())
    for _generation in range(max(0, search.generations)):
        if rng.random() < rng.random():
            historical = rng.permutation(historical)
        scale = 3.0 * float(rng.normal())
        mutant = current.astype(float) + scale * (historical.astype(float) - current.astype(float))
        trial = current.copy()
        for row in range(population):
            if rng.random() < 0.5:
                trial[row] = int(np.clip(np.rint(mutant[row]), 0, len(specs) - 1))
        score_indexes(trial)
        trial_scores = np.asarray([cache[int(index)] for index in trial])
        improved = trial_scores >= scores
        current[improved] = trial[improved]
        scores[improved] = trial_scores[improved]
        if float(scores.max()) > best_score:
            best_score = float(scores.max())
            best_at = int(current[int(scores.argmax())])
    chosen = dict(specs[best_at])
    if pairs is not None:
        chosen["csp_pairs"] = int(chosen_pairs[best_at])
    return chosen, best_score


class _CovBank:
    """One covariance stack per participant, window, and band.

    The search asks for the same filtered trials on every inner fold. The
    stack is built once and then sliced.
    """

    def __init__(self, by_subject: dict[str, SubjectTrials], rate: float) -> None:
        self._people = by_subject
        self._rate = float(rate)
        self._cache: dict[tuple[str, tuple[float, float], tuple[float, float]], tuple[np.ndarray, list[str]]] = {}
        self._pending: dict[tuple[str, tuple[float, float], tuple[float, float]], Event] = {}
        self._errors: dict[tuple[str, tuple[float, float], tuple[float, float]], BaseException] = {}
        self._lock = Lock()

    def gather(
        self,
        subject_ids: Sequence[str],
        window: tuple[float, float],
        band: tuple[float, float],
    ) -> tuple[np.ndarray, list[str], list[str]]:
        blocks: list[np.ndarray] = []
        labels: list[str] = []
        subjects: list[str] = []
        for subject_id in subject_ids:
            covs, names = self._subject(subject_id, window, band)
            blocks.append(covs)
            labels.extend(names)
            subjects.extend([subject_id] * len(names))
        if not blocks:
            raise ValueError("A split has no trials")
        return np.concatenate(blocks, axis=0), labels, subjects

    def _subject(
        self,
        subject_id: str,
        window: tuple[float, float],
        band: tuple[float, float],
    ) -> tuple[np.ndarray, list[str]]:
        key = (subject_id, window, band)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            event = self._pending.get(key)
            if event is None:
                event = Event()
                self._pending[key] = event
                owner = True
            else:
                owner = False
        if not owner:
            event.wait()
            with self._lock:
                cached = self._cache.get(key)
                error = self._errors.get(key)
            if cached is not None:
                return cached
            if error is not None:
                raise error
            raise RuntimeError("covariance cache missed")
        try:
            group = self._people[subject_id]
            with threadpool_limits(limits=1):
                covs = _window_covs(group.trials, self._rate, window, band)
            stored = (covs, list(group.labels))
            with self._lock:
                self._cache[key] = stored
            return stored
        except BaseException as exc:
            with self._lock:
                self._errors[key] = exc
            raise
        finally:
            event.set()
            with self._lock:
                self._pending.pop(key, None)


def apply_signal(
    groups: Sequence[SubjectTrials],
    channels: Sequence[str],
    spatial: str,
    alignment: str,
) -> tuple[list[SubjectTrials], list[str]]:
    """Same spatial step and per-participant Euclidean alignment as the batch."""
    spatial_name = spatial.strip().lower()
    alignment_name = alignment.strip().lower()
    if spatial_name not in SPATIAL_METHODS:
        raise ValueError(f"Unknown spatial method {spatial!r}")
    if alignment_name not in SIGNAL_ALIGNMENTS:
        raise ValueError(f"Unknown alignment {alignment!r}")
    labels = [str(channel) for channel in channels]
    if spatial_name == "none" and alignment_name == "none":
        _check_width(groups, len(labels))
        return list(groups), labels
    tuned = ModelParams()
    transformed: list[SubjectTrials] = []
    kept = list(labels)
    for group in groups:
        trials: list[np.ndarray] = []
        for trial in group.trials:
            signal = np.ascontiguousarray(np.asarray(trial, dtype=np.float64).T)
            if signal.shape[1] != len(labels):
                raise ValueError("Channel list does not match the signal width")
            if spatial_name != "none":
                signal, kept = apply_spatial(signal, labels, spatial_name, tuned)
            trials.append(np.ascontiguousarray(signal.T))
        transformed.append(_replace_trials(group, trials))
    if alignment_name != "euclidean":
        return transformed, [str(name) for name in kept]
    aligned = []
    for group in transformed:
        trials, _weights = align_trials(group.trials)
        aligned.append(_replace_trials(group, trials))
    return aligned, [str(name) for name in kept]


def _replace_trials(group: SubjectTrials, trials: list[np.ndarray]) -> SubjectTrials:
    return SubjectTrials(
        group.subject,
        trials,
        list(group.labels),
        group.dataset_id,
        group.session_id,
        group.trial_ids,
    )


def _check_width(groups: Sequence[SubjectTrials], width: int) -> None:
    for group in groups:
        for trial in group.trials:
            if np.asarray(trial).shape[0] != width:
                raise ValueError("Channel list does not match the signal width")


def _pair_values(pairs: Sequence[int]) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(pair) for pair in pairs))
    if not values or any(pair < 1 for pair in values):
        raise ValueError("CSP pairs must be positive")
    return values


def _csp_filters(covs: np.ndarray, labels: Sequence[str], pairs: int) -> np.ndarray:
    assert_hand_pair(labels)
    coded = np.asarray(labels)
    if not np.any(coded == "left") or not np.any(coded == "right"):
        raise ValueError("CSP needs left and right trials")
    left = covs[coded == "left"].mean(axis=0)
    right = covs[coded == "right"].mean(axis=0)
    return filters_from_covariances(left, right, pairs=int(pairs))


def _project(covs: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Covariance of the CSP time series, ``Wᵀ C W``."""
    projected = np.einsum("ca,ncd,db->nab", filters, covs, filters, optimize=True)
    return _spd_batch(projected)


class _TrialCovBank:
    """Full-trial covariances, one stack per participant."""

    def __init__(self, by_subject: dict[str, SubjectTrials]) -> None:
        self._people = by_subject
        self._cache: dict[str, tuple[np.ndarray, list[str]]] = {}

    def warm(self) -> None:
        for subject_id in self._people:
            self._subject(subject_id)

    def gather(
        self,
        subject_ids: Sequence[str],
    ) -> tuple[np.ndarray, list[str], list[str]]:
        blocks: list[np.ndarray] = []
        labels: list[str] = []
        subjects: list[str] = []
        for subject_id in subject_ids:
            covs, names = self._subject(subject_id)
            blocks.append(covs)
            labels.extend(names)
            subjects.extend([subject_id] * len(names))
        if not blocks:
            raise ValueError("A split has no trials")
        return np.concatenate(blocks, axis=0), labels, subjects

    def _subject(self, subject_id: str) -> tuple[np.ndarray, list[str]]:
        cached = self._cache.get(subject_id)
        if cached is not None:
            return cached
        group = self._people[subject_id]
        stored = (_covariances(group.trials), list(group.labels))
        self._cache[subject_id] = stored
        return stored


def _cores() -> int:
    """One worker per core. Each worker keeps a single BLAS thread."""
    return max(1, os.cpu_count() or 1)


def _map_tasks(tasks: Sequence[object], function: Callable[[object], object]) -> list[object]:
    if not tasks:
        return []
    workers = min(len(tasks), _cores())
    if workers == 1:
        return [function(task) for task in tasks]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, tasks))


def _select_mdm_pairs(
    bank: _TrialCovBank,
    inner: Sequence[dict[str, list[str]]],
    pairs: Sequence[int],
) -> tuple[int, float]:
    bank.warm()
    tasks = [(int(pair), fold) for pair in pairs for fold in inner]

    def score_task(task: tuple[int, dict[str, list[str]]]) -> tuple[int, float]:
        pair, fold = task
        with threadpool_limits(limits=1):
            covs, labels, _subjects = bank.gather(fold["train_subjects"])
            filters = _csp_filters(covs, labels, pair)
            model = MinimumDistanceMean().fit_covs(_project(covs, filters), labels)
            val_covs, val_labels, val_subjects = bank.gather(fold["validation_subjects"])
            true, predicted, _matrix, subjects = _decide_people(
                model,
                (_project(val_covs, filters), val_labels, val_subjects),
            )
            per_person = [item[1] for item in _per_subject(true, predicted, subjects)]
        return pair, float(np.mean(per_person)) if per_person else 0.0

    grouped: dict[int, list[float]] = {}
    for pair, value in _map_tasks(tasks, score_task):
        grouped.setdefault(int(pair), []).append(float(value))
    best_pairs = int(pairs[0])
    best_score = -1.0
    for pair in pairs:
        values = grouped.get(int(pair), [])
        value = float(np.mean(values)) if values else 0.0
        if value > best_score:
            best_score = value
            best_pairs = int(pair)
    return best_pairs, best_score


def _decide_people(
    model: MinimumDistanceMean | DiscriminantGeodesic,
    gathered: tuple[np.ndarray, list[str], list[str]],
) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    covs, true, subjects = gathered
    predicted, scores = model.decide_covs(covs)
    return true, predicted, scores, subjects


def _collect(
    by_subject: dict[str, SubjectTrials],
    subject_ids: Sequence[str],
) -> tuple[list[np.ndarray], list[str]]:
    trials: list[np.ndarray] = []
    labels: list[str] = []
    for subject_id in subject_ids:
        group = by_subject[subject_id]
        trials.extend(group.trials)
        labels.extend(group.labels)
    if len(set(labels)) < 2:
        raise ValueError("A training split needs both hands")
    return trials, labels


def _predict_people(
    model: MinimumDistanceMean | DiscriminantGeodesic,
    by_subject: dict[str, SubjectTrials],
    subject_ids: Sequence[str],
) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    trials: list[np.ndarray] = []
    true: list[str] = []
    subjects: list[str] = []
    for subject_id in subject_ids:
        group = by_subject[subject_id]
        trials.extend(group.trials)
        true.extend(group.labels)
        subjects.extend([subject_id] * len(group.labels))
    predicted = model.predict(trials)
    return true, predicted, model.score_matrix(trials), subjects


def _per_subject(
    true: Sequence[str],
    predicted: Sequence[str],
    subjects: Sequence[str],
) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for subject_id in sorted(set(subjects)):
        mask = [subject == subject_id for subject in subjects]
        y_true = [label for label, keep in zip(true, mask, strict=True) if keep]
        y_pred = [label for label, keep in zip(predicted, mask, strict=True) if keep]
        if len(set(y_true)) < 2:
            continue
        rows.append((subject_id, float(balanced_accuracy_score(y_true, y_pred))))
    return rows


_SOS: dict[tuple[float, float, float], np.ndarray] = {}
_SOS_LOCK = Lock()


def _sos(rate: float, low: float, high: float) -> np.ndarray:
    key = (float(rate), float(low), float(high))
    with _SOS_LOCK:
        cached = _SOS.get(key)
        if cached is None:
            cached = butter(4, [low, high], btype="bandpass", fs=rate, output="sos")
            _SOS[key] = cached
        return cached


def _window_covs(
    trials: Sequence[np.ndarray],
    rate: float,
    window: tuple[float, float],
    band: tuple[float, float],
) -> np.ndarray:
    stacked = np.stack([np.asarray(trial, dtype=np.float64) for trial in trials])
    filtered = sosfiltfilt(_sos(rate, band[0], band[1]), stacked, axis=2)
    start = int(round(window[0] * rate))
    stop = int(round(window[1] * rate))
    piece = filtered[:, :, start:stop]
    if piece.shape[2] < 16:
        raise ValueError("The time window is shorter than 16 samples")
    return _covariances_batch(piece)


def _covariances(trials: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.stack([np.asarray(trial, dtype=np.float64) for trial in trials])
    return _covariances_batch(stacked)


def _covariances_batch(trials: np.ndarray) -> np.ndarray:
    """Centered covariances for ``(trials, channels, samples)``."""
    centered = trials - trials.mean(axis=2, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", centered, centered, optimize=True)
    covs /= max(1, trials.shape[2] - 1)
    return _spd_batch(covs)


def _class_names(labels: Sequence[str]) -> list[str]:
    return sorted(set(labels))


def _class_means(covs: np.ndarray, labels: Sequence[str]) -> tuple[list[str], list[np.ndarray]]:
    names = _class_names(labels)
    coded = np.asarray(labels)
    means = [_riemann_mean(covs[coded == name]) for name in names]
    return names, means


def _spd(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, _FLOOR)
    return vectors @ np.diag(values) @ vectors.T


def _inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(_spd(matrix))
    return vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T


def _sqrtm(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(_spd(matrix))
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


def _expm_sym(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    return vectors @ np.diag(np.exp(values)) @ vectors.T


def _spd_batch(covs: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (covs + np.swapaxes(covs, -1, -2))
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, _FLOOR)
    return np.einsum("nij,nj,nkj->nik", vectors, values, vectors, optimize=True)


def _batch_logm(covs: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (covs + np.swapaxes(covs, -1, -2)))
    values = np.maximum(values, _FLOOR)
    return np.einsum("nij,nj,nkj->nik", vectors, np.log(values), vectors, optimize=True)


def _batch_congruence(reference: np.ndarray, covs: np.ndarray) -> np.ndarray:
    whitener = _inv_sqrt(reference)
    return np.einsum("ij,njk,kl->nil", whitener, covs, whitener, optimize=True)


def _riemann_mean(covs: np.ndarray, iterations: int = 30) -> np.ndarray:
    current = _spd(np.mean(covs, axis=0))
    if len(covs) == 1:
        return current
    for _step in range(iterations):
        step = np.mean(_batch_logm(_batch_congruence(current, covs)), axis=0)
        half = _sqrtm(current)
        current = _spd(half @ _expm_sym(step) @ half)
        if float(np.linalg.norm(step, ord="fro")) < 1e-6:
            break
    return current


def _distance_many(covs: np.ndarray, mean: np.ndarray) -> np.ndarray:
    logged = _batch_logm(_batch_congruence(mean, covs))
    return np.sqrt(np.sum(logged * logged, axis=(1, 2)))


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(_distance_many(left.reshape(1, *left.shape), right)[0])


def _tangent_vectors(covs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    logged = _batch_logm(_batch_congruence(reference, covs))
    rows, cols = np.triu_indices(reference.shape[0])
    values = logged[:, rows, cols]
    scale = np.ones(rows.shape[0], dtype=np.float64)
    scale[rows != cols] = np.sqrt(2.0)
    return values * scale


def _from_tangent(vectors: np.ndarray, reference: np.ndarray) -> np.ndarray:
    half = _sqrtm(reference)
    size = reference.shape[0]
    rows, cols = np.triu_indices(size)
    values = np.asarray(vectors, dtype=np.float64).copy()
    values[:, rows != cols] /= np.sqrt(2.0)
    logged = np.zeros((len(vectors), size, size), dtype=np.float64)
    logged[:, rows, cols] = values
    logged[:, cols, rows] = values
    symmetric = 0.5 * (logged + np.swapaxes(logged, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    mapped_log = np.einsum("nij,nj,nkj->nik", eigvecs, np.exp(eigvals), eigvecs, optimize=True)
    mapped = np.einsum("ij,njk,kl->nil", half, mapped_log, half, optimize=True)
    return _spd_batch(mapped)

