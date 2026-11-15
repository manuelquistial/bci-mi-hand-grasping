"""Window, band, and CSP-pair selection on the inner subject folds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock

import numpy as np
from threadpoolctl import threadpool_limits

from pipeline.classify.paper_banks import _CovBank, _TrialCovBank
from pipeline.classify.paper_csp import _csp_filters, _project
from pipeline.classify.paper_estimators import DiscriminantGeodesic, MinimumDistanceMean
from pipeline.classify.paper_people import _decide_people, _map_tasks, _per_subject


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


@dataclass(frozen=True)
class TwfbSearch:
    """Backtracking search over Liu's window and band grid."""

    population: int = 8
    generations: int = 6
    seed: int = 42
    windows: tuple[tuple[float, float], ...] = _WINDOWS
    bands: tuple[tuple[float, float], ...] = _BANDS


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
    pair_list: tuple[int | None, ...] = (
        (None,) if pairs is None else tuple(int(pair) for pair in pairs)
    )

    def score_spec(
        window: tuple[float, float],
        band: tuple[float, float],
        pair: int | None,
    ) -> float:
        scores: list[float] = []
        for fold in inner:
            covs, labels, _subjects = bank.gather(fold["train_subjects"], window, band)
            val_covs, val_labels, val_subjects = bank.gather(
                fold["validation_subjects"], window, band
            )
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
        mutant = current.astype(float) + scale * (
            historical.astype(float) - current.astype(float)
        )
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
            val_covs, val_labels, val_subjects = bank.gather(
                fold["validation_subjects"]
            )
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
