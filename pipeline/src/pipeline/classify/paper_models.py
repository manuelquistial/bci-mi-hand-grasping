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

import time
from collections.abc import Sequence
from pathlib import Path

from pipeline.classify.evaluate import Evaluation, SubjectTrials
from pipeline.classify.paper_csp import CSP_PAIRS, _csp_filters, _pair_values, _project
from pipeline.classify.paper_estimators import DiscriminantGeodesic, MinimumDistanceMean
from pipeline.classify.paper_evaluate import evaluate_cross_subject
from pipeline.classify.paper_linalg import _distance, _spd
from pipeline.classify.paper_people import _cores, _map_tasks
from pipeline.classify.paper_report import (
    _failed_paper_row,
    _load_paper_snapshot,
    _paper_row_done,
    _upsert_paper_row,
    _write_paper_snapshot,
    comparison_row,
)
from pipeline.classify.paper_search import TwfbSearch
from pipeline.classify.paper_signal import SIGNAL_ALIGNMENTS, apply_signal
from pipeline.spatial import METHODS as SPATIAL_METHODS

__all__ = [
    "CSP_PAIRS",
    "PAPER_MODELS",
    "SIGNAL_ALIGNMENTS",
    "DiscriminantGeodesic",
    "MinimumDistanceMean",
    "TwfbSearch",
    "apply_signal",
    "comparison_row",
    "csp_variants",
    "evaluate_cross_subject",
    "evaluate_csp",
    "evaluate_csp_grid",
    "score_csp_grid",
]

# Private helpers re-exported for tests and tools that import them from here.
_REEXPORTED = (
    _cores,
    _csp_filters,
    _distance,
    _map_tasks,
    _pair_values,
    _project,
    _spd,
)

PAPER_MODELS = ("mdm", "twfb_dgfmdr")


def csp_variants() -> tuple[tuple[str, str], ...]:
    """Every spatial method with Euclidean alignment and without it."""
    return tuple(
        (spatial, alignment)
        for spatial in SPATIAL_METHODS
        for alignment in SIGNAL_ALIGNMENTS
    )


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
