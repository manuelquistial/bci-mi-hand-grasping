"""Score every spatial, alignment, feature, and classifier combination.

The pre-model request still trains one classifier. This batch loads a dataset
once, prepares each spatial method and alignment from those trials, then runs
the classifiers together. A row is written as soon as it finishes. A later
launch keeps rows that succeeded and retries rows that failed.

The implementation lives in ``pipeline.application.batch`` and
``pipeline.adapters.storage.batch_files``. This module keeps the import path
the CLI and the tests use.
"""

from __future__ import annotations

from collections.abc import Sequence

from pipeline.adapters.storage.batch_files import load_snapshot as _load_snapshot  # noqa: F401
from pipeline.application.batch.dry_run import dry_run_estimate
from pipeline.application.batch.jobs import comparison_rows
from pipeline.application.batch.names import (
    ALIGNMENTS,
    DATASETS,
    DEFAULT_EVALUATIONS,
    DEFAULT_FEATURES,
    METRIC_COLUMNS,
    classifier_names,
    iter_jobs,
)
from pipeline.application.batch.pool import ACTIVE as _ACTIVE  # noqa: F401
from pipeline.application.batch.pool import linux_executor as _linux_executor  # noqa: F401
from pipeline.application.batch.pool import run_forked_job as _run_forked_job  # noqa: F401
from pipeline.application.batch.progress import store_row as _store_row  # noqa: F401
from pipeline.application.batch.ram import available_ram as _available_ram
from pipeline.application.batch.ram import fit_workers as _budget_workers
from pipeline.application.batch.runner import run_batch, write_batch
from pipeline.application.batch.snapshot import empty_snapshot as _empty_snapshot  # noqa: F401
from pipeline.classify import SubjectTrials


def _fit_workers(requested: int, groups: Sequence[SubjectTrials]) -> int:
    """Worker count that fits the free RAM, read through this module's ``_available_ram``."""
    return _budget_workers(requested, groups, _available_ram())


__all__ = [
    "ALIGNMENTS",
    "DATASETS",
    "DEFAULT_EVALUATIONS",
    "DEFAULT_FEATURES",
    "METRIC_COLUMNS",
    "classifier_names",
    "comparison_rows",
    "dry_run_estimate",
    "iter_jobs",
    "run_batch",
    "write_batch",
]
