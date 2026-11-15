"""CSP filters on covariance stacks for the paper models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pipeline.classify.paper_linalg import _spd_batch
from pipeline.features.csp import assert_hand_pair, filters_from_covariances

CSP_PAIRS = (1, 2, 3)


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
