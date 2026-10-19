"""Common average reference: ``H = I - 1/C * 11ᵀ``."""

from __future__ import annotations

import numpy as np


def matrix(channels: list[str]) -> tuple[np.ndarray, list[str]]:
    count = len(channels)
    if count < 2:
        raise ValueError("CAR needs at least 2 channels")
    eye = np.eye(count, dtype=np.float64)
    return eye - np.ones((count, count), dtype=np.float64) / count, list(channels)
