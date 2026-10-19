"""Spatial step after the temporal filter. Each method is one matrix ``Y = MX``.

``none`` is the identity. CAR uses every channel. Hjorth, the large Laplacian,
and the spherical spline drop perimeter electrodes that are not surrounded.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pipeline.params import ModelParams
from pipeline.spatial.car import matrix as car_matrix
from pipeline.spatial.hjorth import matrix as hjorth_matrix
from pipeline.spatial.large_laplacian import matrix as large_matrix
from pipeline.spatial.spherical_spline import matrix as spline_matrix

METHODS = ("none", "car", "hjorth", "large_laplacian", "spherical_spline")


def apply_spatial(
    samples: np.ndarray,
    channels: Sequence[str],
    method: str,
    params: ModelParams | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Apply one spatial method to an array shaped ``(samples, channels)``."""
    name = (method or "none").strip().lower()
    labels = [str(channel) for channel in channels]
    array = np.asarray(samples, dtype=np.float64)
    tuned = params or ModelParams()
    if array.ndim != 2:
        raise ValueError("Spatial input must be a 2-D array")
    if array.shape[1] != len(labels):
        raise ValueError("Channel list does not match the signal width")
    if name == "none":
        return array, labels
    if name == "car":
        operator, kept = car_matrix(labels)
    elif name == "hjorth":
        operator, kept = hjorth_matrix(labels, degrees=tuned.hjorth_degrees)
    elif name == "large_laplacian":
        operator, kept = large_matrix(
            labels,
            radius_degrees=tuned.laplacian_radius_degrees,
            sigma_degrees=tuned.laplacian_sigma_degrees,
        )
    elif name == "spherical_spline":
        operator, kept = spline_matrix(
            labels,
            order=tuned.spline_order,
            ridge=tuned.spline_ridge,
        )
    else:
        raise ValueError(f"Unknown spatial method {method!r}")
    return array @ operator.T, kept
