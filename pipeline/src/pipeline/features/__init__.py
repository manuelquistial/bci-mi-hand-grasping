"""One descriptor after Euclidean alignment. CSP and FBCSP stay separate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pipeline.features.csp import csp_features
from pipeline.features.fbcsp import fbcsp_features
from pipeline.features.psd import band_powers
from pipeline.features.tdp import time_domain_parameters
from pipeline.features.ts import time_series
from pipeline.params import ModelParams

METHODS = ("none", "ts", "tdp", "psd", "csp", "fbcsp")


@dataclass(frozen=True)
class FeatureView:
    """Rows and series ready for the pre-model chart."""

    rows: list[str]
    series: np.ndarray
    detail: str


def extract_features(
    method: str,
    trials: Sequence[np.ndarray],
    labels: Sequence[str],
    sample_rate_hz: float,
    channels: Sequence[str],
    params: ModelParams | None = None,
) -> FeatureView:
    """Run one descriptor. ``trials`` are ``(channels, samples)``."""
    name = (method or "none").strip().lower()
    tuned = params or ModelParams()
    if name == "ts":
        waveform = time_series(trials, tuned.ts_points)
        return FeatureView(
            rows=list(channels),
            series=waveform,
            detail=f"downsampled · {len(channels)} channels",
        )
    if name == "tdp":
        feature_names, values = time_domain_parameters(trials, labels, channels)
    elif name == "psd":
        feature_names, values = band_powers(
            trials, sample_rate_hz, channels, tuned.psd_bands()
        )
    elif name == "csp":
        feature_names, values = csp_features(trials, labels, tuned.csp_pairs)
    elif name == "fbcsp":
        feature_names, values = fbcsp_features(
            trials,
            labels,
            sample_rate_hz,
            pairs=tuned.csp_pairs,
            low_hz=tuned.fbcsp_low_hz,
            high_hz=tuned.fbcsp_high_hz,
            band_hz=tuned.fbcsp_band_hz,
            order=tuned.fbcsp_order,
            keep=tuned.fbcsp_keep,
        )
    else:
        raise ValueError(f"Unknown feature method {method!r}")
    detail = f"{len(labels)} trials · {len(feature_names)} features"
    if len(feature_names) <= 6:
        detail = f"{len(labels)} trials · {', '.join(feature_names)}"
    return FeatureView(rows=_row_names(labels), series=values.T, detail=detail)


def _row_names(labels: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    rows: list[str] = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        rows.append(f"{label} {seen[label]}")
    return rows
