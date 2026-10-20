"""Trial descriptors computed once, then reused by every fold and classifier.

The nested search used to rebuild spectra, time-domain parameters, and filter-bank
covariances on every candidate. Those values do not depend on the subject split,
so one pass per prepared cohort is enough. CSP and FBCSP still fit on the
training side of each fold; they just stop revisiting the raw samples.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from pipeline.alignment import trial_covariance
from pipeline.features.fbcsp import _bandpass, _bands_for
from pipeline.features.psd import _one
from pipeline.features.tdp import _trial_matrix
from pipeline.features.ts import downsample
from pipeline.params import ModelParams

_CACHE: dict[int, dict[str, Any]] = {}


def clear() -> None:
    """Drop descriptors from the previous cohort so their arrays can be freed."""
    _CACHE.clear()


def prepare(
    groups: Sequence[Any],
    sample_rate_hz: float,
    channels: Sequence[str],
    params: ModelParams | None = None,
) -> None:
    """Fill the cache for every trial. Later reads share these arrays."""
    clear()
    tuned = params or ModelParams()
    names = [str(channel) for channel in channels]
    rate = float(sample_rate_hz)
    bands = _bands_for(rate, tuned.fbcsp_low_hz, tuned.fbcsp_high_hz, tuned.fbcsp_band_hz)
    settings = {
        "channels": len(names),
        "rate": rate,
        "ts_points": int(tuned.ts_points),
        "psd_bands": tuple(tuned.psd_bands()),
        "fbcsp": (
            float(tuned.fbcsp_low_hz),
            float(tuned.fbcsp_high_hz),
            float(tuned.fbcsp_band_hz),
            int(tuned.fbcsp_order),
            tuple((float(low), float(high)) for low, high in bands),
        ),
    }
    trials = [trial for group in groups for trial in group.trials]
    if not trials:
        return

    def build(trial: np.ndarray) -> dict[str, Any]:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            array = np.asarray(trial, dtype=np.float64)
            covariance = trial_covariance(array)
            fbcsp = []
            for low, high in bands:
                filtered = _bandpass(array, rate, low, high, tuned.fbcsp_order)
                fbcsp.append(
                    (
                        float(low),
                        float(high),
                        trial_covariance(filtered),
                        int(filtered.shape[1]),
                    )
                )
            return {
                "settings": settings,
                "covariance": covariance,
                "samples": int(array.shape[1]),
                "tdp": _trial_matrix(array, names),
                "psd": _one(array, rate, len(names), settings["psd_bands"]),
                "ts": downsample(array, settings["ts_points"]),
                "fbcsp": fbcsp,
            }

    workers = min(len(trials), os.cpu_count() or 1)
    if workers <= 1:
        entries = [build(trial) for trial in trials]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            entries = list(pool.map(build, trials))
    for trial, entry in zip(trials, entries, strict=True):
        _CACHE[id(trial)] = entry
        array = np.asarray(trial)
        if array.flags.writeable and array.base is None:
            array.setflags(write=False)


def covariance_of(trial: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Centered covariance and sample count, when this trial was prepared."""
    entry = _CACHE.get(id(trial))
    if entry is None:
        return None
    return entry["covariance"], int(entry["samples"])


def tdp_matrix(trial: np.ndarray, channels: Sequence[str]) -> np.ndarray | None:
    entry = _match(trial, channels=len(channels))
    if entry is None:
        return None
    return entry["tdp"]


def psd_row(
    trial: np.ndarray,
    sample_rate_hz: float,
    channels: Sequence[str],
    bands: Sequence[tuple[str, float, float]] | None,
) -> np.ndarray | None:
    entry = _match(trial, channels=len(channels), rate=float(sample_rate_hz))
    if entry is None:
        return None
    if bands is not None and tuple(bands) != entry["settings"]["psd_bands"]:
        return None
    return entry["psd"]


def time_series(trial: np.ndarray, points: int) -> np.ndarray | None:
    entry = _CACHE.get(id(trial))
    if entry is None or int(points) != entry["settings"]["ts_points"]:
        return None
    return entry["ts"]


def fbcsp_bands(
    trial: np.ndarray,
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    band_hz: float,
    order: int,
) -> list[tuple[float, float, np.ndarray, int]] | None:
    entry = _match(trial, rate=float(sample_rate_hz))
    if entry is None:
        return None
    expected = (float(low_hz), float(high_hz), float(band_hz), int(order))
    if entry["settings"]["fbcsp"][:4] != expected:
        return None
    return list(entry["fbcsp"])


def _match(trial: np.ndarray, channels: int | None = None, rate: float | None = None) -> dict[str, Any] | None:
    entry = _CACHE.get(id(trial))
    if entry is None:
        return None
    settings = entry["settings"]
    if channels is not None and int(channels) != settings["channels"]:
        return None
    if rate is not None and float(rate) != settings["rate"]:
        return None
    return entry
