"""Share one copy of the prepared trials with fresh worker processes.

Fork would hand each worker the parent's address space. These workers start
from a clean interpreter and attach the trial block by name, so the parent's
saved results are not copied once per core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from multiprocessing import shared_memory

from pipeline.classify import SubjectTrials
from pipeline.features.cache import entry_for, install

_ACTIVE_SHARE: dict[str, Any] = {}


@dataclass
class PublishedCohort:
    """Shared-memory owner. Closing it unlinks the blocks."""

    trials: shared_memory.SharedMemory
    cache: shared_memory.SharedMemory | None
    spec: dict[str, Any]
    groups: list[SubjectTrials]

    def close(self) -> None:
        _close_shared(self.trials)
        if self.cache is not None:
            _close_shared(self.cache)


def publish_cohort(cohort: dict[str, Any]) -> PublishedCohort:
    """Copy the cohort into shared memory and pack the descriptor cache beside it.

    `prepare` must already have filled the cache for these trial objects.
    """
    trial_raw, group_layout = _pack_trials(cohort["groups"])
    trial_shm, trial_buf = _allocate(trial_raw)
    cache_shm: shared_memory.SharedMemory | None = None
    try:
        groups = _groups_from_buffer(trial_buf, group_layout)
        entries = [entry_for(trial) for group in cohort["groups"] for trial in group.trials]
        cache_layout: list[dict[str, Any]] = []
        settings: dict[str, Any] | None = None
        if entries and all(entry is not None for entry in entries):
            settings = _plain_settings(entries[0]["settings"])
            cache_raw, cache_layout = _pack_entries(entries)
            cache_shm, _cache_buf = _allocate(cache_raw)
        else:
            cache_raw = np.zeros(0, dtype=np.float64)
        spec = {
            "trial_name": trial_shm.name,
            "trial_shape": [int(trial_buf.shape[0])],
            "groups": group_layout,
            "cache_name": None if cache_shm is None else cache_shm.name,
            "cache_shape": [int(cache_raw.shape[0])] if cache_shm is not None else [],
            "cache": cache_layout,
            "settings": settings,
            "cohort": {
                "dataset": cohort["dataset"],
                "rate": float(cohort["rate"]),
                "channels": [str(channel) for channel in cohort["channels"]],
                "spatial": cohort["spatial"],
                "alignment": cohort["alignment"],
                "participants": int(cohort["participants"]),
            },
        }
        for group in groups:
            for trial in group.trials:
                trial.setflags(write=False)
        return PublishedCohort(trials=trial_shm, cache=cache_shm, spec=spec, groups=groups)
    except Exception:
        _close_shared(trial_shm)
        if cache_shm is not None:
            _close_shared(cache_shm)
        raise


def attach_cohort(spec: dict[str, Any]) -> dict[str, Any]:
    """Map the shared trials into this process and install their descriptors."""
    from pipeline.features.cache import clear

    clear()
    trial_shm = _attach(str(spec["trial_name"]))
    trial_buf = np.ndarray(tuple(spec["trial_shape"]), dtype=np.float64, buffer=trial_shm.buf)
    groups = _groups_from_buffer(trial_buf, spec["groups"])
    for group in groups:
        for trial in group.trials:
            trial.setflags(write=False)
    shares: list[shared_memory.SharedMemory] = [trial_shm]
    cache_name = spec.get("cache_name")
    if cache_name and spec.get("settings") is not None:
        cache_shm = _attach(str(cache_name))
        shares.append(cache_shm)
        cache_buf = np.ndarray(tuple(spec["cache_shape"]), dtype=np.float64, buffer=cache_shm.buf)
        flat = [trial for group in groups for trial in group.trials]
        for trial, meta in zip(flat, spec["cache"], strict=True):
            install(trial, _entry_from_layout(cache_buf, meta, spec["settings"]))
    meta = dict(spec["cohort"])
    meta["groups"] = groups
    _ACTIVE_SHARE["shares"] = shares
    return meta


def _pack_trials(groups: list[SubjectTrials]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    pieces: list[np.ndarray] = []
    layout: list[dict[str, Any]] = []
    cursor = 0
    for group in groups:
        trials_meta = []
        for trial in group.trials:
            array = np.ascontiguousarray(trial, dtype=np.float64)
            flat = array.ravel()
            pieces.append(flat)
            trials_meta.append({"offset": cursor, "shape": [int(dim) for dim in array.shape]})
            cursor += int(flat.size)
        layout.append(
            {
                "subject": group.subject,
                "labels": list(group.labels),
                "dataset_id": group.dataset_id,
                "session_id": group.session_id,
                "trial_ids": list(group.trial_ids),
                "trials": trials_meta,
            }
        )
    raw = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float64)
    return raw, layout


def _pack_entries(entries: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    pieces: list[np.ndarray] = []
    layout: list[dict[str, Any]] = []
    cursor = 0

    def take(array: np.ndarray) -> dict[str, Any]:
        nonlocal cursor
        flat = np.ascontiguousarray(array, dtype=np.float64).ravel()
        pieces.append(flat)
        meta = {"offset": cursor, "shape": [int(dim) for dim in np.asarray(array).shape]}
        cursor += int(flat.size)
        return meta

    for entry in entries:
        layout.append(
            {
                "covariance": take(entry["covariance"]),
                "samples": int(entry["samples"]),
                "tdp": take(entry["tdp"]),
                "psd": take(entry["psd"]),
                "ts": take(entry["ts"]),
                "fbcsp": [
                    {
                        "low": float(low),
                        "high": float(high),
                        "covariance": take(covariance),
                        "samples": int(count),
                    }
                    for low, high, covariance, count in entry["fbcsp"]
                ],
            }
        )
    raw = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float64)
    return raw, layout


def _entry_from_layout(
    buf: np.ndarray,
    meta: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "settings": _restore_settings(settings),
        "covariance": _view(buf, meta["covariance"]),
        "samples": int(meta["samples"]),
        "tdp": _view(buf, meta["tdp"]),
        "psd": _view(buf, meta["psd"]),
        "ts": _view(buf, meta["ts"]),
        "fbcsp": [
            (float(band["low"]), float(band["high"]), _view(buf, band["covariance"]), int(band["samples"]))
            for band in meta["fbcsp"]
        ],
    }


def _plain_settings(settings: dict[str, Any]) -> dict[str, Any]:
    fbcsp = settings["fbcsp"]
    return {
        "channels": int(settings["channels"]),
        "rate": float(settings["rate"]),
        "ts_points": int(settings["ts_points"]),
        "psd_bands": [list(band) for band in settings["psd_bands"]],
        "fbcsp": [fbcsp[0], fbcsp[1], fbcsp[2], fbcsp[3], [list(band) for band in fbcsp[4]]],
    }


def _restore_settings(settings: dict[str, Any]) -> dict[str, Any]:
    low, high, width, order, bands = settings["fbcsp"]
    return {
        "channels": int(settings["channels"]),
        "rate": float(settings["rate"]),
        "ts_points": int(settings["ts_points"]),
        "psd_bands": tuple((str(name), float(start), float(stop)) for name, start, stop in settings["psd_bands"]),
        "fbcsp": (
            float(low),
            float(high),
            float(width),
            int(order),
            tuple((float(start), float(stop)) for start, stop in bands),
        ),
    }


def _groups_from_buffer(buf: np.ndarray, layout: list[dict[str, Any]]) -> list[SubjectTrials]:
    groups: list[SubjectTrials] = []
    for group in layout:
        trials = [_view(buf, meta) for meta in group["trials"]]
        groups.append(
            SubjectTrials(
                str(group["subject"]),
                trials,
                [str(label) for label in group["labels"]],
                str(group.get("dataset_id") or ""),
                str(group.get("session_id") or ""),
                tuple(str(trial_id) for trial_id in group.get("trial_ids") or ()),
            )
        )
    return groups


def _view(buf: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(dim) for dim in meta["shape"])
    count = int(np.prod(shape)) if shape else 0
    offset = int(meta["offset"])
    piece = buf[offset : offset + count].reshape(shape)
    return piece


def _allocate(values: np.ndarray) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    raw = np.ascontiguousarray(values, dtype=np.float64)
    shm = shared_memory.SharedMemory(create=True, size=max(int(raw.nbytes), 8))
    buf = np.ndarray(raw.shape, dtype=np.float64, buffer=shm.buf)
    buf[:] = raw
    return shm, buf


def _attach(name: str) -> shared_memory.SharedMemory:
    shm = shared_memory.SharedMemory(name=name, create=False)
    # The child must not unlink a block the parent still owns.
    try:
        from multiprocessing.resource_tracker import unregister

        unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
    except (ImportError, ValueError, OSError):
        pass
    return shm


def _close_shared(shm: shared_memory.SharedMemory) -> None:
    try:
        shm.close()
    except OSError:
        pass
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
