"""Covariance caches shared by the inner-fold searches."""

from __future__ import annotations

from collections.abc import Sequence
from threading import Event, Lock

import numpy as np
from threadpoolctl import threadpool_limits

from pipeline.classify.evaluate import SubjectTrials
from pipeline.classify.paper_covariance import _covariances, _window_covs


class _CovBank:
    """One covariance stack per participant, window, and band.

    The search asks for the same filtered trials on every inner fold. The
    stack is built once and then sliced.
    """

    def __init__(self, by_subject: dict[str, SubjectTrials], rate: float) -> None:
        self._people = by_subject
        self._rate = float(rate)
        self._cache: dict[
            tuple[str, tuple[float, float], tuple[float, float]],
            tuple[np.ndarray, list[str]],
        ] = {}
        self._pending: dict[
            tuple[str, tuple[float, float], tuple[float, float]], Event
        ] = {}
        self._errors: dict[
            tuple[str, tuple[float, float], tuple[float, float]], BaseException
        ] = {}
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
