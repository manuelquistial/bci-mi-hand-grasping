"""Local session service the game calls to record or to start online playback.

The browser never opens LSL. This process does.
"""

from __future__ import annotations

import random
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

from pipeline.adapters.http.handler import make_handler
from pipeline.adapters.http.server import serve
from pipeline.live import LiveSignal
from pipeline.params import ModelParams
from pipeline.adapters.lsl.streams import (
    lsl_clock, open_eeg_stream, open_emg_stream, open_marker_stream,
)
from pipeline.adapters.simulator.catalog import (
    query_simulator_datasets,
    query_simulator_subjects,
    query_simulator_trials,
    query_simulator_window,
)
from pipeline.adapters.simulator.process import (
    build_simulator_command, start_simulator_process,
)
from pipeline.application.cohort_service import CohortService
from pipeline.application.dataset_catalog import DatasetCatalog
from pipeline.application.filter_preview import FilterPreviewService
from pipeline.application.model_catalog import list_online_models
from pipeline.application.online_session import OnlineRunManager, OnlineStartRequest
from pipeline.application.preview_window import PreviewWindows
from pipeline.application.recording_service import RecordingService
from pipeline.config import (
    CONTROL_WS_PORT,
    OBSERVE_WS_PORT,
    SIMULATOR_LSL_NAME,
    default_recordings_root,
    default_simulator_project,
)
from pipeline.windows import ActivityWindower

ProcessStarter = Callable[[list[str]], "subprocess.Popen[Any]"]
StreamOpener = Callable[..., Any]

# Re-exports kept for CLI and tests during the WP1 migration.
__all__ = [
    "CONTROL_WS_PORT",
    "OBSERVE_WS_PORT",
    "PipelineService",
    "SIMULATOR_LSL_NAME",
    "build_simulator_command",
    "default_recordings_root",
    "default_simulator_project",
    "lsl_clock",
    "make_handler",
    "open_eeg_stream",
    "open_emg_stream",
    "open_marker_stream",
    "query_simulator_datasets",
    "query_simulator_subjects",
    "query_simulator_trials",
    "query_simulator_window",
    "serve",
    "start_simulator_process",
]

DatasetLister = Callable[[], list[dict[str, str]]]
WindowLoader = Callable[..., dict[str, Any]]
TrialLoader = Callable[..., dict[str, Any]]
SubjectLister = Callable[[str], list[dict[str, str]]]


class PipelineService:
    """One recording and one online run at a time. Delegates to application modules."""

    def __init__(
        self,
        recordings_root: Path,
        *,
        eeg_opener: StreamOpener | None = None,
        emg_opener: StreamOpener | None = None,
        marker_opener: Callable[..., Any] | None = None,
        clock: Callable[[], float] | None = None,
        simulator_project: Path | None = None,
        process_starter: ProcessStarter | None = None,
        dataset_lister: DatasetLister | None = None,
        subject_lister: SubjectLister | None = None,
        subject_chooser: Callable[[list[dict[str, str]]], dict[str, str]] | None = None,
        window_loader: WindowLoader | None = None,
        trial_loader: TrialLoader | None = None,
    ) -> None:
        self.recordings_root = Path(recordings_root)
        self.recordings_root.mkdir(parents=True, exist_ok=True)
        self._eeg_opener = eeg_opener or open_eeg_stream
        self._emg_opener = open_emg_stream if emg_opener is None and eeg_opener is None else emg_opener
        self._marker_opener = open_marker_stream if marker_opener is None and eeg_opener is None else marker_opener
        self._clock = clock or lsl_clock
        self._simulator_project = Path(simulator_project or default_simulator_project())
        self._process_starter = process_starter or start_simulator_process
        self._subject_chooser = subject_chooser or random.choice
        self._lock = threading.Lock()
        self._live = LiveSignal()
        self._recording = RecordingService(
            lock=self._lock,
            live=self._live,
            recordings_root=self.recordings_root,
            eeg_opener=self._eeg_opener,
            emg_opener=self._emg_opener,
            clock=self._clock,
        )
        self._catalog = DatasetCatalog(
            simulator_project=self._simulator_project,
            recording_options=self._recording.recording_options,
            dataset_lister=dataset_lister,
            subject_lister=subject_lister,
        )
        windows = PreviewWindows(
            simulator_project=self._simulator_project,
            window_loader=window_loader,
        )
        self._cohort = CohortService(
            lock=self._lock,
            simulator_project=self._simulator_project,
            trial_loader=trial_loader,
            subjects_for=self._catalog.list_subjects,
            prepare_window=windows.prepare,
        )
        self._preview = FilterPreviewService(
            catalog=self._catalog,
            windows=windows,
            cohort=self._cohort,
            subject_chooser=self._subject_chooser,
        )
        self._online = OnlineRunManager(
            lock=self._lock,
            live=self._live,
            windower=ActivityWindower(),
            recordings_root=self.recordings_root,
            simulator_project=self._simulator_project,
            eeg_opener=self._eeg_opener,
            emg_opener=self._emg_opener,
            marker_opener=self._marker_opener,
            process_starter=self._process_starter,
            clock=self._clock,
        )

    def list_datasets(self) -> dict[str, Any]:
        return self._catalog.list_datasets()

    def list_subjects(self, dataset: str) -> dict[str, Any]:
        return self._catalog.list_subjects(dataset)

    def list_online_models(self, *, loso_only: bool = False) -> dict[str, Any]:
        """List frozen packs in ``online-models/`` for the game registry picker."""
        return list_online_models(loso_only=loso_only)

    def preview_filter(
        self,
        *,
        dataset: str,
        subject: str | None,
        session_dir: str | None,
        notch_hz: float,
        band_low_hz: float,
        band_high_hz: float,
        order: int,
        spatial: str = "none",
        alignment: str = "none",
        features: str = "none",
        classifier: str = "none",
        evaluation: str = "cross_validation",
        params: ModelParams | None = None,
    ) -> dict[str, Any]:
        """Notch, bandpass, spatial, alignment, one descriptor, then a classifier."""
        return self._preview.preview_filter(
            dataset=dataset,
            subject=subject,
            session_dir=session_dir,
            notch_hz=notch_hz,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            order=order,
            spatial=spatial,
            alignment=alignment,
            features=features,
            classifier=classifier,
            evaluation=evaluation,
            params=params,
        )

    def load_cohort(
        self,
        *,
        dataset: str,
        session_dir: str | None = None,
        notch_hz: float = 50.0,
        band_low_hz: float = 8.0,
        band_high_hz: float = 30.0,
        order: int = 4,
        spatial: str = "none",
        alignment: str = "none",
        params: ModelParams | None = None,
        workers: int = 1,
    ) -> dict[str, Any]:
        """Filtered imagery trials for every participant."""
        return self._cohort.load_cohort(
            dataset=dataset,
            session_dir=session_dir,
            notch_hz=notch_hz,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            order=order,
            spatial=spatial,
            alignment=alignment,
            params=params,
            workers=workers,
        )

    def signal(self) -> dict[str, Any]:
        """Latest EEG window for the game session background."""
        return self._live.snapshot()

    def start_recording(
        self,
        session_id: str,
        *,
        stream_name: str | None = None,
        wait_seconds: float = 5.0,
    ) -> dict[str, Any]:
        return self._recording.start(
            session_id, stream_name=stream_name, wait_seconds=wait_seconds
        )

    def add_marker(self, kind: str, code: str, label: str) -> dict[str, Any]:
        return self._recording.add_marker(kind, code, label)

    def stop_recording(self) -> dict[str, Any]:
        return self._recording.stop()

    def start_online(
        self,
        *,
        source: str,
        lsl_name: str,
        dataset: str | None = None,
        session_dir: str | None = None,
        file: str | None = None,
        subject: str | None = None,
        data_root: str | None = None,
        run: str | None = None,
        max_samples: int | None = None,
        wait_seconds: float = 5.0,
        registry: str | None = None,
        control_ws_port: int = CONTROL_WS_PORT,
        observe_ws_port: int | None = None,
        control_strategy: str = "baseline_proba",
        control_window_k: int = 3,
        control_proportional_scale: float = 1.0,
    ) -> dict[str, Any]:
        return self._online.start(
            OnlineStartRequest(
                source=source,
                lsl_name=lsl_name,
                dataset=dataset,
                session_dir=session_dir,
                file=file,
                subject=subject,
                data_root=data_root,
                run=run,
                max_samples=max_samples,
                wait_seconds=wait_seconds,
                registry=registry,
                control_ws_port=control_ws_port,
                observe_ws_port=observe_ws_port,
                control_strategy=control_strategy,
                control_window_k=control_window_k,
                control_proportional_scale=control_proportional_scale,
            )
        )

    def stop_online(self) -> dict[str, Any]:
        return self._online.stop()

    def mark_online(self, kind: str, code: str, label: str) -> dict[str, Any]:
        """Window the live EEG for one Online real trial activity."""
        return self._online.mark(kind, code, label)

    def status(self) -> dict[str, Any]:
        return {
            "recording": self._recording.status_slice(),
            "online": self._online.status_slice(),
        }
