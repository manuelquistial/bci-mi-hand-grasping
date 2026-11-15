"""Local session service the game calls to record or to start online playback.

The browser never opens LSL. This process does.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import numpy as np

from pipeline.alignment import align_trials, apply_alignment
from pipeline.classify import CLASSIFIERS, SCHEMES, SubjectTrials, evaluate_classifier
from pipeline.cyton import channel_labels_for, validate_cyton_stream
from pipeline.features import METHODS as FEATURE_METHODS
from pipeline.features import extract_features
from pipeline.params import ModelParams, model_params
from pipeline.emg import validate_emg_stream
from pipeline.epochs import ACTIVITY_SECONDS, is_activity_onset
from pipeline.live import LiveSignal, parse_marker_text
from pipeline.models import StreamSummary
from pipeline.preprocess import (
    PREVIEW_SECONDS,
    FilterSpec,
    apply_notch_then_bandpass,
    decimate_time,
    traces,
)
from pipeline.recorder import EmgRecorder, SessionRecorder, open_session_writer
from pipeline.session import SESSION_NAME, EmgWriter, list_recordings, load_recording
from pipeline.spatial import apply_spatial
from pipeline.windows import ActivityWindower

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
ProcessStarter = Callable[[list[str]], "subprocess.Popen[Any]"]
StreamOpener = Callable[..., Any]


def default_recordings_root() -> Path:
    return Path(__file__).resolve().parents[2] / "recordings"


def default_simulator_project() -> Path:
    env = os.environ.get("SIMULATOR_PROJECT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "simulator"


# One simulator process publishes one stream. The name is not a user setting.
SIMULATOR_LSL_NAME = "OpenBCI_EEG"
DatasetLister = Callable[[], list[dict[str, str]]]
WindowLoader = Callable[..., dict[str, Any]]
TrialLoader = Callable[..., dict[str, Any]]
SubjectLister = Callable[[str], list[dict[str, str]]]


def query_simulator_datasets(simulator_project: Path) -> list[dict[str, str]]:
    """Ask the simulator which datasets it can play."""
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(simulator_project),
            "simopenbci",
            "--list-datasets",
        ],
        cwd=simulator_project,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "simulator exited").strip()
        last = detail.splitlines()[-1] if detail else "Could not list datasets"
        raise RuntimeError(last)
    data = json.loads(completed.stdout or "[]")
    if not isinstance(data, list):
        raise RuntimeError("dataset list must be a JSON array")
    datasets: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        datasets.append({"id": str(item["id"]), "label": str(item.get("label") or item["id"])})
    return datasets


def query_simulator_subjects(simulator_project: Path, dataset: str) -> list[dict[str, str]]:
    """Ask the simulator which participants a dataset can play."""
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(simulator_project),
            "simopenbci",
            "--dataset",
            dataset,
            "--list-subjects",
            "--json",
        ],
        cwd=simulator_project,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "simulator exited").strip()
        last = detail.splitlines()[-1] if detail else "Could not list subjects"
        raise RuntimeError(last)
    data = json.loads(completed.stdout or "[]")
    if not isinstance(data, list):
        raise RuntimeError("subject list must be a JSON array")
    subjects: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        subjects.append({"id": str(item["id"]), "label": str(item.get("label") or item["id"])})
    return subjects


def query_simulator_window(
    simulator_project: Path,
    *,
    dataset: str,
    subject: str | None,
    session_dir: str | None,
    seconds: float,
) -> dict[str, Any]:
    """Read an opening window from the dataset. The simulator only reads the file."""
    if not dataset:
        raise ValueError("dataset is required")
    command = [
        "uv",
        "run",
        "--project",
        str(simulator_project),
        "simopenbci",
        "--no-dialog",
        "--dataset",
        dataset,
        "--dump-seconds",
        str(seconds),
    ]
    if session_dir:
        command.extend(["--file", session_dir])
    if subject:
        command.extend(["--subject", subject])
    completed = subprocess.run(
        command,
        cwd=simulator_project,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "simulator exited").strip()
        last = detail.splitlines()[-1] if detail else "Could not read the participant"
        raise RuntimeError(last)
    data = json.loads(completed.stdout or "{}")
    if not isinstance(data, dict) or not data.get("samples"):
        raise RuntimeError("participant window must include samples")
    return data


def query_simulator_trials(
    simulator_project: Path,
    *,
    dataset: str,
    subject: str | None,
    session_dir: str | None,
    trial_seconds: float,
) -> dict[str, Any]:
    """Read every marker window from the dataset. The caller keeps the imagery ones."""
    if not dataset:
        raise ValueError("dataset is required")
    command = [
        "uv",
        "run",
        "--project",
        str(simulator_project),
        "simopenbci",
        "--no-dialog",
        "--dataset",
        dataset,
        "--dump-trials",
        "--trial-seconds",
        str(trial_seconds),
    ]
    if session_dir:
        command.extend(["--file", session_dir])
    if subject:
        command.extend(["--subject", subject])
    completed = subprocess.run(
        command,
        cwd=simulator_project,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "simulator exited").strip()
        last = detail.splitlines()[-1] if detail else "Could not read the trials"
        raise RuntimeError(last)
    data = json.loads(completed.stdout or "{}")
    if not isinstance(data, dict) or not data.get("trials"):
        raise RuntimeError("participant trials must include trials")
    return data


def _imagery_trials(
    channels: list[str],
    *,
    dataset: str,
    subject: str | None,
    session_dir: str | None,
    sample_rate_hz: float,
    spec: FilterSpec,
        spatial: str,
    trial_loader: TrialLoader,
    params: ModelParams | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Imagery trials after the same filter and spatial step as the preview."""
    bundle = trial_loader(
        dataset=dataset,
        subject=subject,
        session_dir=session_dir,
        trial_seconds=ACTIVITY_SECONDS,
    )
    source_channels = [str(name) for name in bundle.get("channels") or []]
    rate = float(bundle.get("sample_rate_hz") or sample_rate_hz)
    prepared: list[np.ndarray] = []
    labels: list[str] = []
    for trial in bundle.get("trials") or []:
        if not isinstance(trial, dict):
            continue
        kind = str(trial.get("kind") or "")
        label = str(trial.get("label") or "")
        if not is_activity_onset(dataset, kind, label):
            continue
        array = np.asarray(trial.get("samples"), dtype=np.float64)
        filtered = apply_notch_then_bandpass(array, rate, spec)
        names = source_channels or channels
        if spatial != "none":
            filtered, names = apply_spatial(filtered, names, spatial, params)
        if names != channels:
            raise ValueError("Trial channels do not match the preview")
        prepared.append(filtered.T)
        labels.append(label)
    if not prepared:
        raise ValueError("This step needs at least one imagery trial")
    return prepared, labels


_SUPERVISED_FEATURES = {"tdp", "csp", "fbcsp"}


def _align_groups(
    groups: list[SubjectTrials],
    example: str,
) -> tuple[list[SubjectTrials], np.ndarray]:
    """Each participant is aligned to their own mean covariance."""
    aligned: list[SubjectTrials] = []
    weights: np.ndarray | None = None
    for group in groups:
        trials, matrix = align_trials(group.trials)
        if group.subject == example:
            weights = matrix
        aligned.append(
            SubjectTrials(
                group.subject,
                trials,
                group.labels,
                group.dataset_id,
                group.session_id,
                group.trial_ids,
            )
        )
    if weights is None:
        raise ValueError("This step needs at least one imagery trial")
    return aligned, weights


def _group_named(groups: list[SubjectTrials], example: str) -> SubjectTrials:
    for group in groups:
        if group.subject == example:
            return group
    raise ValueError("This step needs at least one imagery trial")


def _feature_chart(
    feature: str,
    groups: list[SubjectTrials],
    example: str,
    rate: float,
    channels: list[str],
    params: ModelParams | None = None,
) -> tuple[list[str], str, list[list[float]]]:
    """Chart one participant. TDP, CSP, and FBCSP are fit on the whole cohort."""
    shown = _group_named(groups, example)
    if feature not in _SUPERVISED_FEATURES or len(groups) == 1:
        view = extract_features(feature, shown.trials, shown.labels, rate, channels, params)
        detail = view.detail
        if len(groups) > 1:
            detail = f"{detail} · {len(groups)} participants"
        return view.rows, detail, traces(view.series)
    trials = [trial for group in groups for trial in group.trials]
    labels = [label for group in groups for label in group.labels]
    view = extract_features(feature, trials, labels, rate, channels, params)
    start = 0
    rows = view.rows
    series = view.series
    for group in groups:
        end = start + len(group.labels)
        if group.subject == example:
            rows = view.rows[start:end]
            series = view.series[:, start:end]
            break
        start = end
    rest = view.detail.split(" · ", 1)[1] if " · " in view.detail else view.detail
    detail = f"{len(rows)} trials · {rest} · {len(groups)} participants"
    return rows, detail, traces(series)


def lsl_clock() -> float:
    """LSL clock when pylsl is available, otherwise the wall clock."""
    try:
        from pylsl import local_clock

        return float(local_clock())
    except Exception:
        return time.time()


def open_marker_stream(*, eeg_name: str, wait_seconds: float) -> Any | None:
    """Markers inlet paired with the EEG stream, if one is published."""
    from pipeline.reader import LslStreamReader

    try:
        return LslStreamReader.open(
            name=f"{eeg_name}_Markers",
            wait_seconds=wait_seconds,
        )
    except TimeoutError:
        return None


def open_eeg_stream(*, stream_name: str | None, wait_seconds: float) -> Any:
    from pipeline.reader import LslStreamReader

    if stream_name:
        return LslStreamReader.open(name=stream_name, wait_seconds=wait_seconds)
    return LslStreamReader.open(stream_type="EEG", wait_seconds=wait_seconds)


def open_emg_stream(*, wait_seconds: float) -> Any:
    """Forearm EMG inlet. Eight electrodes, type EMG."""
    from pipeline.reader import LslStreamReader

    try:
        reader = LslStreamReader.open(stream_type="EMG", wait_seconds=wait_seconds)
    except TimeoutError as exc:
        raise TimeoutError(
            "No EMG LSL stream. Expect 8 forearm electrodes: "
            "4 on the left arm and 4 on the right, "
            "2 anterior and 2 posterior on each forearm."
        ) from exc
    validate_emg_stream(reader.summary)
    return reader


def start_simulator_process(command: list[str]) -> subprocess.Popen[Any]:
    cwd = None
    if "--project" in command:
        cwd = command[command.index("--project") + 1]
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process.output_lines = []  # type: ignore[attr-defined]

    def _drain() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            lines: list[str] = process.output_lines  # type: ignore[attr-defined]
            lines.append(line.rstrip())
            del lines[:-30]

    threading.Thread(target=_drain, name="sim-output", daemon=True).start()
    return process


def build_simulator_command(
    *,
    simulator_project: Path,
    lsl_name: str,
    dataset: str | None = None,
    session_dir: str | None = None,
    file: str | None = None,
    subject: str | None = None,
    data_root: str | None = None,
    run: str | None = None,
    max_samples: int | None = None,
) -> list[str]:
    """argv for ``uv run simopenbci`` against a recorded session or another dataset."""
    if dataset == "recorded_session" and not (session_dir or file):
        raise ValueError("recorded_session requires session_dir")
    command = [
        "uv",
        "run",
        "--project",
        str(simulator_project),
        "simopenbci",
        "--no-dialog",
        "--lsl-name",
        lsl_name,
    ]
    if dataset:
        command.extend(["--dataset", dataset])
    target = session_dir or file
    if target:
        command.extend(["--file", target])
    if subject:
        command.extend(["--subject", subject])
    if data_root:
        command.extend(["--data-root", data_root])
    if run:
        command.extend(["--run", run])
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    return command


class PipelineService:
    """One recording and one online run at a time."""

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
        self._dataset_lister = dataset_lister
        self._subject_lister = subject_lister
        self._subject_chooser = subject_chooser or random.choice
        self._window_loader = window_loader
        self._trial_loader = trial_loader
        self._datasets: list[dict[str, str]] | None = None
        self._subjects: dict[str, list[dict[str, str]]] = {}
        self._trial_cache: dict[tuple[str, str, str, float], dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._starting = False
        self._recorder: SessionRecorder | None = None
        self._emg_recorder: EmgRecorder | None = None
        self._emg_samples = 0
        self._session_id: str | None = None
        self._proc: subprocess.Popen[Any] | None = None
        self._online_stop: threading.Event | None = None
        self._listener: threading.Thread | None = None
        self._online_samples = 0
        self._online_error: str | None = None
        self._online_source: str | None = None
        self._online_dataset: str | None = None
        self._windower = ActivityWindower()
        self._live = LiveSignal()
        self._control_hub: Any | None = None
        self._observe_hub: Any | None = None
        self._control_bridge: Any | None = None
        self._control_registry: str | None = None
        self._control_ws_port = 8090
        self._observe_ws_port = 8091
        self._frozen_model: Any | None = None
        self._control_decoder_config: Any | None = None
        self._online_subject: str | None = None

    def list_datasets(self) -> dict[str, Any]:
        if self._datasets is None:
            lister = self._dataset_lister or (
                lambda: query_simulator_datasets(self._simulator_project)
            )
            self._datasets = lister()
        datasets: list[dict[str, str]] = [
            {
                "id": item["id"],
                "label": item["label"],
                "dataset": item["id"],
                "sessionDir": "",
            }
            for item in self._datasets
            if item["id"] != "recorded_session"
        ]
        datasets.extend(self._recording_options())
        return {"datasets": datasets}

    def list_subjects(self, dataset: str) -> dict[str, Any]:
        if not dataset or dataset == "recorded_session":
            return {"subjects": []}
        cached = self._subjects.get(dataset)
        if cached is None:
            lister = self._subject_lister or (
                lambda dataset_id: query_simulator_subjects(self._simulator_project, dataset_id)
            )
            cached = lister(dataset)
            self._subjects[dataset] = cached
        return {"subjects": cached}

    def list_online_models(self, *, loso_only: bool = False) -> dict[str, Any]:
        """List frozen packs in ``online-models/`` for the game registry picker."""
        from pipeline.online.registry import OnlineModelRegistry

        models: list[dict[str, Any]] = []
        for entry in OnlineModelRegistry().list_entries():
            offline = dict(entry.offline or {})
            held_out = bool(
                offline.get("held_out_online_subject")
                or offline.get("hold_out_online_subject")
            )
            if loso_only and not held_out:
                continue
            online_subject = str(
                offline.get("online_subject") or offline.get("subject") or ""
            )
            n_train = offline.get("n_train_subjects")
            head = entry.control_classifier or (
                entry.classifiers[0] if entry.classifiers else ""
            )
            head_short = {
                "linear_svm": "SVM",
                "logistic_regression": "LR",
            }.get(str(head), str(head) or "?")
            tag = "LOSO" if held_out else "pooled"
            subject_bit = online_subject or "-"
            train_bit = f", n_train={n_train}" if n_train is not None else ""
            models.append(
                {
                    "id": entry.id,
                    "name": entry.name,
                    "label": f"{head_short} · {tag} {subject_bit}{train_bit}",
                    "held_out": held_out,
                    "online_subject": online_subject,
                    "dataset": str(offline.get("dataset") or ""),
                    "control_classifier": str(head),
                    "n_train_subjects": n_train,
                }
            )
        return {"models": models}

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
        """Notch, bandpass, spatial, alignment, one descriptor, then a classifier.

        The client does not choose a participant. Charts use one person drawn
        from the dataset. Supervised descriptors and the classifier use every
        participant. Euclidean alignment stays on each participant's own trials.
        """
        del subject
        tuned = params or ModelParams()
        example = self._pick_subject(dataset)
        prepared = self._preview_window(
            dataset=dataset,
            subject=example,
            session_dir=session_dir,
            notch_hz=notch_hz,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            order=order,
            spatial=spatial,
            params=tuned,
        )
        rate = prepared["rate"]
        spec = prepared["spec"]
        channels = prepared["channels"]
        spatial_name = prepared["spatial_name"]
        spatial_labels = prepared["spatial_labels"]
        spatial_view = prepared["spatial_view"]
        current = prepared["current"]
        current_channels = prepared["current_channels"]
        raw_view = prepared["raw_view"]
        filtered_view = prepared["filtered_view"]
        seconds = prepared["seconds"]
        alignment_name = (alignment or "none").strip().lower()
        feature_name = (features or "none").strip().lower()
        if feature_name not in FEATURE_METHODS:
            raise ValueError(f"Unknown feature method {features!r}")
        classifier_name = (classifier or "none").strip().lower()
        evaluation_name = (evaluation or "cross_validation").strip().lower()
        if classifier_name not in CLASSIFIERS:
            raise ValueError(f"Unknown classifier {classifier!r}")
        if classifier_name != "none" and evaluation_name not in SCHEMES:
            raise ValueError(f"Unknown evaluation {evaluation!r}")
        needs_trials = (
            alignment_name == "euclidean" or feature_name != "none" or classifier_name != "none"
        )
        needs_cohort = classifier_name != "none" or feature_name in _SUPERVISED_FEATURES
        groups: list[SubjectTrials] = []
        if needs_trials:
            groups = self._cohort(
                dataset=dataset,
                session_dir=session_dir,
                sample_rate_hz=rate,
                spec=spec,
                spatial=spatial_name,
                channels=current_channels,
                only_subject=None if needs_cohort else example,
                params=tuned,
            )
            shown_name = example or "participant"
            if needs_cohort and shown_name not in {group.subject for group in groups}:
                picked = self._subject_chooser(
                    [{"id": group.subject, "label": group.subject} for group in groups]
                )
                example = None if str(picked["id"]) == "participant" else str(picked["id"])
                prepared = self._preview_window(
                    dataset=dataset,
                    subject=example,
                    session_dir=session_dir,
                    notch_hz=notch_hz,
                    band_low_hz=band_low_hz,
                    band_high_hz=band_high_hz,
                    order=order,
                    spatial=spatial,
                    params=tuned,
                )
                current = prepared["current"]
                current_channels = prepared["current_channels"]
                channels = prepared["channels"]
                spatial_labels = prepared["spatial_labels"]
                spatial_view = prepared["spatial_view"]
                raw_view = prepared["raw_view"]
                filtered_view = prepared["filtered_view"]
                seconds = prepared["seconds"]
        if alignment_name == "euclidean":
            groups, weights = _align_groups(groups, example or "participant")
            current = apply_alignment(current, weights)
            alignment_view = traces(decimate_time(current))
            shown = _group_named(groups, example or "participant")
            alignment_trials = len(shown.trials)
        elif alignment_name != "none":
            raise ValueError(f"Unknown alignment {alignment!r}")
        else:
            alignment_view = []
            alignment_trials = 0
        feature_view: list[list[float]] = []
        feature_rows: list[str] = []
        feature_detail = ""
        if feature_name != "none":
            feature_rows, feature_detail, feature_view = _feature_chart(
                feature_name,
                groups,
                example or "participant",
                rate,
                current_channels,
                tuned,
            )
        classifier_detail = ""
        classifier_folds: list[dict[str, float | str]] = []
        classifier_accuracy: float | None = None
        classifier_metrics: dict[str, float | None] = {}
        confusion_labels: list[str] = []
        confusion_matrix: list[list[int]] = []
        if classifier_name != "none":
            if feature_name == "none":
                raise ValueError("Choose a feature method before the classifier")
            scored = evaluate_classifier(
                classifier_name,
                evaluation_name,
                feature_name,
                groups,
                sample_rate_hz=rate,
                channels=current_channels,
                params=tuned,
            )
            classifier_accuracy = scored.accuracy
            classifier_detail = scored.detail
            classifier_folds = [
                {"name": item.name, "accuracy": round(item.accuracy, 4)}
                for item in scored.folds
            ]
            classifier_metrics = {
                key: None if value is None else round(float(value), 4)
                for key, value in scored.metrics.items()
            }
            confusion_labels = scored.labels
            confusion_matrix = scored.matrix
        catalog = self.list_subjects(dataset)["subjects"]
        participants = len(groups) if groups else max(len(catalog), 1)
        shown_subject = "" if example in (None, "participant") else example
        return {
            "dataset": dataset,
            "subject": shown_subject,
            "participants": participants,
            "sample_rate_hz": rate,
            "seconds": seconds,
            "channels": channels,
            "notch_hz": spec.notch_hz,
            "band_low_hz": spec.band_low_hz,
            "band_high_hz": spec.band_high_hz,
            "order": spec.order,
            "spatial": spatial_name,
            "spatial_channels": spatial_labels,
            "alignment": alignment_name,
            "alignment_trials": alignment_trials,
            "alignment_channels": current_channels if alignment_name == "euclidean" else [],
            "features": feature_name,
            "feature_detail": feature_detail,
            "feature_channels": feature_rows,
            "classifier": classifier_name,
            "evaluation": evaluation_name if classifier_name != "none" else "",
            "accuracy": None if classifier_accuracy is None else round(classifier_accuracy, 4),
            "classifier_detail": classifier_detail,
            "classifier_folds": classifier_folds,
            "metrics": classifier_metrics,
            "confusion_labels": confusion_labels,
            "confusion_matrix": confusion_matrix,
            "raw": traces(raw_view),
            "filtered": traces(filtered_view),
            "spatial_signal": spatial_view,
            "alignment_signal": alignment_view,
            "feature_signal": feature_view,
        }

    def _pick_subject(self, dataset: str) -> str | None:
        listed = list(self.list_subjects(dataset)["subjects"])
        if not listed:
            return None
        picked = self._subject_chooser(listed)
        return str(picked["id"])

    def _preview_window(
        self,
        *,
        dataset: str,
        subject: str | None,
        session_dir: str | None,
        notch_hz: float,
        band_low_hz: float,
        band_high_hz: float,
        order: int,
        spatial: str,
        params: ModelParams | None = None,
    ) -> dict[str, Any]:
        loader = self._window_loader or (
            lambda **kwargs: query_simulator_window(self._simulator_project, **kwargs)
        )
        window = loader(
            dataset=dataset,
            subject=subject,
            session_dir=session_dir,
            seconds=PREVIEW_SECONDS,
        )
        samples = np.asarray(window["samples"], dtype=np.float64)
        rate = float(window["sample_rate_hz"])
        tuned = params or ModelParams()
        spec = FilterSpec(
            notch_hz=float(notch_hz),
            notch_q=float(tuned.notch_q),
            band_low_hz=float(band_low_hz),
            band_high_hz=float(band_high_hz),
            order=int(order),
        )
        filtered = apply_notch_then_bandpass(samples, rate, spec)
        raw_view = decimate_time(samples)
        channels = [str(name) for name in window.get("channels") or []]
        while len(channels) < int(raw_view.shape[1]):
            channels.append(f"Ch{len(channels) + 1}")
        channels = channels[: int(raw_view.shape[1])]
        spatial_name = (spatial or "none").strip().lower()
        spatial_labels: list[str] = []
        spatial_view: list[list[float]] = []
        current = filtered
        current_channels = channels
        if spatial_name != "none":
            current, spatial_labels = apply_spatial(filtered, channels, spatial_name, tuned)
            current_channels = spatial_labels
            spatial_view = traces(decimate_time(current))
        return {
            "rate": rate,
            "spec": spec,
            "seconds": round(samples.shape[0] / rate, 3) if rate else 0,
            "channels": channels,
            "raw_view": raw_view,
            "filtered_view": decimate_time(filtered),
            "spatial_name": spatial_name,
            "spatial_labels": spatial_labels,
            "spatial_view": spatial_view,
            "current": current,
            "current_channels": current_channels,
        }

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
        """Filtered imagery trials for every participant.

        The pre-model request still draws one person for its charts. This load
        is the shared cohort a batch scores with every classifier.
        """
        tuned = params or ModelParams()
        listed = list(self.list_subjects(dataset)["subjects"])
        example = str(listed[0]["id"]) if listed else None
        prepared = self._preview_window(
            dataset=dataset,
            subject=example,
            session_dir=session_dir,
            notch_hz=notch_hz,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            order=order,
            spatial=spatial,
            params=tuned,
        )
        alignment_name = (alignment or "none").strip().lower()
        groups = self._cohort(
            dataset=dataset,
            session_dir=session_dir,
            sample_rate_hz=prepared["rate"],
            spec=prepared["spec"],
            spatial=prepared["spatial_name"],
            channels=prepared["current_channels"],
            only_subject=None,
            params=tuned,
            workers=workers,
        )
        if alignment_name == "euclidean":
            groups, _weights = _align_groups(groups, groups[0].subject)
        elif alignment_name != "none":
            raise ValueError(f"Unknown alignment {alignment!r}")
        return {
            "dataset": dataset,
            "groups": groups,
            "rate": prepared["rate"],
            "channels": prepared["current_channels"],
            "spatial": prepared["spatial_name"],
            "alignment": alignment_name,
            "participants": len(groups),
        }

    def _cohort(
        self,
        *,
        dataset: str,
        session_dir: str | None,
        sample_rate_hz: float,
        spec: FilterSpec,
        spatial: str,
        channels: list[str],
        only_subject: str | None,
        params: ModelParams | None = None,
        workers: int = 1,
    ) -> list[SubjectTrials]:
        """Imagery trials for every participant, or for the chart example alone."""
        listed = [str(item["id"]) for item in self.list_subjects(dataset)["subjects"]]
        if only_subject is not None and listed:
            targets: list[str | None] = [only_subject]
        elif listed:
            targets = list(listed)
        else:
            targets = [None]

        def load(subject_id: str | None) -> SubjectTrials | None:
            try:
                try:
                    trials, labels = _imagery_trials(
                        channels,
                        dataset=dataset,
                        subject=subject_id,
                        session_dir=session_dir,
                        sample_rate_hz=sample_rate_hz,
                        spec=spec,
                        spatial=spatial,
                        trial_loader=self._cached_trials,
                        params=params,
                    )
                except ValueError as exc:
                    if "imagery trial" not in str(exc):
                        raise
                    return None
                session_id = {"shuqfa2024": "02", "liu2024_stroke": "imagery"}.get(dataset, "")
                subject_name = subject_id or "participant"
                trial_ids = tuple(
                    f"{dataset}:{subject_name}:{session_id}:{index}" for index in range(len(labels))
                )
                return SubjectTrials(
                    subject_name,
                    trials,
                    labels,
                    dataset,
                    session_id,
                    trial_ids,
                )
            finally:
                # The numpy trials are copies. The raw window lists are not needed again.
                self._forget_trials(
                    dataset=dataset,
                    subject=subject_id,
                    session_dir=session_dir,
                    trial_seconds=ACTIVITY_SECONDS,
                )

        if workers <= 1 or len(targets) <= 1:
            loaded = [load(subject_id) for subject_id in targets]
        else:
            with ThreadPoolExecutor(max_workers=min(int(workers), len(targets))) as pool:
                loaded = list(pool.map(load, targets))
        groups = [group for group in loaded if group is not None]
        if not groups:
            raise ValueError("This step needs at least one imagery trial")
        return groups

    def _trial_key(self, **kwargs: Any) -> tuple[str, str, str, float]:
        return (
            str(kwargs.get("dataset") or ""),
            str(kwargs.get("subject") or ""),
            str(kwargs.get("session_dir") or ""),
            float(kwargs.get("trial_seconds") or 0),
        )

    def _cached_trials(self, **kwargs: Any) -> dict[str, Any]:
        key = self._trial_key(**kwargs)
        with self._lock:
            cached = self._trial_cache.get(key)
        if cached is not None:
            return cached
        loader = self._trial_loader or (
            lambda **load_kwargs: query_simulator_trials(self._simulator_project, **load_kwargs)
        )
        bundle = loader(**kwargs)
        with self._lock:
            self._trial_cache[key] = bundle
        return bundle

    def _forget_trials(self, **kwargs: Any) -> None:
        with self._lock:
            self._trial_cache.pop(self._trial_key(**kwargs), None)

    def _recording_options(self) -> list[dict[str, str]]:
        options: list[dict[str, str]] = []
        for path in list_recordings(self.recordings_root):
            try:
                meta = json.loads((path / SESSION_NAME).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            session_id = str(meta.get("session_id") or path.name)
            written = str(meta.get("written_at") or "")
            when = written[:16].replace("T", " ")
            label = f"{when} · {session_id}" if when else session_id
            options.append(
                {
                    "id": f"recording:{session_id}",
                    "label": label,
                    "dataset": "recorded_session",
                    "sessionDir": str(path),
                }
            )
        return options

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
        self._require_session_id(session_id)
        root = self.recordings_root / session_id
        with self._lock:
            if self._recorder is not None or self._starting:
                raise RuntimeError("A recording is already running")
            if root.exists():
                raise FileExistsError(f"Session already recorded: {session_id}")
            self._starting = True
        reader = None
        emg_reader = None
        try:
            reader = self._eeg_opener(stream_name=stream_name or None, wait_seconds=wait_seconds)
            summary: StreamSummary = reader.summary
            validate_cyton_stream(summary)
            if self._emg_opener is None:
                raise RuntimeError("EMG inlet is not configured")
            emg_reader = self._emg_opener(wait_seconds=wait_seconds)
            emg_summary: StreamSummary = emg_reader.summary
            validate_emg_stream(emg_summary)
            writer = open_session_writer(root, reader)
            emg_writer = EmgWriter(
                root,
                sample_rate_hz=float(emg_summary.nominal_srate),
                stream_name=emg_summary.name,
            )
            self._live.configure(
                summary.channel_count,
                summary.nominal_srate,
                channel_labels_for(summary),
            )
            recorder = SessionRecorder(
                reader,
                writer,
                clock=self._clock,
                on_samples=self._live.append,
            )
            emg_recorder = EmgRecorder(emg_reader, emg_writer)
            recorder.start()
            emg_recorder.start()
        except Exception:
            self._live.close()
            if emg_reader is not None:
                emg_reader.close()
            if reader is not None:
                reader.close()
            with self._lock:
                self._starting = False
            raise
        with self._lock:
            self._recorder = recorder
            self._emg_recorder = emg_recorder
            self._session_id = session_id
            self._starting = False
        return {
            "session_id": session_id,
            "path": str(root),
            "stream_name": summary.name,
            "channels": summary.channel_count,
            "sample_rate_hz": summary.nominal_srate,
            "emg_stream_name": emg_summary.name,
            "emg_channels": emg_summary.channel_count,
            "emg_sample_rate_hz": emg_summary.nominal_srate,
        }

    def add_marker(self, kind: str, code: str, label: str) -> dict[str, Any]:
        with self._lock:
            recorder = self._recorder
        if recorder is None:
            raise RuntimeError("No recording in progress")
        if not kind or not label:
            raise ValueError("marker kind and label are required")
        recorder.add_marker(kind, code or kind, label)
        return {"accepted": True}

    def stop_recording(self) -> dict[str, Any]:
        with self._lock:
            recorder = self._recorder
            emg_recorder = self._emg_recorder
            self._recorder = None
            self._emg_recorder = None
            self._session_id = None
        self._live.close()
        if recorder is None and emg_recorder is None:
            return {"stopped": False}
        emg_samples = emg_recorder.stop() if emg_recorder is not None else 0
        path = recorder.stop() if recorder is not None else None
        if path is None:
            return {"stopped": True, "emg_samples": emg_samples}
        recording = load_recording(path)
        return {
            "stopped": True,
            "path": str(path),
            "session_id": recording.session_id,
            "n_samples": recording.n_samples,
            "n_markers": len(recording.markers),
            "emg_samples": recording.emg_samples.shape[0],
        }

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
        control_ws_port: int = 8090,
        control_strategy: str = "baseline_proba",
        control_window_k: int = 3,
        control_proportional_scale: float = 1.0,
    ) -> dict[str, Any]:
        if source not in {"simulator", "cyton", "pre_model"}:
            raise ValueError("source must be 'simulator', 'cyton', or 'pre_model'")
        command: list[str] | None = None
        if source in {"simulator", "pre_model"}:
            command = build_simulator_command(
                simulator_project=self._simulator_project,
                lsl_name=lsl_name,
                dataset=dataset,
                session_dir=session_dir,
                file=file,
                subject=subject,
                data_root=data_root,
                run=run,
                max_samples=max_samples,
            )
        # A refresh or a finished playback can leave the previous run registered.
        # Start means "run this session", so the previous one is closed first.
        self.stop_online()
        frozen = None
        hub = None
        observe_hub = None
        observe_ws_port = int(control_ws_port) + 1
        if registry:
            from pipeline.online.control_bridge import ControlWsHub
            from pipeline.online.control_decoder import ControlDecoderConfig
            from pipeline.online.registry import OnlineModelRegistry

            frozen = OnlineModelRegistry().load_model(registry)
            hub = ControlWsHub(port=int(control_ws_port))
            hub.start()
            observe_hub = ControlWsHub(port=observe_ws_port)
            observe_hub.start()
            self._control_decoder_config = ControlDecoderConfig(
                strategy=control_strategy,  # type: ignore[arg-type]
                window_k=int(control_window_k),
                proportional_scale=float(control_proportional_scale),
            )
        else:
            self._control_decoder_config = None
        with self._lock:
            self._online_samples = 0
            self._emg_samples = 0
            self._online_error = None
            self._online_source = source
            self._online_dataset = dataset
            self._online_subject = subject
            self._control_registry = registry
            self._control_ws_port = int(control_ws_port)
            self._observe_ws_port = observe_ws_port
            self._frozen_model = frozen
            self._control_hub = hub
            self._observe_hub = observe_hub
            self._control_bridge = None
            self._windower.reset()
            self._online_stop = threading.Event()
        if command is not None:
            proc = self._process_starter(command)
            with self._lock:
                self._proc = proc
        stream_name = lsl_name if source in {"simulator", "pre_model"} else (lsl_name or None)
        listener = threading.Thread(
            target=self._listen,
            args=(stream_name, source == "cyton", wait_seconds),
            name="online-inlet",
            daemon=True,
        )
        with self._lock:
            self._listener = listener
        listener.start()
        return {
            "source": source,
            "lsl_name": lsl_name,
            "command": command,
            "pid": self._proc.pid if self._proc is not None else None,
            "subject": subject,
            "registry": registry,
            "control_ws_url": hub.url if hub is not None else None,
            "observe_ws_url": observe_hub.url if observe_hub is not None else None,
            "control_classifier": (
                None if frozen is None else frozen.control_name
            ),
            "control_strategy": control_strategy,
            "protocol_version": 2,
        }

    def stop_online(self) -> dict[str, Any]:
        with self._lock:
            proc = self._proc
            stop = self._online_stop
            listener = self._listener
            hub = self._control_hub
            observe_hub = self._observe_hub
            bridge = self._control_bridge
            self._proc = None
            self._listener = None
            self._online_stop = None
            self._control_hub = None
            self._observe_hub = None
            self._control_bridge = None
            self._frozen_model = None
            self._control_registry = None
        if stop is not None:
            stop.set()
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        if listener is not None:
            listener.join(timeout=2)
        if hub is not None:
            hub.stop()
        if observe_hub is not None:
            observe_hub.stop()
        self._live.close()
        with self._lock:
            samples = self._online_samples
            error = self._online_error
            source = self._online_source
            published = bridge.published_count if bridge is not None else 0
            self._online_source = None
            self._online_dataset = None
            self._windower.reset()
        if proc is None and listener is None and hub is None and observe_hub is None:
            return {"stopped": False}
        return {
            "stopped": True,
            "source": source,
            "samples_seen": samples,
            "control_samples": published,
            "error": error,
        }

    def mark_online(self, kind: str, code: str, label: str) -> dict[str, Any]:
        """Window the live EEG for one Online real trial activity.

        The session already defines the trial. This marks the activity phase
        of that trial. Online simulated does not use it: its activity times
        come from the dataset markers.
        """
        if kind not in {"imagery_start", "imagery_end"}:
            raise ValueError("marker kind must be imagery_start or imagery_end")
        if not label:
            raise ValueError("marker label is required")
        now = float(self._clock())
        with self._lock:
            if self._online_source != "cyton":
                raise RuntimeError("No online real session in progress")
            bridge = self._control_bridge
        if kind == "imagery_start":
            current = self._windower.begin(now, label, code or kind)
        else:
            current = self._windower.close_open(now)
            if current is None:
                raise RuntimeError("Motor imagery has not started")
        if bridge is not None:
            bridge.on_marker(now, kind, f"{kind}:{code or kind}:{label}", label)
        return {"accepted": True, "activity": current}

    def status(self) -> dict[str, Any]:
        with self._lock:
            recorder = self._recorder
            emg_recorder = self._emg_recorder
            session_id = self._session_id
            proc = self._proc
            samples = self._online_samples
            emg_samples = self._emg_samples
            error = self._online_error
            source = self._online_source
            activity = self._windower.latest_view()
        recording = None
        if recorder is not None and session_id is not None:
            recording = {
                "session_id": session_id,
                "samples": recorder.writer.sample_count,
                "emg_samples": emg_recorder.writer.sample_count if emg_recorder is not None else 0,
                "path": str(recorder.writer.root),
            }
        online = None
        with self._lock:
            registry = self._control_registry
            hub = self._control_hub
            observe_hub = self._observe_hub
            bridge = self._control_bridge
        if source is not None:
            online = {
                "source": source,
                "pid": proc.pid if proc is not None else None,
                "running": proc.poll() is None if proc is not None else True,
                "samples_seen": samples,
                "emg_samples_seen": emg_samples if source == "cyton" else 0,
                "error": error,
                "activity": activity,
                "registry": registry,
                "control_ws_url": None if hub is None else hub.url,
                "observe_ws_url": None if observe_hub is None else observe_hub.url,
                "control_samples": None if bridge is None else bridge.published_count,
                "last_control_sample": None if bridge is None else bridge.last_sample,
                "last_pipeline_stage": None if bridge is None else bridge.last_stage,
                "last_stream_frame": None if bridge is None else bridge.last_stream_frame,
            }
        return {"recording": recording, "online": online}

    def _ingest_markers(self, reader: Any) -> None:
        chunk = reader.pull_chunk(timeout=0.0, max_samples=64)
        for sample, timestamp in zip(chunk.samples, chunk.timestamps):
            raw = sample[0] if sample else ""
            if not isinstance(raw, str):
                raw = str(raw)
            kind, label = parse_marker_text(raw)
            self._live.add_marker(float(timestamp), kind, label)
            self._window_dataset_activity(float(timestamp), kind, raw, label)
            with self._lock:
                bridge = self._control_bridge
            if bridge is not None:
                bridge.on_marker(float(timestamp), kind, raw, label)

    def _window_dataset_activity(self, timestamp: float, kind: str, raw: str, label: str) -> None:
        with self._lock:
            source = self._online_source
            dataset = self._online_dataset or ""
        if source != "simulator" or not is_activity_onset(dataset, kind, label):
            return
        parts = (raw or "").strip().split(":")
        code = parts[1] if len(parts) >= 3 and parts[1] else kind
        self._windower.begin(timestamp, label, code)

    def _pull_emg(self, reader: Any) -> None:
        chunk = reader.pull_chunk(timeout=0.0, max_samples=2048)
        if not chunk.samples:
            return
        with self._lock:
            self._emg_samples += len(chunk.samples)

    def _listen(self, stream_name: str | None, require_cyton: bool, wait_seconds: float) -> None:
        reader = None
        marker_reader = None
        emg_reader = None
        deadline = time.monotonic() + wait_seconds
        last_error = "No LSL stream"
        try:
            while self._online_stop is not None and not self._online_stop.is_set() and reader is None:
                proc = self._proc
                if proc is not None and proc.poll() is not None:
                    tail = getattr(proc, "output_lines", [])
                    detail = tail[-1] if tail else f"simulator exited ({proc.returncode})"
                    raise RuntimeError(detail)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(last_error)
                try:
                    reader = self._eeg_opener(
                        stream_name=stream_name,
                        wait_seconds=min(1.0, remaining),
                    )
                except TimeoutError as exc:
                    last_error = str(exc)
            if reader is None:
                return
            if require_cyton:
                validate_cyton_stream(reader.summary)
                if self._emg_opener is None:
                    raise RuntimeError("EMG inlet is not configured")
                emg_reader = self._emg_opener(wait_seconds=min(1.0, wait_seconds))
                validate_emg_stream(emg_reader.summary)
            self._live.configure(
                reader.summary.channel_count,
                reader.summary.nominal_srate,
                channel_labels_for(reader.summary),
            )
            self._maybe_start_control_bridge(reader.summary)
            if self._marker_opener is not None:
                try:
                    marker_reader = self._marker_opener(
                        eeg_name=reader.summary.name,
                        wait_seconds=min(1.0, wait_seconds),
                    )
                except (TimeoutError, OSError):
                    marker_reader = None
            while self._online_stop is not None and not self._online_stop.is_set():
                if marker_reader is not None:
                    self._ingest_markers(marker_reader)
                if emg_reader is not None:
                    self._pull_emg(emg_reader)
                chunk = reader.pull_chunk(timeout=0.1, max_samples=2048)
                if chunk.samples:
                    samples = np.asarray(chunk.samples, dtype=np.float64)
                    self._live.append(samples, chunk.timestamps)
                    self._windower.push(samples, chunk.timestamps)
                    with self._lock:
                        bridge = self._control_bridge
                        self._online_samples += len(chunk.samples)
                    if bridge is not None:
                        bridge.push_eeg(samples, chunk.timestamps)
                    continue
                proc = self._proc
                if proc is not None and proc.poll() is not None:
                    break
        except Exception as exc:  # noqa: BLE001 — reported on /status
            with self._lock:
                self._online_error = str(exc)
        finally:
            if marker_reader is not None:
                marker_reader.close()
            if emg_reader is not None:
                emg_reader.close()
            if reader is not None:
                reader.close()
            self._finish_listener()

    def _maybe_start_control_bridge(self, summary: StreamSummary) -> None:
        with self._lock:
            frozen = self._frozen_model
            hub = self._control_hub
            dataset = self._online_dataset or "shuqfa2024"
            subject = self._online_subject or ""
            decoder_config = self._control_decoder_config
            if frozen is None or hub is None or self._control_bridge is not None:
                return
        from pipeline.online.control_bridge import LiveControlBridge

        with self._lock:
            observe_hub = self._observe_hub
        log_dir = self.recordings_root.parent / "batch-results"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(log_dir / f"control-log-{int(time.time())}.jsonl")
        channel_labels = [str(name) for name in (summary.channel_labels or ())]
        bridge = LiveControlBridge(
            frozen,
            dataset=dataset,
            hub=hub,
            sample_rate_hz=float(summary.nominal_srate or frozen.sample_rate_hz),
            channel_count=int(summary.channel_count),
            decoder_config=decoder_config,
            subject_id=str(subject or ""),
            log_path=log_path,
            observe_hub=observe_hub,
            channel_labels=channel_labels or None,
        )
        with self._lock:
            self._control_bridge = bridge

    def _finish_listener(self) -> None:
        with self._lock:
            if self._listener is not threading.current_thread():
                return
            self._listener = None
            self._proc = None
            self._online_stop = None
            if self._online_error is None:
                self._online_source = None

    @staticmethod
    def _require_session_id(session_id: str) -> None:
        if not SESSION_ID_RE.fullmatch(session_id or ""):
            raise ValueError(
                "session_id must contain only letters, digits, '_' and '-'"
            )


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


def make_handler(service: PipelineService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def handle(self) -> None:
            try:
                super().handle()
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                # Browser / Next aborted a poll or closed keep-alive early.
                return

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[pipeline] {self.address_string()} {fmt % args}")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send(204, {})

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/health":
                self._send(200, {"ok": True})
                return
            if path == "/v1/status":
                self._send(200, service.status())
                return
            if path == "/v1/signal":
                self._send(200, service.signal())
                return
            if path == "/v1/datasets":
                try:
                    self._send(200, service.list_datasets())
                except (RuntimeError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
                    self._send(400, {"error": str(exc)})
                return
            if path == "/v1/subjects":
                dataset = (parse_qs(urlparse(self.path).query).get("dataset") or [""])[0]
                try:
                    self._send(200, service.list_subjects(dataset))
                except (RuntimeError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
                    self._send(400, {"error": str(exc)})
                return
            if path == "/v1/online/models":
                query = parse_qs(urlparse(self.path).query)
                loso_raw = (query.get("loso") or ["0"])[0].lower()
                loso_only = loso_raw in {"1", "true", "yes"}
                try:
                    self._send(200, service.list_online_models(loso_only=loso_only))
                except (RuntimeError, OSError, json.JSONDecodeError) as exc:
                    self._send(400, {"error": str(exc)})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                body = _read_json(self)
                if path == "/v1/record/start":
                    payload = service.start_recording(
                        str(body.get("session_id") or ""),
                        stream_name=_blank_to_none(body.get("stream_name")),
                        wait_seconds=float(body.get("wait_seconds") or 5),
                    )
                elif path == "/v1/record/marker":
                    payload = service.add_marker(
                        str(body.get("kind") or ""),
                        str(body.get("code") or ""),
                        str(body.get("label") or ""),
                    )
                elif path == "/v1/record/stop":
                    payload = service.stop_recording()
                elif path == "/v1/online/start":
                    payload = service.start_online(
                        source=str(body.get("source") or "simulator"),
                        lsl_name=str(body.get("lsl_name") or "OpenBCI_EEG"),
                        dataset=_blank_to_none(body.get("dataset")),
                        session_dir=_blank_to_none(body.get("session_dir")),
                        file=_blank_to_none(body.get("file")),
                        subject=_blank_to_none(body.get("subject")),
                        data_root=_blank_to_none(body.get("data_root")),
                        run=_blank_to_none(body.get("run")),
                        max_samples=_optional_int(body.get("max_samples")),
                        wait_seconds=float(body.get("wait_seconds") or 5),
                        registry=_blank_to_none(body.get("registry")),
                        control_ws_port=int(body.get("control_ws_port") or 8090),
                        control_strategy=str(
                            body.get("control_strategy") or "baseline_proba"
                        ),
                        control_window_k=int(body.get("control_window_k") or 3),
                        control_proportional_scale=float(
                            body.get("control_proportional_scale") or 1.0
                        ),
                    )
                elif path == "/v1/online/marker":
                    payload = service.mark_online(
                        str(body.get("kind") or ""),
                        str(body.get("code") or ""),
                        str(body.get("label") or ""),
                    )
                elif path == "/v1/online/stop":
                    payload = service.stop_online()
                elif path == "/v1/pre-model/start":
                    payload = service.start_online(
                        source="pre_model",
                        lsl_name="OpenBCI_EEG",
                        dataset=_blank_to_none(body.get("dataset")),
                        session_dir=_blank_to_none(body.get("session_dir")),
                        subject=_blank_to_none(body.get("subject")),
                        wait_seconds=float(body.get("wait_seconds") or 5),
                    )
                elif path == "/v1/pre-model/stop":
                    payload = service.stop_online()
                elif path == "/v1/pre-model/filter":
                    payload = service.preview_filter(
                        dataset=str(body.get("dataset") or ""),
                        subject=_blank_to_none(body.get("subject")),
                        session_dir=_blank_to_none(body.get("session_dir")),
                        notch_hz=_number(body.get("notch_hz"), 50),
                        band_low_hz=_number(body.get("band_low_hz"), 8),
                        band_high_hz=_number(body.get("band_high_hz"), 30),
                        order=int(_number(body.get("order"), 4)),
                        spatial=str(body.get("spatial") or "none"),
                        alignment=str(body.get("alignment") or "none"),
                        features=str(body.get("features") or "none"),
                        classifier=str(body.get("classifier") or "none"),
                        evaluation=str(body.get("evaluation") or "cross_validation"),
                        params=model_params(body.get("hyperparameters")),
                    )
                else:
                    self._send(404, {"error": "not found"})
                    return
            except (
                ValueError,
                FileExistsError,
                RuntimeError,
                TimeoutError,
                OSError,
                subprocess.TimeoutExpired,
            ) as exc:
                self._send(400, {"error": str(exc)})
                return
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid JSON"})
                return
            self._send(200, payload)

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()
                self.wfile.write(body)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                return

    return Handler


def _blank_to_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _number(value: object, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """Do not dump a traceback when the game closes a keep-alive socket."""

    def handle_error(self, request: object, client_address: object) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    recordings_root: Path | None = None,
) -> None:
    """Block and serve the session API."""
    root = Path(recordings_root) if recordings_root is not None else default_recordings_root()
    service = PipelineService(root)
    server = _QuietThreadingHTTPServer((host, port), make_handler(service))
    print(f"pipeline serve http://{host}:{port}  recordings={root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        service.stop_recording()
        service.stop_online()
        server.server_close()
