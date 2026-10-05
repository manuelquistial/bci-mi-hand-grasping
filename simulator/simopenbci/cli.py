"""CLI entrypoint for SimOpenBCI."""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from dataset_loaders import (
    DATASET_IDS,
    BaseDatasetLoader,
    create_loader,
    get_loader_class_by_id,
    list_datasets,
    list_subjects_for_dataset,
    resolve_data_root,
)
from dataset_loaders.events import resolve_events

from .clock import RealtimeClock
from .config import PlaybackConfig
from .gui import QtFilePicker
from .notifiers.console import ConsoleNotifier
from .notifiers.qt import QtNotifier
from .playback import PlaybackService
from .ports import Notifier, SampleSink
from .sinks.lsl import LslSampleSink
from .sinks.markers import LslMarkerSink

SinkFactory = Callable[[], SampleSink]

# Default dataset when starting without --file / --pick-file
DEFAULT_DATASET = "bci_iv2a"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SimOpenBCI: stream EEG data from a dataset to LSL. "
            f"With no args, streams the first subject of {DEFAULT_DATASET} "
            "(data/<dataset>/)."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_IDS),
        default=None,
        help=f"Dataset type (default: {DEFAULT_DATASET} when not using --file/--pick-file)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="EEG file path (optional; prefer --dataset/--subject)",
    )
    parser.add_argument(
        "--pick-file",
        action="store_true",
        help="Open a file dialog to choose an EEG file (instead of using a dataset subject)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after sending this many samples (for testing; default: send all)",
    )
    parser.add_argument(
        "--no-dialog",
        action="store_true",
        help="Do not show GUI finish/error dialogs (console only)",
    )
    parser.add_argument(
        "--lsl-name",
        type=str,
        default=None,
        help="LSL stream name (default: from dataset). Use e.g. OpenBCI_EEG or BioSemi to connect to articulo_1.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        help="Use only first N channels (e.g. 8 for articulo_1).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root folder of the dataset (for --subject/--list-subjects).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Subject ID (e.g. A01, S001). Use with --dataset and optional --run. Overrides --file.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run/session (e.g. T, E). Only with --subject.",
    )
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="List available subjects and runs for the dataset, then exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --list-subjects, print subjects as JSON.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print the datasets this simulator can play as JSON, then exit.",
    )
    parser.add_argument(
        "--events",
        type=str,
        default=None,
        help=(
            "Path to trial/events CSV (sample_index,code,label[,kind]). "
            "Default: sidecar <eeg_file>.events.csv next to the EEG file."
        ),
    )
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="Do not open an LSL Markers stream even if an events file is present.",
    )
    return parser


def _build_loader_overrides(
    channels: int | None,
    lsl_name: str | None,
) -> dict:
    overrides: dict = {}
    if channels is not None:
        overrides["n_channels"] = channels
    if lsl_name is not None:
        overrides["stream_name"] = lsl_name
    return overrides


def _resolve_file_from_subject(
    dataset_id: str,
    data_root: Path,
    subject: str,
    run: str | None,
) -> Path:
    cls = get_loader_class_by_id(dataset_id)
    if cls is None:
        raise SystemExit(f"Error: unknown dataset {dataset_id!r}")
    resolved = cls.resolve_subject_file(data_root, subject, run)
    if resolved is None:
        raise SystemExit(
            f"Error: could not resolve subject={subject!r} run={run!r} in {data_root}"
        )
    if not resolved.exists():
        raise SystemExit(f"Error: file not found {resolved}")
    run_label = run
    if run_label is None:
        for info in cls.list_subjects(data_root):
            if info.subject_id == subject and info.runs:
                run_label = info.runs[0]
                break
        run_label = run_label or "?"
    print(f"Using subject {subject} run {run_label}: {resolved.name}")
    return resolved


def _first_subject_file(dataset_id: str, data_root: Path) -> Path | None:
    """Resolve the first available subject file for a dataset (loader picks default run)."""
    subjects = list_subjects_for_dataset(dataset_id, data_root)
    if not subjects:
        return None
    first = subjects[0]
    cls = get_loader_class_by_id(dataset_id)
    if cls is None:
        return None
    # Pass run=None so loaders apply their own default (e.g. BCI prefers T).
    return cls.resolve_subject_file(data_root, first.subject_id, None)


def run(
    argv: Sequence[str] | None = None,
    *,
    sink_factory: SinkFactory | None = None,
) -> int:
    """
    Parse argv and run the simulator.
    sink_factory: injectable for tests (defaults to LslSampleSink).
    Returns process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.list_datasets:
        print(json.dumps(list_datasets()))
        return 0

    # Prefer a dataset unless the user explicitly passed a file or pick-file
    dataset_id = args.dataset
    if dataset_id is None and not args.file and not args.pick_file:
        dataset_id = DEFAULT_DATASET

    data_root = resolve_data_root(dataset_id, args.data_root)

    if args.list_subjects:
        if not dataset_id:
            print("Error: --list-subjects requires --dataset (e.g. bci_iv2a)")
            return 1
        root = data_root if data_root is not None else Path(args.data_root or ".")
        subjects = list_subjects_for_dataset(dataset_id, root)
        if args.json:
            print(
                json.dumps(
                    [
                        {"id": s.subject_id, "label": s.subject_id, "runs": s.runs}
                        for s in subjects
                    ]
                )
            )
            return 0
        if not subjects:
            print(
                f"No subjects found in {root} (or dataset does not support multi-subject)."
            )
        else:
            print(f"Dataset: {dataset_id}  data_root: {root}")
            print("Subject ID | Runs")
            print("-----------|-----")
            for s in subjects:
                print(f"  {s.subject_id:10} | {', '.join(s.runs)}")
        return 0

    file_path = args.file
    if args.subject:
        if not dataset_id:
            print("Error: --subject requires --dataset")
            return 1
        if data_root is None:
            print(
                "Error: --subject requires a data root "
                "(use --data-root or a dataset with default_data_root)"
            )
            return 1
        file_path = str(
            _resolve_file_from_subject(dataset_id, data_root, args.subject, args.run)
        )
    elif not file_path and not args.pick_file and dataset_id and data_root is not None:
        # Default start: first subject under data/<dataset>/.
        resolved = _first_subject_file(dataset_id, data_root)
        if resolved is not None:
            subjects = list_subjects_for_dataset(dataset_id, data_root)
            sid = subjects[0].subject_id
            print(f"Dataset {dataset_id} @ {data_root}: subject {sid} ({resolved.name})")
            file_path = str(resolved)

    no_dialog = bool(args.no_dialog)
    notifier: Notifier
    if no_dialog:
        notifier = ConsoleNotifier()
    else:
        notifier = QtNotifier()

    if not file_path:
        if args.pick_file and not no_dialog:
            file_path = QtFilePicker().pick_file()
            if not file_path:
                notifier.warning("No file selected")
                return 1
        else:
            hint = (
                f"No dataset files found for {dataset_id!r} in {data_root}. "
                "Put files in data/<dataset>/, or pass --subject/--file, "
                "or use --pick-file."
            )
            notifier.error(hint)
            return 1

    overrides = _build_loader_overrides(args.channels, args.lsl_name)
    loader: BaseDatasetLoader
    try:
        if dataset_id:
            loader = create_loader(dataset_id=dataset_id, **overrides)
        else:
            loader = create_loader(file_path=file_path, **overrides)
    except ValueError as exc:
        notifier.error(str(exc))
        return 1

    bind = getattr(loader, "bind", None)
    if callable(bind):
        try:
            bind(file_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            notifier.error(str(exc))
            return 1

    factory = sink_factory or (lambda: LslSampleSink())
    sink = factory()
    use_lsl_timing = sink_factory is None
    config = PlaybackConfig(
        max_samples=args.max_samples,
        stream_name=args.lsl_name,
        notify_finish=not no_dialog,
        # LSL discovery needs the outlet to stay up for ~1s+ on the network.
        advertise_seconds=1.5 if use_lsl_timing else 0.0,
        hold_seconds=0.5 if use_lsl_timing else 0.0,
    )

    events_from_loader = getattr(loader, "load_events", None)
    if args.events:
        events = resolve_events(file_path, args.events)
        events_origin = args.events
    elif callable(events_from_loader):
        events = events_from_loader(file_path)
        events_origin = "session markers"
    else:
        events = resolve_events(file_path, None)
        events_origin = Path(file_path).with_suffix(".events.csv").name
    marker_sink = None
    if events and not args.no_markers and use_lsl_timing:
        marker_sink = LslMarkerSink()

    if events:
        notifier.progress(f"Events: {events_origin} ({len(events)} markers)")
    elif args.events:
        notifier.warning(f"No events loaded from {args.events}")

    service = PlaybackService()
    try:
        service.run(
            loader=loader,
            file_path=file_path,
            sink=sink,
            clock=RealtimeClock(),
            notifier=notifier,
            config=config,
            events=events,
            marker_sink=marker_sink,
        )
    except Exception:
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))
