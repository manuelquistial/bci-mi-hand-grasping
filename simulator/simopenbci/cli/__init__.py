"""CLI entrypoint for SimOpenBCI (thin wiring; logic lives in submodules)."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from dataset_loaders import list_datasets, resolve_data_root

from ..notifiers.console import ConsoleNotifier
from ..notifiers.qt import QtNotifier
from ..ports import Notifier
from .dump import dump_seconds, dump_trials
from .parser import DEFAULT_DATASET, build_parser
from .resolve_source import build_loader, list_subjects_command, resolve_file_path
from .run_playback import SinkFactory, run_playback

__all__ = ["DEFAULT_DATASET", "build_parser", "main", "run"]


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
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.list_datasets:
        print(json.dumps(list_datasets()))
        return 0

    # Prefer a dataset unless the user explicitly passed a file or pick-file.
    dataset_id = args.dataset
    if dataset_id is None and not args.file and not args.pick_file:
        dataset_id = DEFAULT_DATASET
    data_root: Path | None = resolve_data_root(dataset_id, args.data_root)

    if args.list_subjects:
        return list_subjects_command(args, dataset_id, data_root)

    notifier: Notifier = ConsoleNotifier() if args.no_dialog else QtNotifier()
    file_path = resolve_file_path(args, dataset_id, data_root, notifier)
    if file_path is None:
        return 1
    loader = build_loader(args, dataset_id, file_path, notifier)
    if loader is None:
        return 1

    if args.dump_trials:
        return dump_trials(loader, file_path, float(args.trial_seconds))
    if args.dump_seconds is not None:
        return dump_seconds(loader, file_path, float(args.dump_seconds))
    return run_playback(args, loader, file_path, notifier, sink_factory)


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))
