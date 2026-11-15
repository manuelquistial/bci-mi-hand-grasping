"""Resolve which dataset, subject and file the CLI should play."""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

from dataset_loaders import (
    BaseDatasetLoader,
    create_loader,
    get_loader_class_by_id,
    list_subjects_for_dataset,
)

from ..gui import QtFilePicker
from ..ports import Notifier


def build_loader_overrides(channels: int | None, lsl_name: str | None) -> dict:
    overrides: dict = {}
    if channels is not None:
        overrides["n_channels"] = channels
    if lsl_name is not None:
        overrides["stream_name"] = lsl_name
    return overrides


def resolve_file_from_subject(
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
    print(f"Using subject {subject} run {run_label}: {resolved.name}", file=sys.stderr)
    return resolved


def first_subject_file(dataset_id: str, data_root: Path) -> Path | None:
    """Resolve the first available subject file (the loader picks the default run)."""
    subjects = list_subjects_for_dataset(dataset_id, data_root)
    if not subjects:
        return None
    cls = get_loader_class_by_id(dataset_id)
    if cls is None:
        return None
    # run=None lets loaders apply their own default (e.g. BCI prefers T).
    return cls.resolve_subject_file(data_root, subjects[0].subject_id, None)


def list_subjects_command(args: Namespace, dataset_id: str | None, data_root: Path | None) -> int:
    """Handle ``--list-subjects`` (text or ``--json``)."""
    if not dataset_id:
        print("Error: --list-subjects requires --dataset (e.g. shuqfa2024)")
        return 1
    root = data_root if data_root is not None else Path(args.data_root or ".")
    subjects = list_subjects_for_dataset(dataset_id, root)
    if args.json:
        print(
            json.dumps(
                [{"id": s.subject_id, "label": s.subject_id, "runs": s.runs} for s in subjects]
            )
        )
        return 0
    if not subjects:
        print(f"No subjects found in {root} (or dataset does not support multi-subject).")
        return 0
    print(f"Dataset: {dataset_id}  data_root: {root}")
    print("Subject ID | Runs")
    print("-----------|-----")
    for s in subjects:
        print(f"  {s.subject_id:10} | {', '.join(s.runs)}")
    return 0


def resolve_file_path(
    args: Namespace,
    dataset_id: str | None,
    data_root: Path | None,
    notifier: Notifier,
) -> str | None:
    """Return the EEG file to play, or None after reporting why via notifier/stdout."""
    file_path = args.file
    if args.subject:
        if not dataset_id:
            print("Error: --subject requires --dataset")
            return None
        if data_root is None:
            print(
                "Error: --subject requires a data root "
                "(use --data-root or a dataset with default_data_root)"
            )
            return None
        return str(resolve_file_from_subject(dataset_id, data_root, args.subject, args.run))
    if not file_path and not args.pick_file and dataset_id and data_root is not None:
        # Default start: first subject under data/<dataset>/.
        resolved = first_subject_file(dataset_id, data_root)
        if resolved is not None:
            subjects = list_subjects_for_dataset(dataset_id, data_root)
            if args.dump_seconds is None:
                print(
                    f"Dataset {dataset_id} @ {data_root}: "
                    f"subject {subjects[0].subject_id} ({resolved.name})"
                )
            file_path = str(resolved)
    if file_path:
        return file_path
    if args.pick_file and not args.no_dialog:
        picked = QtFilePicker().pick_file()
        if not picked:
            notifier.warning("No file selected")
        return picked or None
    notifier.error(
        f"No dataset files found for {dataset_id!r} in {data_root}. "
        "Put files in data/<dataset>/, or pass --subject/--file, "
        "or use --pick-file."
    )
    return None


def build_loader(
    args: Namespace,
    dataset_id: str | None,
    file_path: str,
    notifier: Notifier,
) -> BaseDatasetLoader | None:
    """Create the loader and bind it to the file; report errors via notifier."""
    overrides = build_loader_overrides(args.channels, args.lsl_name)
    try:
        if dataset_id:
            loader = create_loader(dataset_id=dataset_id, **overrides)
        else:
            loader = create_loader(file_path=file_path, **overrides)
    except ValueError as exc:
        notifier.error(str(exc))
        return None
    bind = getattr(loader, "bind", None)
    if callable(bind):
        try:
            bind(file_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            notifier.error(str(exc))
            return None
    return loader
