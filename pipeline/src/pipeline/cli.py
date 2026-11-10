"""CLI for discovering, reading, and visualizing Lab Streaming Layer (LSL) streams."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pipeline.spatial import METHODS as SPATIAL_METHODS


def _add_stream_select_args(parser: argparse.ArgumentParser, *, default_type: str | None) -> None:
    parser.add_argument(
        "--type",
        dest="stream_type",
        default=default_type,
        help=f"LSL type filter{'' if default_type is None else f' (default: {default_type})'}",
    )
    parser.add_argument("--name", default=None, help="Exact stream name (optional)")
    parser.add_argument(
        "--wait",
        type=float,
        default=5.0 if default_type else 2.0,
        help="Discovery wait time in seconds",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Discover, read, and visualize Lab Streaming Layer (LSL) streams.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="List available LSL streams")
    _add_stream_select_args(list_p, default_type=None)
    list_p.set_defaults(wait=2.0)

    read_p = sub.add_parser("read", help="Connect to a stream and print samples")
    _add_stream_select_args(read_p, default_type="EEG")
    read_p.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Pull timeout per chunk in seconds (default: 1)",
    )
    read_p.add_argument(
        "--max-samples",
        type=int,
        default=32,
        help="Max samples per printed chunk (default: 32)",
    )
    read_p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Stop after N chunks (default: run until Ctrl+C)",
    )
    read_p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after N seconds of reading (optional)",
    )

    view_p = sub.add_parser("view", help="Live multi-channel LSL waveform viewer")
    _add_stream_select_args(view_p, default_type="EEG")
    view_p.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Visible time window in seconds (default: 5)",
    )
    view_p.add_argument(
        "--refresh",
        type=float,
        default=30.0,
        help="UI refresh rate in Hz (default: 30)",
    )
    view_p.add_argument(
        "--channels",
        default=None,
        help="Channel indices to plot, e.g. 0,1,2 or 0-7 (default: all)",
    )
    view_p.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Fixed vertical spacing between channels (default: auto)",
    )
    view_p.add_argument(
        "--markers-name",
        default=None,
        help="LSL Markers stream name (default: <eeg_name>_Markers)",
    )
    view_p.add_argument(
        "--no-markers",
        action="store_true",
        help="Do not look for a Markers stream",
    )

    show_p = sub.add_parser("show", help="Summarize a recorded EEG session directory")
    show_p.add_argument("path", help="Session directory written by pipeline record/serve")

    serve_p = sub.add_parser("serve", help="Session API for the game (record and online)")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8765)
    serve_p.add_argument(
        "--recordings",
        default=None,
        help="Directory for session folders (default: pipeline/recordings)",
    )

    batch_p = sub.add_parser(
        "batch",
        help="Score every classifier on one shared cohort per dataset",
    )
    batch_p.add_argument(
        "--datasets",
        default="shuqfa2024,liu2024_stroke",
        help="Comma-separated dataset ids (default: both)",
    )
    batch_p.add_argument(
        "--classifiers",
        default="",
        help="Comma-separated classifiers (default: all)",
    )
    batch_p.add_argument(
        "--features",
        default="ts,tdp,psd,csp,fbcsp",
        help="Comma-separated descriptors (default: ts,tdp,psd,csp,fbcsp)",
    )
    batch_p.add_argument(
        "--evaluations",
        default="nested",
        help="Comma-separated schemes (default: nested)",
    )
    batch_p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel folds and subject reads (default: CPU count)",
    )
    batch_p.add_argument(
        "--spatial",
        default=",".join(SPATIAL_METHODS),
        help="Comma-separated spatial methods (default: none,car,hjorth,large_laplacian,spherical_spline)",
    )
    batch_p.add_argument(
        "--alignment",
        default="none,euclidean",
        help="Comma-separated alignments (default: none,euclidean)",
    )
    batch_p.add_argument("--notch", type=float, default=50.0)
    batch_p.add_argument("--band-low", type=float, default=8.0)
    batch_p.add_argument("--band-high", type=float, default=30.0)
    batch_p.add_argument("--order", type=int, default=4)
    batch_p.add_argument(
        "--run",
        action="store_true",
        help="Train the nested comparison. Without this flag a nested batch only estimates the fits",
    )
    batch_p.add_argument(
        "--output",
        default=None,
        help="JSON path (default: batch-results/<timestamp>.json)",
    )

    paper_p = sub.add_parser(
        "paper-csp",
        help="Score MDM and TWFB after every spatial method, both alignments, and CSP",
    )
    paper_p.add_argument("--datasets", default="shuqfa2024,liu2024_stroke")
    paper_p.add_argument("--models", default="mdm,twfb_dgfmdr")
    paper_p.add_argument(
        "--spatial",
        default="none,car,hjorth,large_laplacian,spherical_spline",
    )
    paper_p.add_argument("--alignment", default="none,euclidean")
    paper_p.add_argument(
        "--output",
        default=None,
        help="JSON path (default: batch-results/paper-csp.json). The same path resumes.",
    )
    paper_p.add_argument(
        "--run",
        action="store_true",
        help="Train the models. Without this flag only the job list is printed",
    )

    record_p = sub.add_parser("record", help="Record a Cyton LSL stream to a session folder")
    record_p.add_argument("--session-id", required=True)
    record_p.add_argument("--duration", type=float, required=True, help="Seconds to record")
    record_p.add_argument("--output", default=None, help="Recordings root (default: pipeline/recordings)")
    record_p.add_argument(
        "--stream-name",
        default=None,
        help="Exact LSL name (default: first stream of type EEG)",
    )

    online_p = sub.add_parser(
        "online-sim",
        help="Play a dataset on the simulator and classify each imagery window online",
    )
    online_p.add_argument(
        "--dataset",
        default="shuqfa2024",
        help="Dataset id for the simulator (default: shuqfa2024)",
    )
    online_p.add_argument("--subject", default=None, help="Participant id (default: first listed)")
    online_p.add_argument(
        "--session-dir",
        default=None,
        help="Recorded session folder when dataset=recorded_session",
    )
    online_p.add_argument(
        "--classifiers",
        default="linear_svm,logistic_regression",
        help=(
            "Comma-separated frozen heads scored on every window "
            "(default: linear_svm,logistic_regression)"
        ),
    )
    online_p.add_argument(
        "--classifier",
        default=None,
        choices=("linear_svm", "logistic_regression"),
        help="Single classifier (alias). Overrides --classifiers when set",
    )
    online_p.add_argument(
        "--control-classifier",
        default=None,
        help="Which head drives the stability gate / avatar (default: first listed)",
    )
    online_p.add_argument(
        "--model",
        default=None,
        help="Path to a saved frozen model (.joblib). When omitted, train from the dataset",
    )
    online_p.add_argument(
        "--registry",
        default=None,
        help="Load a registered offline export by id or name (from pipeline offline-export)",
    )
    online_p.add_argument(
        "--registry-latest",
        action="store_true",
        help="Load the latest registered offline export (optionally filtered by --dataset/--subject)",
    )
    online_p.add_argument(
        "--save-model",
        default=None,
        help="Where to write the frozen model after training",
    )
    online_p.add_argument(
        "--register",
        action="store_true",
        help="After training, also register the model under online-models/ for later online runs",
    )
    online_p.add_argument(
        "--register-as",
        default=None,
        help="Name used when --register is set (default: <dataset>-<subject>-online)",
    )

    replay_p = sub.add_parser(
        "online-replay",
        help="Classify a dataset file online without LSL (Slurm / headless)",
    )
    replay_p.add_argument("--dataset", default="shuqfa2024")
    replay_p.add_argument("--subject", default=None)
    replay_p.add_argument("--session-dir", default=None)
    replay_p.add_argument(
        "--registry",
        required=True,
        help="Registered offline export id or name",
    )
    replay_p.add_argument("--max-trials", type=int, default=None)
    replay_p.add_argument("--log", default=None)
    replay_p.add_argument(
        "--dump-seconds",
        type=float,
        default=200.0,
        help="How many seconds of the recording to replay (default: 200)",
    )

    offline_export_p = sub.add_parser(
        "offline-export",
        help="Train offline mode(s) and register frozen models for online-sim",
    )
    offline_export_p.add_argument(
        "--config",
        default=None,
        help="JSON file with one mode or {modes: [...]} offline recipes",
    )
    offline_export_p.add_argument("--name", default=None, help="Mode name when not using --config")
    offline_export_p.add_argument("--dataset", default=None)
    offline_export_p.add_argument(
        "--subject",
        default=None,
        help="Legacy alias for --online-subject (verification subject only)",
    )
    offline_export_p.add_argument(
        "--online-subject",
        default=None,
        help="Subject used later for online verification (training uses all subjects)",
    )
    offline_export_p.add_argument("--session-dir", default=None)
    offline_export_p.add_argument(
        "--classifiers",
        default="linear_svm,logistic_regression",
        help="Comma-separated classifiers (default: linear_svm,logistic_regression)",
    )
    offline_export_p.add_argument("--control-classifier", default=None)
    offline_export_p.add_argument(
        "--cross-subject",
        dest="cross_subject",
        action="store_true",
        default=True,
        help="Train on all subjects (default)",
    )
    offline_export_p.add_argument(
        "--single-subject",
        dest="cross_subject",
        action="store_false",
        help="Legacy smoke: train on --online-subject / --subject only",
    )
    offline_export_p.add_argument("--notch", type=float, default=50.0)
    offline_export_p.add_argument("--band-low", type=float, default=8.0)
    offline_export_p.add_argument("--band-high", type=float, default=30.0)
    offline_export_p.add_argument("--order", type=int, default=4)
    offline_export_p.add_argument("--csp-pairs", type=int, default=2)
    offline_export_p.add_argument(
        "--registry-root",
        default=None,
        help="Registry directory (default: pipeline/online-models)",
    )
    offline_export_p.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite an existing registry id with the same name stamp collision",
    )

    offline_list_p = sub.add_parser(
        "offline-list",
        help="List offline models registered for online use",
    )
    offline_list_p.add_argument(
        "--registry-root",
        default=None,
        help="Registry directory (default: pipeline/online-models)",
    )
    offline_list_p.add_argument("--dataset", default=None, help="Filter by dataset id")
    offline_list_p.add_argument("--subject", default=None, help="Filter by subject id")
    online_p.add_argument("--max-trials", type=int, default=None, help="Stop after N hand trials")
    online_p.add_argument(
        "--log",
        default=None,
        help="JSONL path for per-window decisions (default: batch-results/online-<stamp>.jsonl)",
    )
    online_p.add_argument("--wait", type=float, default=8.0, help="LSL discover timeout")
    online_p.add_argument("--notch", type=float, default=50.0)
    online_p.add_argument("--band-low", type=float, default=8.0)
    online_p.add_argument("--band-high", type=float, default=30.0)
    online_p.add_argument("--order", type=int, default=4)
    online_p.add_argument("--csp-pairs", type=int, default=2)
    return parser


def _cmd_list(args: argparse.Namespace) -> int:
    from pipeline.reader import list_streams

    streams = list_streams(
        stream_type=args.stream_type,
        name=args.name,
        wait_seconds=args.wait,
    )
    if not streams:
        print("No LSL streams found.")
        return 1

    print(f"Found {len(streams)} stream(s):\n")
    for i, stream in enumerate(streams, start=1):
        print(f"  {i}. {stream.display_name}")
    return 0


def _cmd_read(args: argparse.Namespace) -> int:
    from pipeline.reader import LslStreamReader

    print(
        f"Looking for LSL stream "
        f"(name={args.name!r}, type={args.stream_type!r}, wait={args.wait}s)…"
    )
    try:
        reader = LslStreamReader.open(
            name=args.name,
            stream_type=None if args.name else args.stream_type,
            wait_seconds=args.wait,
        )
    except TimeoutError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Connected: {reader.summary.display_name}")
    print("Reading… (Ctrl+C to stop)\n")

    started = time.monotonic()
    chunks = 0
    total_samples = 0
    try:
        for chunk in reader.iter_chunks(
            timeout=args.timeout,
            max_samples=args.max_samples,
            max_chunks=args.max_chunks,
        ):
            chunks += 1
            total_samples += len(chunk.samples)
            first = chunk.samples[0]
            preview = ", ".join(f"{v:.4f}" for v in first[:8])
            if len(first) > 8:
                preview += ", …"
            print(
                f"[{chunks}] n={len(chunk.samples)} "
                f"t0={chunk.timestamps[0]:.4f} sample=[{preview}]"
            )
            if args.duration is not None and (time.monotonic() - started) >= args.duration:
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        reader.close()

    elapsed = time.monotonic() - started
    print(f"\nDone: {total_samples} samples in {chunks} chunks ({elapsed:.1f}s).")
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    from pipeline.visualizer import run_visualizer

    try:
        return run_visualizer(
            name=args.name,
            stream_type=args.stream_type,
            wait_seconds=args.wait,
            window_seconds=args.window,
            refresh_hz=args.refresh,
            channels=args.channels,
            scale=args.scale,
            markers_name=args.markers_name,
            no_markers=args.no_markers,
        )
    except TimeoutError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _cmd_show(args: argparse.Namespace) -> int:
    from pipeline.session import load_recording, sample_index_for_time

    try:
        recording = load_recording(args.path)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    duration = 0.0
    if recording.n_samples and recording.sample_rate_hz:
        duration = recording.n_samples / recording.sample_rate_hz
    device = recording.device
    print(f"session: {recording.session_id}")
    print(f"path: {recording.root}")
    print(
        f"signal: {recording.n_samples} samples, {recording.n_channels} ch, "
        f"{recording.sample_rate_hz:g} Hz, {recording.units}, {duration:.2f}s"
    )
    print(f"channels: {', '.join(recording.channel_labels)}")
    print(
        f"device: {device.get('board')}  ground={device.get('ground')}  "
        f"reference={device.get('reference')}"
    )
    print(f"markers: {len(recording.markers)}")
    for marker in recording.markers:
        index = sample_index_for_time(recording.timestamps, marker.lsl_time)
        print(f"  [{index}] {marker.kind}:{marker.code}:{marker.label} t={marker.lsl_time:.4f}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from pathlib import Path

    from pipeline.service import default_recordings_root, serve

    root = Path(args.recordings) if args.recordings else default_recordings_root()
    serve(host=args.host, port=args.port, recordings_root=root)
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    import os
    from datetime import datetime, timezone

    from pipeline.batch import dry_run_estimate, run_batch
    from pipeline.service import PipelineService, default_recordings_root

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    output = args.output
    if not output:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = str(default_recordings_root().parent / "batch-results" / f"{stamp}.json")
    service = PipelineService(default_recordings_root())

    def report(row: dict) -> None:
        if row.get("error"):
            print(
                f"{row['dataset']} · {row.get('spatial') or 'none'} · "
                f"{row.get('alignment') or 'none'} · {row['features']} · {row['evaluation']} · "
                f"{row['classifier']} · {row['error']}",
                flush=True,
            )
            return
        accuracy = row.get("accuracy")
        shown = f"{accuracy:.0%}" if isinstance(accuracy, float) else ""
        print(
            f"{row['dataset']} · {row.get('spatial') or 'none'} · "
            f"{row.get('alignment') or 'none'} · {row['features']} · {row['evaluation']} · "
            f"{row['classifier']} · {shown} · {row['seconds']:.0f}s",
            flush=True,
        )

    destination = Path(output)
    evaluations = _csv(args.evaluations)
    nested = any(item in {"nested", "nested_loso"} for item in evaluations)
    if nested and not args.run:
        try:
            estimate = dry_run_estimate(
                service,
                datasets=_csv(args.datasets),
                classifiers=_csv(args.classifiers),
                features=_csv(args.features),
                evaluations=evaluations,
                spatial=_csv(args.spatial),
                alignment=_csv(args.alignment),
            )
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        estimate_path = destination.with_name(f"{destination.stem}-estimate.json")
        estimate_path.parent.mkdir(parents=True, exist_ok=True)
        estimate_path.write_text(json.dumps(estimate, indent=2), encoding="utf-8")
        print(
            f"Dry run: {estimate['total_fits']} fits across {estimate['pipelines']} pipelines.",
            flush=True,
        )
        print("No model was trained. Pass --run to start, and power the machine off when it finishes.", flush=True)
        print(f"Wrote the estimate to {estimate_path}", flush=True)
        return 0
    try:
        results = run_batch(
            service,
            datasets=_csv(args.datasets),
            classifiers=_csv(args.classifiers),
            features=_csv(args.features),
            evaluations=evaluations,
            workers=workers,
            spatial=_csv(args.spatial),
            alignment=_csv(args.alignment),
            notch_hz=args.notch,
            band_low_hz=args.band_low,
            band_high_hz=args.band_high,
            order=args.order,
            on_result=report,
            output=destination,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    table = destination.with_name(f"{destination.stem}-comparison.csv")
    print(f"Wrote {len(results)} rows to {destination}", flush=True)
    print(f"Comparison, best accuracy first: {table}", flush=True)
    return 0


def _cmd_paper_csp(args: argparse.Namespace) -> int:
    import os
    from pathlib import Path

    from pipeline.classify.paper_models import PAPER_MODELS, score_csp_grid
    from pipeline.service import PipelineService, default_recordings_root
    from pipeline.spatial import METHODS as SPATIAL_METHODS

    datasets = _csv(args.datasets)
    models = _csv(args.models)
    spatials = _csv(args.spatial)
    alignments = _csv(args.alignment)
    unknown_models = [name for name in models if name not in PAPER_MODELS]
    unknown_spatial = [name for name in spatials if name not in SPATIAL_METHODS]
    unknown_alignment = [name for name in alignments if name not in {"none", "euclidean"}]
    if unknown_models or unknown_spatial or unknown_alignment or not datasets:
        print("error: unknown dataset, model, spatial method, or alignment", file=sys.stderr)
        return 1
    variants = [(spatial, alignment) for spatial in spatials for alignment in alignments]
    jobs = len(datasets) * len(models) * len(variants)
    destination = Path(args.output) if args.output else default_recordings_root().parent / "batch-results" / "paper-csp.json"
    cores = os.cpu_count() or 1
    print(
        f"{jobs} jobs · {len(models)} models · {len(variants)} signals · {cores} cores, one BLAS thread each.",
        flush=True,
    )
    print(f"Results: {destination}", flush=True)
    print(f"Comparison: {destination.with_name(destination.stem + '-comparison.csv')}", flush=True)
    if not args.run:
        print("No model was trained. Pass --run to start.", flush=True)
        return 0
    service = PipelineService(default_recordings_root())
    try:
        for dataset in datasets:
            cohort = service.load_cohort(
                dataset=dataset,
                spatial="none",
                alignment="none",
            )
            score_csp_grid(
                cohort["groups"],
                float(cohort["rate"]),
                list(cohort["channels"]),
                destination,
                dataset=str(cohort["dataset"]),
                models=models,
                variants=variants,
            )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {destination}", flush=True)
    return 0


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _cmd_record(args: argparse.Namespace) -> int:
    from pathlib import Path

    from pipeline.service import PipelineService, default_recordings_root

    root = Path(args.output) if args.output else default_recordings_root()
    service = PipelineService(root)
    print(
        f"Recording Cyton stream into {root / args.session_id} for {args.duration:g}s…",
        flush=True,
    )
    try:
        started = service.start_recording(
            args.session_id,
            stream_name=args.stream_name,
        )
    except (TimeoutError, ValueError, FileExistsError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Connected: {started['stream_name']}", flush=True)
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        result = service.stop_recording()
    print(
        f"Saved {result.get('n_samples', 0)} samples and "
        f"{result.get('n_markers', 0)} markers to {result.get('path')}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return _cmd_list(args)
    if args.command == "read":
        return _cmd_read(args)
    if args.command == "view":
        return _cmd_view(args)
    if args.command == "show":
        return _cmd_show(args)
    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "batch":
        return _cmd_batch(args)
    if args.command == "paper-csp":
        return _cmd_paper_csp(args)
    if args.command == "record":
        return _cmd_record(args)
    if args.command == "online-sim":
        return _cmd_online_sim(args)
    if args.command == "online-replay":
        return _cmd_online_replay(args)
    if args.command == "offline-export":
        return _cmd_offline_export(args)
    if args.command == "offline-list":
        return _cmd_offline_list(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _cmd_online_replay(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    from pipeline.online.registry import OnlineModelRegistry
    from pipeline.online.replay_session import ReplayOnlineConfig, ReplayOnlineSession
    from pipeline.service import default_recordings_root

    log_path = Path(args.log) if args.log else (
        default_recordings_root().parent
        / "batch-results"
        / f"online-replay-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.jsonl"
    )
    try:
        registry = OnlineModelRegistry()
        entry = registry.get(args.registry)
        model = registry.load_model(args.registry)
        dataset = str(entry.offline.get("dataset") or args.dataset)
        subject = (
            args.subject
            or str(entry.offline.get("online_subject") or "")
            or str(entry.offline.get("subject") or "")
            or None
        )
        session_dir = args.session_dir or str(entry.offline.get("session_dir") or "") or None
        n_train = entry.offline.get("n_train_subjects")
        print(
            f"Replay registry {entry.id} · dataset={dataset} · "
            f"online_subject={subject or '-'} · n_train={n_train} · "
            f"heads={list(model.classifier_names)}",
            flush=True,
        )
        summary = ReplayOnlineSession(
            model,
            ReplayOnlineConfig(
                dataset=dataset,
                subject=subject,
                session_dir=session_dir,
                max_trials=args.max_trials,
                log_path=log_path,
                dump_seconds=args.dump_seconds,
            ),
        ).run()
    except (TimeoutError, ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Log: {log_path}", flush=True)
    return 0


def _cmd_offline_export(args: argparse.Namespace) -> int:
    from pipeline.online.registry import (
        OfflineModeSpec,
        OnlineModelRegistry,
        export_offline_modes,
        load_offline_modes,
    )

    try:
        if args.config:
            modes = load_offline_modes(args.config)
        else:
            if not args.dataset or not args.name:
                print(
                    "error: provide --config, or both --name and --dataset",
                    file=sys.stderr,
                )
                return 2
            online_subject = args.online_subject or args.subject
            modes = [
                OfflineModeSpec(
                    name=args.name,
                    dataset=args.dataset,
                    online_subject=online_subject,
                    subject=online_subject,
                    session_dir=args.session_dir,
                    classifiers=tuple(
                        part.strip()
                        for part in str(args.classifiers).split(",")
                        if part.strip()
                    ),
                    control_classifier=args.control_classifier,
                    cross_subject=bool(args.cross_subject),
                    hold_out_online_subject=bool(args.cross_subject),
                    notch_hz=args.notch,
                    band_low_hz=args.band_low,
                    band_high_hz=args.band_high,
                    order=args.order,
                    csp_pairs=args.csp_pairs,
                )
            ]
        registry = OnlineModelRegistry(
            Path(args.registry_root) if args.registry_root else None
        )
        print(f"Exporting {len(modes)} offline mode(s) → {registry.root}", flush=True)
        entries = export_offline_modes(modes, registry=registry, replace=args.replace)
    except (TimeoutError, ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for entry in entries:
        offline = ", ".join(
            f"{name}={acc:.0%}" if acc is not None else f"{name}=n/a"
            for name, acc in entry.offline_accuracies.items()
        )
        n_train = entry.offline.get("n_train_subjects")
        verify = entry.offline.get("online_subject") or entry.offline.get("subject") or "-"
        held = entry.offline.get("held_out_online_subject")
        print(
            f"registered {entry.id} · name={entry.name} · "
            f"dataset={entry.offline.get('dataset')} · "
            f"cross_subject={entry.offline.get('cross_subject')} · "
            f"held_out={held} · n_train={n_train} · online_subject={verify} · "
            f"heads={entry.classifiers} · offline=[{offline}]",
            flush=True,
        )
    print(
        "Use: uv run pipeline online-sim --registry <id-or-name>",
        flush=True,
    )
    return 0


def _cmd_offline_list(args: argparse.Namespace) -> int:
    from pipeline.online.registry import OnlineModelRegistry

    registry = OnlineModelRegistry(Path(args.registry_root) if args.registry_root else None)
    entries = registry.list_entries()
    if args.dataset:
        entries = [
            item for item in entries if str(item.offline.get("dataset") or "") == args.dataset
        ]
    if args.subject is not None:
        entries = [
            item
            for item in entries
            if str(
                item.offline.get("online_subject") or item.offline.get("subject") or ""
            )
            == str(args.subject)
        ]
    if not entries:
        print("No registered offline models.")
        return 0
    for entry in entries:
        offline = ", ".join(
            f"{name}={acc:.0%}" if acc is not None else f"{name}=n/a"
            for name, acc in entry.offline_accuracies.items()
        )
        verify = entry.offline.get("online_subject") or entry.offline.get("subject") or "-"
        n_train = entry.offline.get("n_train_subjects")
        xs = "xs" if entry.offline.get("cross_subject") else "ss"
        print(
            f"{entry.id}\t{entry.name}\t"
            f"{entry.offline.get('dataset')}/online={verify}"
            f"{f'/n_train={n_train}' if n_train is not None else ''}/{xs}\t"
            f"{','.join(entry.classifiers)}\t[{offline}]"
        )
    return 0


def _cmd_online_sim(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    from pipeline.online.dataset_model import train_frozen_from_dataset
    from pipeline.online.model import load_frozen_model, save_frozen_model
    from pipeline.online.registry import OfflineModeSpec, OnlineModelRegistry
    from pipeline.online.simulator_session import (
        SimulatorOnlineConfig,
        SimulatorOnlineSession,
    )
    from pipeline.preprocess import FilterSpec
    from pipeline.service import default_recordings_root

    log_path = Path(args.log) if args.log else (
        default_recordings_root().parent
        / "batch-results"
        / f"online-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.jsonl"
    )
    try:
        registry = OnlineModelRegistry()
        dataset = args.dataset
        subject = args.subject
        session_dir = args.session_dir
        if args.registry or args.registry_latest:
            if args.registry:
                entry = registry.get(args.registry)
                model = registry.load_model(args.registry)
            else:
                entry = registry.latest(dataset=args.dataset, subject=args.subject)
                model = registry.load_model(entry.id)
            dataset = str(entry.offline.get("dataset") or dataset)
            subject = (
                subject
                or str(entry.offline.get("online_subject") or "")
                or str(entry.offline.get("subject") or "")
                or None
            )
            session_dir = str(entry.offline.get("session_dir") or "") or session_dir
            print(
                f"Loaded registry {entry.id} · name={entry.name} · "
                f"online_subject={subject or '-'} · "
                f"n_train={entry.offline.get('n_train_subjects')} · "
                f"heads={list(model.classifier_names)}",
                flush=True,
            )
        elif args.model:
            model = load_frozen_model(args.model)
            print(
                f"Loaded frozen model from {args.model} · heads={list(model.classifier_names)}",
                flush=True,
            )
        else:
            classifiers = args.classifier or args.classifiers
            print(
                f"Training frozen [{classifiers}] on {dataset}"
                + (f" / {subject}" if subject else "")
                + "…",
                flush=True,
            )
            model = train_frozen_from_dataset(
                dataset=dataset,
                subject=subject,
                session_dir=session_dir,
                classifier_names=None if args.classifier else args.classifiers,
                classifier_name=args.classifier,
                control_classifier=args.control_classifier,
                filter=FilterSpec(
                    notch_hz=args.notch,
                    band_low_hz=args.band_low,
                    band_high_hz=args.band_high,
                    order=args.order,
                ),
                csp_pairs=args.csp_pairs,
            )
            offline = ", ".join(
                f"{name}={acc:.0%}" if acc is not None else f"{name}=n/a"
                for name, acc in model.offline_accuracies.items()
            )
            print(
                f"Offline train accuracy: {offline} · "
                f"{model.channel_count} ch · control={model.control_name} · {model.meta}",
                flush=True,
            )
            if args.save_model:
                saved = save_frozen_model(model, args.save_model)
                print(f"Wrote frozen model to {saved}", flush=True)
            if args.register:
                reg_name = args.register_as or (
                    f"{dataset}-{subject or 'all'}-online"
                )
                entry = registry.register(
                    model,
                    OfflineModeSpec(
                        name=reg_name,
                        dataset=dataset,
                        online_subject=subject,
                        subject=subject,
                        session_dir=session_dir,
                        classifiers=tuple(model.classifier_names),
                        control_classifier=model.control_name,
                        cross_subject=bool((model.meta or {}).get("cross_subject", True)),
                        notch_hz=args.notch,
                        band_low_hz=args.band_low,
                        band_high_hz=args.band_high,
                        order=args.order,
                        csp_pairs=args.csp_pairs,
                    ),
                )
                print(f"Registered offline export as {entry.id}", flush=True)

        config = SimulatorOnlineConfig(
            dataset=dataset,
            subject=subject,
            session_dir=session_dir,
            wait_seconds=args.wait,
            max_trials=args.max_trials,
            log_path=log_path,
        )
        print(
            f"Starting simulator playback → online windows → {list(model.classifier_names)}",
            flush=True,
        )
        summary = SimulatorOnlineSession(model, config).run()
    except (TimeoutError, ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Log: {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
