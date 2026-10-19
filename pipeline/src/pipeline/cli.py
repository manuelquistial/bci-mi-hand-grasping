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

    record_p = sub.add_parser("record", help="Record a Cyton LSL stream to a session folder")
    record_p.add_argument("--session-id", required=True)
    record_p.add_argument("--duration", type=float, required=True, help="Seconds to record")
    record_p.add_argument("--output", default=None, help="Recordings root (default: pipeline/recordings)")
    record_p.add_argument(
        "--stream-name",
        default=None,
        help="Exact LSL name (default: first stream of type EEG)",
    )
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
    if args.command == "record":
        return _cmd_record(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
