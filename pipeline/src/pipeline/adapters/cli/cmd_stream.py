"""LSL stream commands: list, read, view, show."""

from __future__ import annotations

import argparse
import sys
import time


def cmd_list(args: argparse.Namespace) -> int:
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


def cmd_read(args: argparse.Namespace) -> int:
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


def cmd_view(args: argparse.Namespace) -> int:
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


def cmd_show(args: argparse.Namespace) -> int:
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
