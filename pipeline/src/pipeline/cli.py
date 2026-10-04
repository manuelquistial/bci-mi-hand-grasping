"""CLI for discovering, reading, and visualizing Lab Streaming Layer (LSL) streams."""

from __future__ import annotations

import argparse
import sys
import time


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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return _cmd_list(args)
    if args.command == "read":
        return _cmd_read(args)
    if args.command == "view":
        return _cmd_view(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
