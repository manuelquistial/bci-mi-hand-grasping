"""Argument parsers for LSL stream commands (list, read, view, show, serve, record)."""

from __future__ import annotations

import argparse

from pipeline.adapters.cli.common import add_stream_select_args


def register_stream_commands(sub: argparse._SubParsersAction) -> None:
    list_p = sub.add_parser("list", help="List available LSL streams")
    add_stream_select_args(list_p, default_type=None)
    list_p.set_defaults(wait=2.0)

    read_p = sub.add_parser("read", help="Connect to a stream and print samples")
    add_stream_select_args(read_p, default_type="EEG")
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
    add_stream_select_args(view_p, default_type="EEG")
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


def register_record_command(sub: argparse._SubParsersAction) -> None:
    record_p = sub.add_parser("record", help="Record a Cyton LSL stream to a session folder")
    record_p.add_argument("--session-id", required=True)
    record_p.add_argument("--duration", type=float, required=True, help="Seconds to record")
    record_p.add_argument("--output", default=None, help="Recordings root (default: pipeline/recordings)")
    record_p.add_argument(
        "--stream-name",
        default=None,
        help="Exact LSL name (default: first stream of type EEG)",
    )
