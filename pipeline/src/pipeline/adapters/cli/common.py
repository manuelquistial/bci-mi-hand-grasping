"""Helpers shared by CLI parsers and command handlers."""

from __future__ import annotations

import argparse


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def add_stream_select_args(parser: argparse.ArgumentParser, *, default_type: str | None) -> None:
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
