"""Top-level argument parser for the ``pipeline`` CLI."""

from __future__ import annotations

import argparse

from pipeline.adapters.cli.parser_batch import register_batch_commands
from pipeline.adapters.cli.parser_online import register_online_commands
from pipeline.adapters.cli.parser_stream import (
    register_record_command,
    register_stream_commands,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Discover, read, and visualize Lab Streaming Layer (LSL) streams.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    register_stream_commands(sub)
    register_batch_commands(sub)
    register_record_command(sub)
    register_online_commands(sub)
    return parser
