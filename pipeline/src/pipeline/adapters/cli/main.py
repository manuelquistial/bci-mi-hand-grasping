"""Entry point and command dispatch for the ``pipeline`` CLI."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from pipeline.adapters.cli.cmd_batch import cmd_batch, cmd_paper_csp
from pipeline.adapters.cli.cmd_offline import cmd_offline_export, cmd_offline_list
from pipeline.adapters.cli.cmd_online import cmd_online_replay, cmd_online_sim
from pipeline.adapters.cli.cmd_session import cmd_record, cmd_serve
from pipeline.adapters.cli.cmd_stream import cmd_list, cmd_read, cmd_show, cmd_view
from pipeline.adapters.cli.parser import build_parser

COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "list": cmd_list,
    "read": cmd_read,
    "view": cmd_view,
    "show": cmd_show,
    "serve": cmd_serve,
    "batch": cmd_batch,
    "paper-csp": cmd_paper_csp,
    "record": cmd_record,
    "online-sim": cmd_online_sim,
    "online-replay": cmd_online_replay,
    "offline-export": cmd_offline_export,
    "offline-list": cmd_offline_list,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMANDS.get(args.command)
    if handler is not None:
        return handler(args)
    parser.error(f"Unknown command: {args.command}")
    return 2
