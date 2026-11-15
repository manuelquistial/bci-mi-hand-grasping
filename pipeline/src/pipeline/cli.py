"""Console entry point: ``pipeline`` delegates to :mod:`pipeline.adapters.cli`."""

from __future__ import annotations

from pipeline.adapters.cli.cmd_batch import cmd_batch as _cmd_batch
from pipeline.adapters.cli.cmd_batch import cmd_paper_csp as _cmd_paper_csp
from pipeline.adapters.cli.cmd_offline import cmd_offline_export as _cmd_offline_export
from pipeline.adapters.cli.cmd_offline import cmd_offline_list as _cmd_offline_list
from pipeline.adapters.cli.cmd_online import cmd_online_replay as _cmd_online_replay
from pipeline.adapters.cli.cmd_online import cmd_online_sim as _cmd_online_sim
from pipeline.adapters.cli.cmd_session import cmd_record as _cmd_record
from pipeline.adapters.cli.cmd_session import cmd_serve as _cmd_serve
from pipeline.adapters.cli.cmd_stream import cmd_list as _cmd_list
from pipeline.adapters.cli.cmd_stream import cmd_read as _cmd_read
from pipeline.adapters.cli.cmd_stream import cmd_show as _cmd_show
from pipeline.adapters.cli.cmd_stream import cmd_view as _cmd_view
from pipeline.adapters.cli.common import add_stream_select_args as _add_stream_select_args
from pipeline.adapters.cli.common import csv_list as _csv
from pipeline.adapters.cli.main import main
from pipeline.adapters.cli.parser import build_parser as _build_parser

__all__ = [
    "main",
    "_add_stream_select_args",
    "_build_parser",
    "_cmd_batch",
    "_cmd_list",
    "_cmd_offline_export",
    "_cmd_offline_list",
    "_cmd_online_replay",
    "_cmd_online_sim",
    "_cmd_paper_csp",
    "_cmd_read",
    "_cmd_record",
    "_cmd_serve",
    "_cmd_show",
    "_cmd_view",
    "_csv",
]

if __name__ == "__main__":
    raise SystemExit(main())
