"""Session service commands: serve, record."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def cmd_serve(args: argparse.Namespace) -> int:
    from pipeline.service import default_recordings_root, serve

    root = Path(args.recordings) if args.recordings else default_recordings_root()
    serve(host=args.host, port=args.port, recordings_root=root)
    return 0


def cmd_record(args: argparse.Namespace) -> int:
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
