"""Compose loader, sinks, markers and notifier into a playback run."""
from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from pathlib import Path

from dataset_loaders import BaseDatasetLoader
from dataset_loaders.events import resolve_events

from ..clock import RealtimeClock
from ..config import PlaybackConfig
from ..playback import PlaybackService
from ..ports import Notifier, SampleSink
from ..sinks.lsl import LslSampleSink
from ..sinks.markers import LslMarkerSink

SinkFactory = Callable[[], SampleSink]


def run_playback(
    args: Namespace,
    loader: BaseDatasetLoader,
    file_path: str,
    notifier: Notifier,
    sink_factory: SinkFactory | None = None,
) -> int:
    """Stream the file through PlaybackService. Returns a process exit code."""
    factory = sink_factory or (lambda: LslSampleSink())
    sink = factory()
    use_lsl_timing = sink_factory is None
    no_dialog = bool(args.no_dialog)
    config = PlaybackConfig(
        max_samples=args.max_samples,
        stream_name=args.lsl_name,
        notify_finish=not no_dialog,
        # LSL discovery needs the outlet to stay up for ~1s+ on the network.
        advertise_seconds=1.5 if use_lsl_timing else 0.0,
        hold_seconds=0.5 if use_lsl_timing else 0.0,
    )

    events_from_loader = getattr(loader, "load_events", None)
    if args.events:
        events = resolve_events(file_path, args.events)
        events_origin = args.events
    elif callable(events_from_loader):
        events = events_from_loader(file_path)
        events_origin = "session markers"
    else:
        events = resolve_events(file_path, None)
        events_origin = Path(file_path).with_suffix(".events.csv").name
    marker_sink = None
    if events and not args.no_markers and use_lsl_timing:
        marker_sink = LslMarkerSink()

    if events:
        notifier.progress(f"Events: {events_origin} ({len(events)} markers)")
    elif args.events:
        notifier.warning(f"No events loaded from {args.events}")

    try:
        PlaybackService().run(
            loader=loader,
            file_path=file_path,
            sink=sink,
            clock=RealtimeClock(),
            notifier=notifier,
            config=config,
            events=events,
            marker_sink=marker_sink,
        )
    except Exception:
        return 1
    return 0
