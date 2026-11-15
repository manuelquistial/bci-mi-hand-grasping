"""JSON dump modes (``--dump-seconds`` / ``--dump-trials``). No LSL is opened."""
from __future__ import annotations

import json
import sys

from dataset_loaders import BaseDatasetLoader
from dataset_loaders.events import resolve_events


def dump_seconds(loader: BaseDatasetLoader, file_path: str, seconds: float) -> int:
    """Write one JSON object of the opening window. Nothing else goes to stdout."""
    if seconds <= 0:
        print("Error: --dump-seconds must be greater than 0", file=sys.stderr)
        return 1
    rate = float(loader.metadata.sampling_rate)
    if rate <= 0:
        print("Error: dataset sample rate is missing", file=sys.stderr)
        return 1
    limit = max(1, int(round(seconds * rate)))
    rows: list[list[float]] = []
    for index, sample in enumerate(loader.iter_samples(file_path)):
        if index >= limit:
            break
        rows.append([float(value) for value in sample])
    if not rows:
        print(f"Error: no samples in {file_path}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "sample_rate_hz": rate,
                "channels": list(loader.metadata.channel_names or []),
                "samples": rows,
            }
        )
    )
    return 0


def dump_trials(loader: BaseDatasetLoader, file_path: str, seconds: float) -> int:
    """Write every complete marker window. The caller decides which are imagery."""
    if seconds <= 0:
        print("Error: --trial-seconds must be greater than 0", file=sys.stderr)
        return 1
    rate = float(loader.metadata.sampling_rate)
    if rate <= 0:
        print("Error: dataset sample rate is missing", file=sys.stderr)
        return 1
    events_from_loader = getattr(loader, "load_events", None)
    events = events_from_loader(file_path) if callable(events_from_loader) else []
    if not events:
        events = resolve_events(file_path, None)
    width = max(1, int(round(seconds * rate)))
    spans = [
        (int(event.sample_index), int(event.sample_index) + width, event)
        for event in events
        if int(event.sample_index) >= 0
    ]
    if not spans:
        print(f"Error: no markers in {file_path}", file=sys.stderr)
        return 1
    last = max(end for _start, end, _event in spans)
    rows: list[list[float]] = []
    for index, sample in enumerate(loader.iter_samples(file_path)):
        rows.append([float(value) for value in sample])
        if index + 1 >= last:
            break
    trials = []
    for start, end, event in spans:
        if end > len(rows):
            continue
        trials.append(
            {
                "onset": start / rate,
                "kind": event.kind,
                "label": event.label,
                "samples": rows[start:end],
            }
        )
    # Window slices keep the samples they need. Drop the rest of the recording.
    del rows
    if not trials:
        print(f"Error: no complete trial windows in {file_path}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "sample_rate_hz": rate,
                "channels": list(loader.metadata.channel_names or []),
                "trials": trials,
            }
        )
    )
    return 0
