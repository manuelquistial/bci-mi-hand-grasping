"""Application service: stream EEG samples from a loader to a sink."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from dataset_loaders.base import BaseDatasetLoader
from dataset_loaders.events import TrialEvent

from .config import PlaybackConfig
from .ports import Clock, Notifier, SampleSink


class PlaybackService:
    """Orchestrates loader → sink playback without UI/LSL coupling."""

    def run(
        self,
        loader: BaseDatasetLoader,
        file_path: str | Path,
        sink: SampleSink,
        clock: Clock,
        notifier: Notifier,
        config: PlaybackConfig | None = None,
        *,
        events: list[TrialEvent] | None = None,
        marker_sink: Any | None = None,
    ) -> int:
        """
        Stream samples from file_path using loader into sink.
        When ``events`` and ``marker_sink`` are provided, emits markers at the
        matching sample indices (synchronized to the EEG push timestamp when available).
        Returns the number of samples pushed.
        Raises on read errors after notifying.
        """
        cfg = config or PlaybackConfig()
        path = Path(file_path)
        sink.open(loader.metadata, stream_name=cfg.stream_name)
        eeg_name = cfg.stream_name or loader.metadata.name
        if marker_sink is not None:
            marker_sink.open(eeg_name)

        pending = sorted(events or [], key=lambda e: e.sample_index)
        event_i = 0

        if cfg.advertise_seconds > 0:
            notifier.progress(
                f"LSL outlet open — waiting {cfg.advertise_seconds:.1f}s for discovery..."
            )
            if pending:
                notifier.progress(f"Loaded {len(pending)} trial/event marker(s)")
            time.sleep(cfg.advertise_seconds)
        clock.start()
        fs = loader.metadata.sampling_rate
        count = 0
        try:
            for sample in loader.iter_samples(path):
                push_result = sink.push(sample)
                ts = push_result if isinstance(push_result, (int, float)) else None
                count += 1
                sample_index = count - 1

                while event_i < len(pending) and pending[event_i].sample_index == sample_index:
                    event = pending[event_i]
                    if marker_sink is not None:
                        marker_sink.push(event, timestamp=ts)
                    event_i += 1

                if cfg.progress_every > 0 and count % cfg.progress_every == 0:
                    notifier.progress(f"Samples sent: {count}")
                if cfg.max_samples is not None and count >= cfg.max_samples:
                    break
                clock.wait_for_sample(count - 1, fs)
        except Exception as exc:
            notifier.error(str(exc), title="Error reading file")
            raise
        finally:
            if cfg.hold_seconds > 0:
                time.sleep(cfg.hold_seconds)
            if marker_sink is not None:
                marker_sink.close()
            sink.close()

        if cfg.notify_finish:
            notifier.info(
                f"All data has been read successfully.\nSamples sent: {count}",
                title="Read finished",
            )
        else:
            notifier.progress(f"Done. Samples sent: {count}")
        return count
