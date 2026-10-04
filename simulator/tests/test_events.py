"""Tests for trial/event sidecar loading and marker emission during playback."""

from __future__ import annotations

from pathlib import Path

from dataset_loaders.events import TrialEvent, load_events_csv, resolve_events
from dataset_loaders import GenericCSVLoader

from simopenbci.clock import InstantClock
from simopenbci.config import PlaybackConfig
from simopenbci.notifiers.console import ConsoleNotifier
from simopenbci.playback import PlaybackService
from simopenbci.sinks.recording import RecordingMarkerSink, RecordingSink


class CapturingNotifier(ConsoleNotifier):
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.errors: list[str] = []
        self.progresses: list[str] = []

    def info(self, message: str, *, title: str = "Info") -> None:
        self.infos.append(message)

    def error(self, message: str, *, title: str = "Error") -> None:
        self.errors.append(message)

    def progress(self, message: str) -> None:
        self.progresses.append(message)


def test_load_events_csv(tmp_path: Path) -> None:
    path = tmp_path / "S001.events.csv"
    path.write_text(
        "sample_index,code,label,kind\n"
        "2,T1,left,trial_start\n"
        "0,T0,rest,rest\n"
        "2,T2,right,trial_start\n"
    )
    events = load_events_csv(path)
    assert [e.sample_index for e in events] == [0, 2, 2]
    assert events[1].label == "left"
    assert events[1].marker_string == "trial_start:T1:left"


def test_resolve_events_sidecar(tmp_path: Path) -> None:
    eeg = tmp_path / "S001.txt"
    eeg.write_text("1,2,3,4,5,6,7,8\n")
    (tmp_path / "S001.events.csv").write_text(
        "sample_index,code,label,kind\n1,T1,left,trial_start\n"
    )
    events = resolve_events(eeg)
    assert len(events) == 1
    assert events[0].code == "T1"


def test_playback_emits_markers_at_sample_index(generic_sample: Path) -> None:
    sink = RecordingSink()
    markers = RecordingMarkerSink()
    events = [
        TrialEvent(0, "T0", "rest", "rest"),
        TrialEvent(2, "T1", "left", "trial_start"),
    ]
    count = PlaybackService().run(
        loader=GenericCSVLoader(),
        file_path=generic_sample,
        sink=sink,
        clock=InstantClock(),
        notifier=CapturingNotifier(),
        config=PlaybackConfig(notify_finish=False, progress_every=0),
        events=events,
        marker_sink=markers,
    )
    assert count == 3
    assert markers.opened and markers.closed
    assert [e.label for e, _ in markers.events] == ["rest", "left"]
