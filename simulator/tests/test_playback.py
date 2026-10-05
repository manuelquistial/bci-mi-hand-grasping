"""Tests for PlaybackService."""
from pathlib import Path

from dataset_loaders import BCICompetitionIV2aLoader

from simopenbci.clock import InstantClock
from simopenbci.config import PlaybackConfig
from simopenbci.notifiers.console import ConsoleNotifier
from simopenbci.playback import PlaybackService
from simopenbci.sinks.recording import RecordingSink


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


def test_playback_records_all_samples(bci_sample: Path):
    sink = RecordingSink()
    notifier = CapturingNotifier()
    count = PlaybackService().run(
        loader=BCICompetitionIV2aLoader(),
        file_path=bci_sample,
        sink=sink,
        clock=InstantClock(),
        notifier=notifier,
        config=PlaybackConfig(notify_finish=True, progress_every=0),
    )
    assert count == 3
    assert len(sink.samples) == 3
    assert sink.opened and sink.closed
    assert notifier.infos
    assert "Samples sent: 3" in notifier.infos[0]


def test_playback_max_samples(bci_sample: Path):
    sink = RecordingSink()
    count = PlaybackService().run(
        loader=BCICompetitionIV2aLoader(),
        file_path=bci_sample,
        sink=sink,
        clock=InstantClock(),
        notifier=CapturingNotifier(),
        config=PlaybackConfig(max_samples=2, notify_finish=False, progress_every=0),
    )
    assert count == 2
    assert len(sink.samples) == 2


def test_playback_stream_name_override(bci_sample: Path):
    sink = RecordingSink()
    PlaybackService().run(
        loader=BCICompetitionIV2aLoader(),
        file_path=bci_sample,
        sink=sink,
        clock=InstantClock(),
        notifier=CapturingNotifier(),
        config=PlaybackConfig(stream_name="OpenBCI_EEG", notify_finish=False, progress_every=0),
    )
    assert sink.stream_name == "OpenBCI_EEG"
