"""SimOpenBCI: stream EEG datasets to Lab Streaming Layer."""
from .config import PlaybackConfig
from .playback import PlaybackService

__all__ = ["PlaybackConfig", "PlaybackService"]
