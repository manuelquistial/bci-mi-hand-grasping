"""Sample sink adapters."""
from .lsl import LslSampleSink
from .markers import LslMarkerSink
from .recording import RecordingSink

__all__ = ["LslSampleSink", "LslMarkerSink", "RecordingSink"]
