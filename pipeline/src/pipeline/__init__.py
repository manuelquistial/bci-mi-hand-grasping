"""EEG / BCI pipeline — LSL ingestion and live visualization."""

from pipeline.cli import main
from pipeline.models import SampleChunk, StreamSummary
from pipeline.reader import LslStreamReader, list_streams

__all__ = [
    "LslStreamReader",
    "SampleChunk",
    "StreamSummary",
    "list_streams",
    "main",
]
