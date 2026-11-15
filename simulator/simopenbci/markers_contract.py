"""Marker text wire contract for the LSL ``*_Markers`` stream.

Every marker is one string channel sample with this shape::

    <kind>:<code>:<label>

- ``kind``: event class, e.g. ``trial_start`` or ``rest``.
- ``code``: dataset-native code, e.g. ``2``, ``T1``, ``769``.
- ``label``: short display label, e.g. ``left``, ``right``, ``rest``. It may
  itself contain ``:``; everything after the second separator is the label.

The producer is ``dataset_loaders.events.TrialEvent.marker_string``. The
consumer is ``pipeline.live.parse_marker_text``, which reads ``kind`` and
``label`` and ignores ``code``. Examples::

    trial_start:2:left
    trial_start:T1:right
    rest:T0:rest

This module only documents and mirrors that contract. It adds no new behavior;
tests in ``tests/test_marker_contract.py`` freeze the producer side.
"""
from __future__ import annotations

from dataset_loaders.events import TrialEvent

MARKER_SEPARATOR = ":"


def format_marker(kind: str, code: str, label: str) -> str:
    """Build the marker payload; identical to ``TrialEvent.marker_string``."""
    return f"{kind}{MARKER_SEPARATOR}{code}{MARKER_SEPARATOR}{label}"


def marker_from_event(event: TrialEvent) -> str:
    """Marker payload for a trial event."""
    return event.marker_string


def parse_marker(raw: str) -> tuple[str, str, str]:
    """Split ``kind:code:label`` into ``(kind, code, label)``.

    The label keeps any extra ``:``. Missing parts come back as empty strings.
    The pipeline's ``parse_marker_text`` returns only ``(kind, label)``.
    """
    parts = (raw or "").strip().split(MARKER_SEPARATOR, 2)
    parts += [""] * (3 - len(parts))
    return parts[0], parts[1], parts[2]
