"""Marker wire format must stay aligned with simulator TrialEvent.marker_string."""

from __future__ import annotations

from pipeline.live import parse_marker_text


def test_parse_kind_code_label() -> None:
    kind, label = parse_marker_text("trial_start:2:left")
    assert kind == "trial_start"
    assert label == "left"


def test_parse_shuqfa_style() -> None:
    kind, label = parse_marker_text("trial_start:T1:right")
    assert kind == "trial_start"
    assert label == "right"


def test_parse_rest() -> None:
    kind, label = parse_marker_text("rest:T0:rest")
    assert kind == "rest"
    assert label == "rest"
