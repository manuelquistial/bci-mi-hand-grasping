"""Frozen marker wire contract (kind:code:label) for LSL Markers stream."""

from __future__ import annotations

from dataset_loaders.events import TrialEvent


def test_marker_string_is_kind_code_label() -> None:
    event = TrialEvent(sample_index=10, code="2", label="left", kind="trial_start")
    assert event.marker_string == "trial_start:2:left"


def test_shuqfa_style_marker_string() -> None:
    event = TrialEvent(sample_index=0, code="T1", label="right", kind="trial_start")
    assert event.marker_string == "trial_start:T1:right"


def test_rest_marker_format() -> None:
    event = TrialEvent(sample_index=0, code="T0", label="rest", kind="rest")
    assert event.marker_string == "rest:T0:rest"


def test_markers_contract_module_matches_event_string() -> None:
    from simopenbci.markers_contract import format_marker, marker_from_event, parse_marker

    event = TrialEvent(sample_index=1, code="T1", label="a:b", kind="trial_start")
    assert marker_from_event(event) == event.marker_string == "trial_start:T1:a:b"
    assert format_marker("rest", "T0", "rest") == "rest:T0:rest"
    assert parse_marker(event.marker_string) == ("trial_start", "T1", "a:b")
