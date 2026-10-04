"""Unit tests for marker payload parsing used by the LSL visualizer."""

from pipeline.visualizer import _parse_marker_payload, _segments_in_window


def test_parse_marker_payload_full() -> None:
    kind, label = _parse_marker_payload("trial_start:T1:left")
    assert kind == "trial_start"
    assert label == "left"


def test_parse_marker_payload_plain() -> None:
    kind, label = _parse_marker_payload("cue")
    assert kind == "event"
    assert label == "cue"


def test_segments_include_rest_until_the_next_marker() -> None:
    markers = [
        (10.0, "rest", "rest"),
        (14.0, "trial_start", "left"),
        (18.0, "rest", "rest"),
        (22.0, "trial_start", "right"),
    ]
    segments = _segments_in_window(markers, last_eeg_ts=20.0, window_seconds=8.0)
    assert [item[4] for item in segments] == ["rest", "left", "rest"]
    rest_before, left, rest_open = segments
    assert rest_before[:3] == (-8.0, -6.0, -10.0)
    assert rest_before[5] is False
    assert left[:3] == (-6.0, -2.0, -6.0)
    assert left[5] is True
    assert rest_open[:3] == (-2.0, 0.0, -2.0)
    assert rest_open[5] is True
