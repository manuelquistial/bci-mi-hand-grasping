"""Live EEG window served to the game."""

import numpy as np

from pipeline.live import LiveSignal


def test_snapshot_is_empty_until_the_stream_opens() -> None:
    live = LiveSignal()
    assert live.snapshot()["active"] is False
    live.append(np.ones((2, 8)))
    assert live.snapshot()["samples"] == []


def test_snapshot_keeps_each_channel_in_order() -> None:
    live = LiveSignal()
    live.configure(8, 250.0, [f"Ch{i + 1}" for i in range(8)])
    rows = np.arange(16, dtype=np.float64).reshape(2, 8)
    live.append(rows)
    snap = live.snapshot()
    assert snap["active"] is True
    assert snap["channels"][0] == "Ch1"
    assert snap["sample_rate_hz"] == 250.0
    assert snap["samples"][0] == [0.0, 8.0]
    assert snap["samples"][7] == [7.0, 15.0]


def test_close_drops_the_window() -> None:
    live = LiveSignal()
    live.configure(1, 250.0, ["Ch1"])
    live.append(np.array([[1.0]]))
    live.close()
    assert live.snapshot() == {
        "active": False,
        "channels": [],
        "sample_rate_hz": 0.0,
        "samples": [],
        "markers": [],
    }


def test_marker_onsets_line_up_with_the_visible_samples() -> None:
    live = LiveSignal()
    live.configure(1, 250.0, ["Ch1"])
    live.append(np.array([[1.0], [2.0], [3.0]]), [10.0, 10.5, 11.0])
    live.add_marker(10.0, "rest", "rest")
    live.add_marker(10.5, "trial_start", "left")
    live.add_marker(11.0, "trial_start", "right")
    live.add_marker(9.0, "trial_start", "before")
    assert live.snapshot()["markers"] == [
        {"kind": "rest", "label": "rest", "position": 0.0},
        {"kind": "trial_start", "label": "left", "position": 0.5},
        {"kind": "trial_start", "label": "right", "position": 1.0},
    ]
