"""Five overlapping windows inside a 4 s motor-imagery activity."""

from pipeline.epochs import (
    ACTIVITY_SECONDS,
    STEP_SECONDS,
    WINDOW_SECONDS,
    activity_windows,
    is_activity_onset,
)


def test_activity_windows_are_two_seconds_every_half_second() -> None:
    windows = activity_windows(10.0)
    assert (WINDOW_SECONDS, STEP_SECONDS, ACTIVITY_SECONDS) == (2.0, 0.5, 4.0)
    assert windows == [
        (10.0, 12.0),
        (10.5, 12.5),
        (11.0, 13.0),
        (11.5, 13.5),
        (12.0, 14.0),
    ]


def test_each_dataset_names_its_own_activity_onset() -> None:
    assert is_activity_onset("liu2024_stroke", "trial_start", "left") is True
    assert is_activity_onset("liu2024_stroke", "instruction", "left") is False
    assert is_activity_onset("liu2024_stroke", "rest", "rest") is False
    assert is_activity_onset("shuqfa2024", "trial_start", "right") is True
    assert is_activity_onset("shuqfa2024", "rest", "rest") is False
    assert is_activity_onset("recorded_session", "trial_start", "left") is True
    assert is_activity_onset("recorded_session", "trial_end", "left") is False
