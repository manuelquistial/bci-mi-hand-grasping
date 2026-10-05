"""Windows cut from one motor-imagery activity.

The activity lasts 4 s. Five overlapping 2 s windows step by 0.5 s:
0–2 s, 0.5–2.5 s, 1–3 s, 1.5–3.5 s, and 2–4 s.
"""

from __future__ import annotations

WINDOW_SECONDS = 2.0
STEP_SECONDS = 0.5
ACTIVITY_SECONDS = 4.0


def activity_windows(onset: float) -> list[tuple[float, float]]:
    """Absolute ``(start, end)`` of each window, measured from ``onset``."""
    windows: list[tuple[float, float]] = []
    offset = 0.0
    while offset + WINDOW_SECONDS <= ACTIVITY_SECONDS + 1e-9:
        start = float(onset) + offset
        windows.append((start, start + WINDOW_SECONDS))
        offset += STEP_SECONDS
    return windows


# Shuqfa task periods that are the activity. Rest codes stay out.
_SHUQFA_ACTIVITY = frozenset({"left", "right", "both_fists", "both_feet"})
_HANDS = frozenset({"left", "right"})


def is_activity_onset(dataset: str, kind: str, label: str) -> bool:
    """Whether a dataset marker is the start of motor imagery.

    Liu publishes imagery as ``trial_start`` (marker 2). Shuqfa publishes each
    task period as ``trial_start`` and rest as ``rest``. A recorded session
    uses the same ``trial_start`` hand labels it stored.
    """
    if dataset == "liu2024_stroke":
        return kind == "trial_start"
    if dataset == "shuqfa2024":
        return kind == "trial_start" and label in _SHUQFA_ACTIVITY
    if dataset == "recorded_session":
        return kind == "trial_start" and label in _HANDS
    return False
