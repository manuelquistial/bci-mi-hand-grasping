"""Live slices of one 4 s motor-imagery activity."""

import numpy as np

from pipeline.windows import ActivityWindower


def test_five_windows_are_cut_once_the_activity_has_arrived() -> None:
    rate = 250.0
    onset = 10.0
    count = int(rate * 4.2)
    stamps = [onset + index / rate for index in range(count)]
    samples = np.arange(count * 8, dtype=np.float64).reshape(count, 8)

    windower = ActivityWindower()
    opened = windower.begin(onset, "left", "T1")
    assert [window["samples"] for window in opened["windows"]] == [0, 0, 0, 0, 0]

    windower.push(samples, stamps)
    done = windower.latest_view()
    assert done is not None
    assert [window["samples"] for window in done["windows"]] == [500, 500, 500, 500, 500]
    assert done["windows"][0]["start"] == 10.0
    assert done["windows"][-1]["end"] == 14.0
