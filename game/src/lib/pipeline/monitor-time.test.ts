import { describe, expect, it } from "vitest";

import {
  formatRelativeSeconds,
  secondsAfterOnset,
  toRelativeTimeline,
} from "./monitor-time";

describe("monitor-time", () => {
  it("anchors the axis at MI onset so windows sit on +0…+4 s", () => {
    const relative = toRelativeTimeline(
      { start: 959400.75, end: 959405.25 },
      [
        {
          t: 959401,
          kind: "trial_start",
          label: "left",
          role: "imagery_onset",
        },
      ],
      [
        { index: 0, start: 959401, end: 959403 },
        { index: 4, start: 959403, end: 959405 },
      ],
      959401,
    );
    expect(relative?.imageryOnset).toBe(0);
    expect(relative?.range.start).toBeCloseTo(-0.25);
    expect(relative?.range.end).toBeCloseTo(4.25);
    expect(relative?.windows[0]).toEqual({ index: 0, start: 0, end: 2 });
    expect(relative?.windows[1]).toEqual({ index: 4, start: 2, end: 4 });
  });

  it("formats relative seconds and onset lag", () => {
    expect(formatRelativeSeconds(1.5)).toBe("+1.50 s");
    expect(secondsAfterOnset(12, 10)).toBe(2);
  });
});
