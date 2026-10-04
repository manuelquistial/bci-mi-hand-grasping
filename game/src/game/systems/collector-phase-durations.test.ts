import { describe, expect, it } from "vitest";

import {
  COLLECTOR_PREPARING_MS,
  collectorPhaseDurationMs,
} from "./collector-phase-durations";

const trial = {
  index: 0,
  blockIndex: 0,
  mode: "offline" as const,
  label: "left" as const,
  targetSide: "left" as const,
  fixationDurationMs: 100,
  cueDurationMs: 200,
  controlDurationMs: 300,
  feedbackDurationMs: 400,
  itiDurationMs: 50,
};

describe("collectorPhaseDurationMs", () => {
  it("uses fixed preparing duration", () => {
    expect(collectorPhaseDurationMs(trial, "preparing")).toBe(
      COLLECTOR_PREPARING_MS,
    );
  });

  it("maps trial timing fields", () => {
    expect(collectorPhaseDurationMs(trial, "fixation")).toBe(100);
    expect(collectorPhaseDurationMs(trial, "cue")).toBe(200);
    expect(collectorPhaseDurationMs(trial, "control")).toBe(300);
    expect(collectorPhaseDurationMs(trial, "feedback")).toBe(400);
    expect(collectorPhaseDurationMs(trial, "inter_trial_interval")).toBe(50);
  });

  it("returns 0 for unknown phases", () => {
    expect(collectorPhaseDurationMs(trial, "idle")).toBe(0);
    expect(collectorPhaseDurationMs(trial, "finished")).toBe(0);
  });
});
