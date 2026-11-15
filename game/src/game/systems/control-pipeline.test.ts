import { describe, expect, it } from "vitest";

import {
  createInitialPipelineState,
  processControlCommand,
  processControlSample,
} from "./control-pipeline";

describe("processControlSample", () => {
  it("applies dead zone and maps to velocity", () => {
    const prev = createInitialPipelineState();
    const out = processControlSample(
      { left: 0.2, right: 0.8, timestamp: 0 },
      prev,
      {
        deadZone: 0.1,
        gain: 1,
        smoothing: 1,
        maxHorizontalSpeed: 100,
      },
    );
    expect(out.velocityX).toBeGreaterThan(0);
  });

  it("zeros inside dead zone", () => {
    const prev = createInitialPipelineState();
    const out = processControlSample(
      { left: 0.49, right: 0.51, timestamp: 0 },
      prev,
      {
        deadZone: 0.2,
        gain: 1,
        smoothing: 1,
        maxHorizontalSpeed: 100,
      },
    );
    expect(out.velocityX).toBe(0);
  });
});

describe("processControlCommand", () => {
  it("maps negative control to leftward velocity", () => {
    const out = processControlCommand(
      -1,
      createInitialPipelineState(),
      {
        deadZone: 0,
        gain: 1,
        smoothing: 1,
        maxHorizontalSpeed: 200,
      },
      { applySmoothing: false },
    );
    expect(out.velocityX).toBe(-200);
  });

  it("can skip smoothing for temporal strategies", () => {
    const prev = { smoothedSignal: 0.9 };
    const out = processControlCommand(
      0,
      prev,
      {
        deadZone: 0,
        gain: 1,
        smoothing: 0.5,
        maxHorizontalSpeed: 100,
      },
      { applySmoothing: false },
    );
    expect(out.smoothedSignal).toBe(0);
  });
});
