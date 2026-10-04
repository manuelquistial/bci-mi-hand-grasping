import { describe, expect, it } from "vitest";

import {
  createInitialPipelineState,
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
