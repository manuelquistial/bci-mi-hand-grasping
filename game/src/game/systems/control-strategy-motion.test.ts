import { describe, expect, it } from "vitest";

import {
  DISCRETE_STEP_PX,
  discreteStepDirection,
  motionModeForStrategy,
} from "./control-strategy-motion";

describe("motionModeForStrategy", () => {
  it("maps A/D to proportional and B/C to discrete steps", () => {
    expect(motionModeForStrategy("baseline_proba")).toBe("proportional");
    expect(motionModeForStrategy("temporal_proportional")).toBe("proportional");
    expect(motionModeForStrategy("discrete")).toBe("discrete_step");
    expect(motionModeForStrategy("temporal_discrete")).toBe("discrete_step");
  });
});

describe("discreteStepDirection", () => {
  it("signs control for left/right steps", () => {
    expect(discreteStepDirection(-0.2)).toBe(-1);
    expect(discreteStepDirection(1)).toBe(1);
    expect(discreteStepDirection(0)).toBe(0);
    expect(DISCRETE_STEP_PX).toBeGreaterThan(0);
  });
});
