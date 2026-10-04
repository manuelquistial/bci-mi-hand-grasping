import { describe, expect, it } from "vitest";

import { countSuccess, maxAbs, mean } from "./aggregate";

describe("aggregate", () => {
  it("mean", () => {
    expect(mean([1, 2, 3])).toBe(2);
  });

  it("maxAbs", () => {
    expect(maxAbs([-2, 1])).toBe(2);
  });

  it("countSuccess", () => {
    expect(countSuccess(["success", "failure", "success"])).toBe(2);
  });
});
