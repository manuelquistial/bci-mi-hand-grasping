import { describe, expect, it } from "vitest";

import { scoreForCollection, successRate } from "./scoring";

describe("scoring", () => {
  it("rewards success", () => {
    const r = scoreForCollection(true, false);
    expect(r.outcome).toBe("success");
    expect(r.delta).toBe(1);
  });

  it("successRate", () => {
    expect(successRate(3, 10)).toBe(0.3);
  });
});
