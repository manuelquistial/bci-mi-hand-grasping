import { describe, expect, it } from "vitest";

import { parseBCIControlSample } from "./validate-sample";

describe("parseBCIControlSample", () => {
  it("accepts a valid sample", () => {
    const r = parseBCIControlSample({ left: 0.2, right: 0.8, timestamp: 12_000 });
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.value).toEqual({ left: 0.2, right: 0.8, timestamp: 12_000 });
    }
  });

  it("rejects non-numeric fields", () => {
    expect(parseBCIControlSample({ left: "0.1", right: 0.9, timestamp: 1 }).ok).toBe(
      false,
    );
    expect(parseBCIControlSample({ left: 0.1, right: 0.9, timestamp: NaN }).ok).toBe(
      false,
    );
  });

  it("rejects out-of-range probabilities", () => {
    expect(parseBCIControlSample({ left: -0.1, right: 0.5, timestamp: 0 }).ok).toBe(
      false,
    );
    expect(parseBCIControlSample({ left: 0.2, right: 1.2, timestamp: 0 }).ok).toBe(
      false,
    );
  });
});
