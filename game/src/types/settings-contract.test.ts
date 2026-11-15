import { describe, expect, it } from "vitest";

import { defaultSessionSettings } from "./settings";

describe("session settings contract (WP0)", () => {
  it("keeps the top-level settings contexts", () => {
    const settings = defaultSessionSettings();
    expect(Object.keys(settings).sort()).toEqual(
      [
        "experimental",
        "general",
        "offline",
        "online",
        "pipeline",
        "preModel",
      ].sort(),
    );
  });

  it("exposes pipeline control ports used by monitor/open-monitor", () => {
    const { pipeline, online } = defaultSessionSettings();
    expect(pipeline.baseUrl).toBe("http://127.0.0.1:8765");
    expect(pipeline.controlWsPort).toBe(8090);
    expect(online.websocketUrl).toBe("ws://127.0.0.1:8090");
  });

  it("lists the four control strategies", () => {
    const strategies = [
      "baseline_proba",
      "discrete",
      "temporal_discrete",
      "temporal_proportional",
    ] as const;
    expect(strategies).toContain(
      defaultSessionSettings().pipeline.controlStrategy,
    );
  });
});
