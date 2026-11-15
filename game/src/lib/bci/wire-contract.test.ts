import { describe, expect, it } from "vitest";

import { observeWsUrlFromControl } from "@/lib/pipeline/observe";

import { parseBCIWireMessage } from "./validate-sample";

describe("wire contracts (WP0 characterization)", () => {
  it("parses control v2 bci_control messages", () => {
    const parsed = parseBCIWireMessage({
      version: 2,
      type: "bci_control",
      trial_id: "trial_000",
      sequence: 1,
      source_window_end: 2,
      issued_at: 1,
      control: -0.25,
      valid: true,
      window_index: 0,
      strategy: "baseline_proba",
      reason: "classified",
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.kind).toBe("v2_control");
    if (parsed.value.kind === "v2_control") {
      expect(parsed.value.command.trialId).toBe("trial_000");
      expect(parsed.value.command.control).toBe(-0.25);
      expect(parsed.value.command.valid).toBe(true);
    }
  });

  it("parses trial_start events", () => {
    const parsed = parseBCIWireMessage({
      version: 2,
      type: "trial_start",
      trial_id: "trial_001",
      cue: "left",
      subject_id: "S001",
      source_window_end: 1.5,
      issued_at: 1.5,
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.kind).toBe("v2_trial");
    if (parsed.value.kind === "v2_trial") {
      expect(parsed.value.event.type).toBe("trial_start");
      expect(parsed.value.event.trialId).toBe("trial_001");
      expect(parsed.value.event.cue).toBe("left");
    }
  });

  it("derives observe WS as control port + 1 (current coupling)", () => {
    expect(observeWsUrlFromControl("ws://127.0.0.1:8090")).toBe(
      "ws://127.0.0.1:8091",
    );
  });
});
