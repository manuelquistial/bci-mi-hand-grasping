import { describe, expect, it } from "vitest";

import {
  commandV2ToSample,
  parseBCIControlSample,
  parseBCIWireMessage,
} from "./validate-sample";

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

describe("parseBCIWireMessage v2", () => {
  it("accepts a valid control command", () => {
    const r = parseBCIWireMessage({
      version: 2,
      type: "bci_control",
      trial_id: "trial_001",
      sequence: 3,
      source_window_end: 10.5,
      issued_at: 11.0,
      control: 0.65,
      valid: true,
    });
    expect(r.ok).toBe(true);
    if (r.ok && r.value.kind === "v2_control") {
      expect(r.value.command.control).toBe(0.65);
      expect(r.value.command.trialId).toBe("trial_001");
    }
  });

  it("rejects control outside [-1,1]", () => {
    expect(
      parseBCIWireMessage({
        version: 2,
        type: "bci_control",
        trial_id: "t",
        sequence: 1,
        source_window_end: null,
        issued_at: 1,
        control: 1.5,
        valid: true,
      }).ok,
    ).toBe(false);
  });

  it("accepts trial_start / trial_end", () => {
    const start = parseBCIWireMessage({
      version: 2,
      type: "trial_start",
      trial_id: "trial_000",
      cue: "left",
      issued_at: 1,
    });
    expect(start.ok).toBe(true);
    const end = parseBCIWireMessage({
      version: 2,
      type: "trial_end",
      trial_id: "trial_000",
      cue: "left",
      fused_label: "left",
      correct: true,
      issued_at: 2,
    });
    expect(end.ok).toBe(true);
    if (end.ok && end.value.kind === "v2_trial") {
      expect(end.value.event.fusedLabel).toBe("left");
      expect(end.value.event.correct).toBe(true);
    }
  });

  it("maps control to left/right sample", () => {
    const sample = commandV2ToSample({
      version: 2,
      type: "bci_control",
      trialId: "t",
      sequence: 1,
      sourceWindowEnd: null,
      issuedAt: 5,
      control: 1,
      valid: true,
    });
    expect(sample.right).toBeCloseTo(1);
    expect(sample.left).toBeCloseTo(0);
  });
});
