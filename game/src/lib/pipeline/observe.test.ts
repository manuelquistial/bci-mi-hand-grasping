import { describe, expect, it } from "vitest";

import {
  observeWsUrlFromControl,
  parseObserveMessage,
} from "./observe";

describe("observe helpers", () => {
  it("derives observe WS from control WS port + 1", () => {
    expect(observeWsUrlFromControl("ws://127.0.0.1:8090")).toBe(
      "ws://127.0.0.1:8091",
    );
    expect(observeWsUrlFromControl(null)).toBe("ws://127.0.0.1:8091");
  });

  it("parses pipeline_stage messages", () => {
    const message = parseObserveMessage({
      version: 1,
      type: "pipeline_stage",
      trial_id: "trial_001",
      cue: "right",
      window_index: 1,
      window_start: 2,
      window_end: 4,
      channels: ["Ch1", "Ch2"],
      raw: [[1, 2], [3, 4]],
      filtered: [[0.5, 1], [1.5, 2]],
      aligned: [[0.1], [0.2]],
      csp_traces: [[0.3]],
      csp_channels: ["CSP1"],
      features: [0.1, 0.2],
      p_left: 0.2,
      p_right: 0.8,
      label: "right",
      margin: -0.5,
      control: 0.6,
      valid: true,
      strategy: "baseline_proba",
      gate: { command: "right", stable: true },
      emg_rejected: false,
      reason: "classified",
      classify_ms: 2.1,
    });
    expect(message?.type).toBe("pipeline_stage");
    if (message?.type === "pipeline_stage") {
      expect(message.trial_id).toBe("trial_001");
      expect(message.p_right).toBe(0.8);
      expect(message.features).toEqual([0.1, 0.2]);
    }
  });

  it("parses stream_frame messages with timeline cues", () => {
    const message = parseObserveMessage({
      version: 1,
      type: "stream_frame",
      channels: ["Cz"],
      sample_rate_hz: 250,
      t_start: 10,
      t_end: 12,
      imagery_onset: 11,
      markers: [
        { t: 11, kind: "trial_start", label: "left", role: "imagery_onset" },
      ],
      windows: [{ index: 0, start: 11, end: 13 }],
      raw: [[1, 2, 3]],
      filtered: [[0, 1, 2]],
    });
    expect(message?.type).toBe("stream_frame");
    if (message?.type === "stream_frame") {
      expect(message.t_start).toBe(10);
      expect(message.imagery_onset).toBe(11);
      expect(message.markers[0]?.role).toBe("imagery_onset");
      expect(message.windows[0]?.index).toBe(0);
    }
  });

  it("rejects unknown types", () => {
    expect(parseObserveMessage({ type: "bci_control" })).toBeNull();
  });
});
