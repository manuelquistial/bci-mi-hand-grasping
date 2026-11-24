import { describe, expect, it } from "vitest";

import type { PipelineStageMessage } from "./observe";
import { stitchFieldAcrossWindows, upsertStageHistory } from "./stage-history";

function stage(
  partial: Partial<PipelineStageMessage> &
    Pick<PipelineStageMessage, "trial_id" | "window_index">,
): PipelineStageMessage {
  return {
    version: 1,
    type: "pipeline_stage",
    cue: "left",
    window_start: 10 + partial.window_index * 0.5,
    window_end: 12 + partial.window_index * 0.5,
    imagery_onset: 10,
    channels: ["Cz"],
    raw: [],
    filtered: [],
    aligned: [[0, 1, 0, -1]],
    csp_traces: [[0.2, 0.4, 0.2]],
    csp_channels: ["CSP1"],
    features: [],
    predictions: [
      {
        classifier: "linear_svm",
        label: "left",
        score: 0.5,
        p_left: 0.7,
        p_right: 0.3,
      },
    ],
    p_left: 0.7,
    p_right: 0.3,
    label: "left",
    margin: 0.4,
    control: -0.2,
    valid: true,
    strategy: "baseline_proba",
    gate: {},
    emg_rejected: false,
    reason: "classified",
    classify_ms: 1,
    ...partial,
  };
}

describe("stage-history", () => {
  it("resets when the trial id changes", () => {
    const first = upsertStageHistory([], stage({ trial_id: "t1", window_index: 0 }));
    const next = upsertStageHistory(
      first,
      stage({ trial_id: "t2", window_index: 0 }),
    );
    expect(next).toHaveLength(1);
    expect(next[0]?.trial_id).toBe("t2");
  });

  it("stitches EA traces onto a 0–4 s MI axis", () => {
    const history = [
      stage({ trial_id: "t1", window_index: 0 }),
      stage({
        trial_id: "t1",
        window_index: 1,
        window_start: 10.5,
        window_end: 12.5,
        aligned: [[1, 2, 1]],
      }),
    ];
    const stitched = stitchFieldAcrossWindows(history, "aligned");
    expect(stitched?.range).toEqual({ start: 0, end: 4 });
    expect(stitched?.windows).toHaveLength(2);
    expect(stitched?.samples[0]?.some((value) => value !== 0)).toBe(true);
  });
});
