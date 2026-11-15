import { describe, expect, it } from "vitest";

import type { TrialDefinition, TrialPhase } from "@/types/domain";
import { defaultSessionSettings } from "@/types/settings";

import {
  ExternalTrialClock,
  parseTrialIndex,
  type ExternalTrialHost,
} from "./external-trial-clock";

function makeHost(phase: TrialPhase) {
  const calls = { begun: [] as TrialDefinition[], ended: [] as string[] };
  const state = { phase, collected: false };
  const host: ExternalTrialHost = {
    enabled: () => true,
    phase: () => state.phase,
    collected: () => state.collected,
    settings: () => defaultSessionSettings(),
    begin: (t) => calls.begun.push(t),
    end: (outcome) => calls.ended.push(outcome),
  };
  return { host, calls, state };
}

const event = (trialId: string, cue: "left" | "right") => ({
  version: 2 as const,
  type: "trial_start" as const,
  trialId,
  cue,
  issuedAt: 0,
});

describe("ExternalTrialClock", () => {
  it("parses trial indices", () => {
    expect(parseTrialIndex("trial_7")).toBe(7);
    expect(parseTrialIndex("x")).toBe(-1);
  });

  it("starts a trial from the cue and scores a miss on trial_end", () => {
    const clock = new ExternalTrialClock();
    const { host, calls, state } = makeHost("idle");
    clock.onTrialStart(event("trial_2", "left"), host);
    expect(calls.begun[0]?.targetSide).toBe("left");
    state.phase = "control";
    clock.onTrialEnd(host);
    expect(calls.ended).toEqual(["timeout"]);
  });

  it("queues a trial_start during feedback until feedback is done", () => {
    const clock = new ExternalTrialClock();
    const { host, calls } = makeHost("feedback");
    clock.onTrialStart(event("trial_1", "right"), host);
    expect(calls.begun).toHaveLength(0);
    clock.onFeedbackDone(host);
    expect(calls.begun[0]?.targetSide).toBe("right");
  });
});
