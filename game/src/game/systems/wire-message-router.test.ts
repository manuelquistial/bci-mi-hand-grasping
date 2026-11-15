import { describe, expect, it } from "vitest";

import { defaultSessionSettings } from "@/types/settings";

import { ExternalTrialClock, type ExternalTrialHost } from "./external-trial-clock";
import { OnlineMotionController } from "./online-motion-controller";
import { WireMessageRouter } from "./wire-message-router";

describe("WireMessageRouter", () => {
  it("routes trial_start to the clock and control to motion", () => {
    const begun: string[] = [];
    const host: ExternalTrialHost = {
      enabled: () => true,
      phase: () => "idle",
      collected: () => false,
      settings: () => defaultSessionSettings(),
      begin: (t) => begun.push(String(t.index)),
      end: () => undefined,
    };
    const motion = new OnlineMotionController();
    const router = new WireMessageRouter({
      motion,
      clock: new ExternalTrialClock(),
      host,
      collected: () => false,
      debug: () => false,
    });
    router.handle({
      kind: "v2_trial",
      event: { version: 2, type: "trial_start", trialId: "trial_3", cue: "left", issuedAt: 0 },
    });
    expect(begun).toEqual(["3"]);
    router.handle({
      kind: "v2_control",
      command: {
        version: 2,
        type: "bci_control",
        trialId: "trial_3",
        sequence: 0,
        sourceWindowEnd: null,
        issuedAt: 0,
        control: 1,
        valid: true,
      },
    });
    const frame = motion.step(
      performance.now(),
      0.05,
      { getX: () => 0, applyHorizontalStep: () => 0, applyHorizontalVelocity: () => 5 },
      defaultSessionSettings().online,
    );
    expect(frame.controlSignal).toBeGreaterThan(0);
  });
});
