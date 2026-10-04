import { describe, expect, it } from "vitest";

import { defaultSessionSettings } from "@/types/settings";

import { generateTrialSequence } from "./sequence";

describe("generateTrialSequence", () => {
  it("produces balanced labels in random_balanced mode", () => {
    const s = defaultSessionSettings();
    s.experimental.numberOfTrials = 10;
    s.experimental.sequenceMode = "random_balanced";
    s.experimental.randomSeed = 123;
    const trials = generateTrialSequence(s, "offline");
    const left = trials.filter((t) => t.label === "left").length;
    const right = trials.filter((t) => t.label === "right").length;
    expect(left).toBe(5);
    expect(right).toBe(5);
  });
});
