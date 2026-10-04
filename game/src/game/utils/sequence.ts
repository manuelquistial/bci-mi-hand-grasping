import type { SessionMode, TargetSide, TrialDefinition, TrialLabel } from "@/types/domain";
import type { SessionSettings } from "@/types/settings";

function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pickSide(rng: () => number): TargetSide {
  return rng() < 0.5 ? "left" : "right";
}

export function buildTrialDurations(settings: SessionSettings): Pick<
  TrialDefinition,
  | "fixationDurationMs"
  | "cueDurationMs"
  | "controlDurationMs"
  | "feedbackDurationMs"
  | "itiDurationMs"
> {
  return {
    fixationDurationMs: settings.offline.fixationDurationMs,
    cueDurationMs: settings.offline.cueDurationMs,
    controlDurationMs: settings.offline.controlDurationMs,
    feedbackDurationMs: settings.offline.feedbackDurationMs,
    itiDurationMs: settings.offline.itiDurationMs,
  };
}

export function generateTrialSequence(
  settings: SessionSettings,
  mode: SessionMode,
): TrialDefinition[] {
  const n = Math.max(1, settings.experimental.numberOfTrials);
  const blockSize = Math.max(1, settings.experimental.blockSize);
  const seed =
    settings.experimental.randomSeed ?? Date.now();
  const rng = mulberry32(seed);

  const labels: TrialLabel[] = [];
  const seqMode = settings.experimental.sequenceMode;

  if (seqMode === "random_balanced") {
    const half = Math.floor(n / 2);
    for (let i = 0; i < half; i++) labels.push("left");
    for (let i = 0; i < half; i++) labels.push("right");
    if (n % 2 === 1) labels.push(pickSide(rng) === "left" ? "left" : "right");
    for (let i = labels.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [labels[i], labels[j]] = [labels[j], labels[i]];
    }
  } else if (seqMode === "random") {
    for (let i = 0; i < n; i++) labels.push(pickSide(rng) === "left" ? "left" : "right");
  } else {
    for (let i = 0; i < n; i++) labels.push(pickSide(rng) === "left" ? "left" : "right");
  }

  const durations = buildTrialDurations(settings);
  const trials: TrialDefinition[] = [];

  for (let i = 0; i < n; i++) {
    const label = labels[i] ?? "left";
    const targetSide: TargetSide = label;
    trials.push({
      index: i,
      blockIndex: Math.floor(i / blockSize),
      mode,
      label,
      targetSide,
      ...durations,
    });
  }

  return trials;
}
