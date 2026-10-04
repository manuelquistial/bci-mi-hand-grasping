import type { TrialPhase } from "@/types/domain";

export type FsmPhaseConfig = {
  preparingMs: number;
  fixationMs: number;
  cueMs: number;
  controlMs: number;
  feedbackMs: number;
  itiMs: number;
};

export type TrialFsmState = {
  phase: TrialPhase;
  phaseStartedAt: number;
};

export function initialFsm(now: number): TrialFsmState {
  return { phase: "idle", phaseStartedAt: now };
}

export function transitionTo(
  phase: TrialPhase,
  now: number,
): TrialFsmState {
  return { phase, phaseStartedAt: now };
}

export function phaseElapsed(state: TrialFsmState, now: number): number {
  return now - state.phaseStartedAt;
}

export function nextPhaseAfter(current: TrialPhase): TrialPhase | null {
  switch (current) {
    case "idle":
      return "preparing";
    case "preparing":
      return "fixation";
    case "fixation":
      return "cue";
    case "cue":
      return "control";
    case "control":
      return "feedback";
    case "feedback":
      return "inter_trial_interval";
    case "inter_trial_interval":
      return null;
    case "finished":
      return null;
    default:
      return null;
  }
}

export function durationForPhase(
  phase: TrialPhase,
  config: FsmPhaseConfig,
): number {
  switch (phase) {
    case "preparing":
      return config.preparingMs;
    case "fixation":
      return config.fixationMs;
    case "cue":
      return config.cueMs;
    case "control":
      return config.controlMs;
    case "feedback":
      return config.feedbackMs;
    case "inter_trial_interval":
      return config.itiMs;
    default:
      return 0;
  }
}
