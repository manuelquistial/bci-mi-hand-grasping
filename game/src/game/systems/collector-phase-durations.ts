import type { TrialDefinition, TrialPhase } from "@/types/domain";

/** Fixed lab-style preparing gap before fixation (not in SessionSettings). */
export const COLLECTOR_PREPARING_MS = 280;

export function collectorPhaseDurationMs(
  trial: TrialDefinition,
  phase: TrialPhase,
): number {
  switch (phase) {
    case "preparing":
      return COLLECTOR_PREPARING_MS;
    case "fixation":
      return trial.fixationDurationMs;
    case "cue":
      return trial.cueDurationMs;
    case "control":
      return trial.controlDurationMs;
    case "feedback":
      return trial.feedbackDurationMs;
    case "inter_trial_interval":
      return trial.itiDurationMs;
    default:
      return 0;
  }
}
