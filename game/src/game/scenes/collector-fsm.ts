import { collectorPhaseDurationMs } from "@/game/systems/collector-phase-durations";
import { phaseElapsed } from "@/game/systems/trial-fsm";
import type { TrialDefinition, TrialPhase } from "@/types/domain";

const ORDER_BEFORE_CONTROL: TrialPhase[] = ["preparing", "fixation", "cue"];

export type CollectorFsmState = {
  phase: TrialPhase;
  phaseStartedAt: number;
};

export type CollectorFsmAction =
  | { type: "wait" }
  | { type: "enter"; phase: TrialPhase }
  | { type: "timeout" }
  | { type: "next-trial" };

export function nextCollectorAction(
  fsm: CollectorFsmState,
  now: number,
  trial: TrialDefinition,
  collected: boolean,
): CollectorFsmAction {
  const phase = fsm.phase;

  if (
    phase !== "control" &&
    phase !== "idle" &&
    phase !== "finished" &&
    phaseElapsed(fsm, now) < collectorPhaseDurationMs(trial, phase)
  ) {
    return { type: "wait" };
  }

  if (phase === "preparing" || phase === "fixation") {
    const idx = ORDER_BEFORE_CONTROL.indexOf(phase);
    const next = ORDER_BEFORE_CONTROL[idx + 1];
    if (next) return { type: "enter", phase: next };
    return { type: "wait" };
  }

  if (phase === "cue") {
    return { type: "enter", phase: "control" };
  }

  if (phase === "control") {
    if (collected) return { type: "enter", phase: "feedback" };
    if (phaseElapsed(fsm, now) >= collectorPhaseDurationMs(trial, "control")) {
      return { type: "timeout" };
    }
    return { type: "wait" };
  }

  if (phase === "feedback") {
    return { type: "enter", phase: "inter_trial_interval" };
  }

  if (phase === "inter_trial_interval") {
    return { type: "next-trial" };
  }

  return { type: "wait" };
}
