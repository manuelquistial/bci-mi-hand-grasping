import type { GameBridge } from "@/game/bridge/game-bridge";
import type { TrialDefinition, TrialOutcome, TrialPhase } from "@/types/domain";

/** Bridge event emitters shared by collector scenes (no scene state). */
export function emitTrialStarted(bridge: GameBridge, trial: TrialDefinition): void {
  bridge.emit({
    type: "TRIAL_STARTED",
    trialIndex: trial.index,
    blockIndex: trial.blockIndex,
    label: trial.label,
    targetSide: trial.targetSide,
  });
}

export function emitTrialCompleted(
  bridge: GameBridge,
  trial: TrialDefinition,
  outcome: TrialOutcome,
  scoreDelta: number,
  totalScore: number,
): void {
  bridge.emit({
    type: "TRIAL_COMPLETED",
    trialIndex: trial.index,
    blockIndex: trial.blockIndex,
    label: trial.label,
    targetSide: trial.targetSide,
    outcome,
    scoreDelta,
    trialScore: totalScore,
  });
  bridge.emit({ type: "SCORE_CHANGED", totalScore });
}

export function emitPhaseChanged(
  bridge: GameBridge,
  phase: TrialPhase,
  trialIndex: number,
  blockIndex: number,
): void {
  bridge.emit({ type: "PHASE_CHANGED", phase, trialIndex, blockIndex });
}

export function emitSessionCompleted(
  bridge: GameBridge,
  sessionId: string,
  totalScore: number,
  successCount: number,
  trialCount: number,
): void {
  bridge.emit({
    type: "SESSION_COMPLETED",
    sessionId,
    totalScore,
    successCount,
    trialCount,
  });
}
