import type { TrialOutcome } from "@/types/domain";

export function scoreForCollection(
  success: boolean,
  missed: boolean,
): { outcome: TrialOutcome; delta: number } {
  if (success) return { outcome: "success", delta: 1 };
  if (missed) return { outcome: "timeout", delta: -1 };
  return { outcome: "failure", delta: -1 };
}

export function aggregateSessionScore(trialScores: number[]): number {
  return trialScores.reduce((a, b) => a + b, 0);
}

export function successRate(successes: number, total: number): number {
  if (total <= 0) return 0;
  return successes / total;
}
