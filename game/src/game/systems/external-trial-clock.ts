import { scoreForCollection } from "@/game/systems/scoring";
import { buildTrialDurations } from "@/game/utils/sequence";
import type { BCITrialEventV2 } from "@/lib/bci/validate-sample";
import type {
  TargetSide,
  TrialDefinition,
  TrialLabel,
  TrialOutcome,
  TrialPhase,
} from "@/types/domain";
import type { SessionSettings } from "@/types/settings";

/** What the clock needs from the owning scene (all calls are synchronous). */
export type ExternalTrialHost = {
  enabled(): boolean;
  phase(): TrialPhase;
  collected(): boolean;
  settings(): SessionSettings;
  begin(trial: TrialDefinition): void;
  end(outcome: TrialOutcome, delta: number): void;
};

export function parseTrialIndex(trialId: string): number {
  const match = /^trial_(\d+)$/i.exec(trialId);
  if (!match) return -1;
  return Number.parseInt(match[1]!, 10);
}

/** After at least one scored trial, prolonged silence means the run ended. */
export function externalRunIdle(
  trials: readonly TrialDefinition[],
  idleSince: number,
  now: number,
  timeoutMs: number,
): boolean {
  const scored = trials.some((t) => t.outcome != null);
  return scored && idleSince > 0 && now - idleSince >= timeoutMs;
}

/**
 * Simulator-slave trial clock: bridge trial_start / trial_end events drive
 * the scene FSM instead of local timers.
 */
export class ExternalTrialClock {
  private busy = false;
  private pendingStart: BCITrialEventV2 | null = null;

  reset(): void {
    this.busy = false;
    this.pendingStart = null;
  }

  onTrialStart(event: BCITrialEventV2, host: ExternalTrialHost): void {
    if (!host.enabled()) return;
    const phase = host.phase();
    if (phase === "finished") return;
    // Never cut feedback short — queue the next dataset trial.
    if (phase === "feedback") {
      this.pendingStart = event;
      return;
    }
    if (this.busy && phase === "control") return;
    this.startFromEvent(event, host);
  }

  /** Imagery ended: score a miss unless the coin was already collected. */
  onTrialEnd(host: ExternalTrialHost): void {
    if (!host.enabled()) return;
    // Already scored by reaching the coin — ignore late trial_end.
    if (host.phase() !== "control" || host.collected()) return;
    const scored = scoreForCollection(false, true);
    host.end(scored.outcome, scored.delta);
    this.busy = false;
  }

  /** Prediction drove the avatar to the cued coin → success. */
  onCoinCollected(host: ExternalTrialHost): void {
    const scored = scoreForCollection(true, false);
    host.end(scored.outcome, scored.delta);
    this.busy = false;
  }

  /** After feedback: start the queued dataset trial, if any. */
  onFeedbackDone(host: ExternalTrialHost): void {
    const pending = this.pendingStart;
    if (!pending) return;
    this.pendingStart = null;
    this.startFromEvent(pending, host);
  }

  private startFromEvent(event: BCITrialEventV2, host: ExternalTrialHost): void {
    const cue = event.cue;
    if (cue !== "left" && cue !== "right") return;
    const settings = host.settings();
    const index = Math.max(0, parseTrialIndex(event.trialId));
    const blockSize = Math.max(1, settings.experimental.blockSize);
    const trial: TrialDefinition = {
      index,
      blockIndex: Math.floor(index / blockSize),
      mode: "online",
      label: cue as TrialLabel,
      targetSide: cue as TargetSide,
      ...buildTrialDurations(settings),
      controlDurationMs: 60_000,
      metadata: { bciTrialId: event.trialId, source: "simulator" },
    };
    this.busy = true;
    this.pendingStart = null;
    host.begin(trial);
  }
}
