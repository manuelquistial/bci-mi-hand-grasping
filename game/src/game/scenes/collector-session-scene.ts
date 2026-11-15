import Phaser from "phaser";

import type { GameBridge } from "@/game/bridge/game-bridge";
import type { CollectorStartPayload } from "@/game/scenes/collector-session-types";
import { nextCollectorAction } from "@/game/scenes/collector-fsm";
import { GAME_HEIGHT, GAME_WIDTH } from "@/game/config/constants";
import { CollectorStage } from "@/game/stage/CollectorStage";
import { scoreForCollection } from "@/game/systems/scoring";
import { transitionTo } from "@/game/systems/trial-fsm";
import { TelemetryQueue } from "@/game/telemetry/telemetry-queue";
import type { GameCommand } from "@/types/bridge";
import type {
  SessionMode,
  TrialDefinition,
  TrialOutcome,
  TrialPhase,
} from "@/types/domain";
import type { SessionSettings } from "@/types/settings";

export abstract class CollectorSessionScene extends Phaser.Scene {
  protected bridge!: GameBridge;
  protected sessionId = "";
  protected settings!: SessionSettings;
  protected trials: TrialDefinition[] = [];
  protected trialIndex = -1;
  protected score = 0;
  protected successCount = 0;
  protected collected = false;
  protected lastOutcome: TrialOutcome | null = null;
  protected fsm: { phase: TrialPhase; phaseStartedAt: number } = {
    phase: "idle",
    phaseStartedAt: 0,
  };
  protected stage!: CollectorStage;
  protected telemetry = new TelemetryQueue(120);
  /** Wall time when simulator idle-wait began (0 = not waiting). */
  protected externalIdleSince = 0;
  /** If no trial_start arrives after feedback, treat the dataset run as finished. */
  protected readonly externalIdleTimeoutMs = 12_000;

  protected abstract readonly sessionMode: SessionMode;
  protected abstract readonly bannerLabel: string;

  protected onSessionCreate(): void {}
  protected onSessionStop(): void {}
  protected onEnterControl(_trial: TrialDefinition): void {}
  protected onControlFrame(_now: number, _trial: TrialDefinition): void {}
  protected handleExtraCommand(_cmd: GameCommand): void {}
  /** When true, dataset/bridge trials drive cue + control (simulator slave). */
  protected usesExternalTrialClock(): boolean {
    return false;
  }

  create(): void {
    this.bridge = this.registry.get("bridge") as GameBridge;
    const payload = this.registry.get("activeSession") as CollectorStartPayload;
    this.sessionId = payload.sessionId;
    this.settings = payload.settings;
    this.trials = payload.trials;

    this.stage = new CollectorStage(this, GAME_WIDTH, GAME_HEIGHT, this.bannerLabel);
    this.onSessionCreate();

    this.bridge.emit({
      type: "SESSION_STARTED",
      sessionId: this.sessionId,
      mode: this.sessionMode,
    });
    this.bridge.emit({ type: "SCORE_CHANGED", totalScore: this.score });
    if (this.usesExternalTrialClock()) {
      this.enterWaitingForExternalTrial();
    } else {
      this.startNextTrial();
    }
  }

  shutdown(): void {
    this.onSessionStop();
    this.stage?.destroy();
  }

  update(): void {
    const bridge = this.registry.get("bridge") as GameBridge;
    for (const cmd of bridge.consumeCommands()) {
      this.handleCommand(cmd);
    }
    if (this.fsm.phase === "finished") {
      return;
    }
    if (!this.usesExternalTrialClock() && this.trialIndex < 0) {
      return;
    }
    const now = this.time.now;

    if (this.telemetry.shouldFlush()) {
      this.flushTelemetry();
    }

    if (this.usesExternalTrialClock()) {
      this.stepExternalClock(now);
      return;
    }

    const trial = this.trials[this.trialIndex];
    if (!trial) return;

    if (this.fsm.phase === "control") {
      this.onControlFrame(now, trial);
    }

    this.stepFsm(now, trial);
  }

  /** Idle until the pipeline publishes trial_start (simulator mode). */
  protected enterWaitingForExternalTrial(): void {
    this.stage.stopAutoMove();
    this.fsm = transitionTo("idle", this.time.now);
    // Same field as play: avatar centered, no cue yet — waiting for EEG trial.
    this.stage.showForPhase("idle", null, null);
    this.externalIdleSince = this.time.now;
    this.bridge.emit({
      type: "PHASE_CHANGED",
      phase: "idle",
      trialIndex: Math.max(0, this.trialIndex),
      blockIndex: 0,
    });
  }

  /**
   * Begin a trial driven by the BCI bridge (cue from dataset imagery).
   * Enters control immediately so the game phase overlaps EEG imagery windows.
   */
  protected beginExternalTrial(trial: TrialDefinition): void {
    while (this.trials.length <= trial.index) {
      this.trials.push(trial);
    }
    this.trials[trial.index] = trial;
    this.trialIndex = trial.index;
    this.collected = false;
    this.lastOutcome = null;
    this.externalIdleSince = 0;
    // Reset avatar to center before imagery/control overlap begins.
    this.stage.showForPhase("idle", trial.targetSide, null);
    this.bridge.emit({
      type: "TRIAL_STARTED",
      trialIndex: trial.index,
      blockIndex: trial.blockIndex,
      label: trial.label,
      targetSide: trial.targetSide,
    });
    this.enterPhase("control", this.time.now, trial);
  }

  /** Score and leave control (coin reached, or imagery ended without collect). */
  protected endExternalTrial(
    outcome: TrialOutcome,
    delta: number,
  ): void {
    const trial = this.trials[this.trialIndex];
    if (!trial || this.fsm.phase === "finished") return;
    if (this.fsm.phase === "control" || this.fsm.phase === "idle") {
      if (outcome === "success") {
        this.collected = true;
      }
      this.applyOutcome(trial, outcome, delta);
      this.enterPhase("feedback", this.time.now, trial);
    }
  }

  private stepExternalClock(now: number): void {
    if (this.fsm.phase === "idle") {
      // After at least one scored trial, silence means the dataset run ended.
      const scored = this.trials.some((t) => t.outcome != null);
      if (
        scored &&
        this.externalIdleSince > 0 &&
        now - this.externalIdleSince >= this.externalIdleTimeoutMs
      ) {
        this.finishSession();
      }
      return;
    }
    const trial = this.trials[this.trialIndex];
    if (!trial) return;

    if (this.fsm.phase === "control") {
      this.onControlFrame(now, trial);
      // Control ends only on bridge trial_end (not local timeout).
      return;
    }

    if (this.fsm.phase === "feedback") {
      const action = nextCollectorAction(this.fsm, now, trial, this.collected);
      if (action.type === "wait") return;
      if (action.type === "enter" && action.phase === "inter_trial_interval") {
        const expected = Math.max(1, this.settings.experimental.numberOfTrials);
        if (this.trialIndex + 1 >= expected) {
          this.finishSession();
          return;
        }
        this.enterWaitingForExternalTrial();
        this.onExternalFeedbackDone();
      }
    }
  }

  /** Hook: simulator may start a queued trial_start after feedback. */
  protected onExternalFeedbackDone(): void {}

  protected markCollected(trial: TrialDefinition): void {
    if (this.collected || this.fsm.phase !== "control") return;
    this.collected = true;
    const ok = trial.targetSide === trial.label;
    const scored = scoreForCollection(ok, false);
    this.applyOutcome(trial, scored.outcome, scored.delta);
  }

  protected startNextTrial(): void {
    this.trialIndex += 1;
    if (this.trialIndex >= this.trials.length) {
      this.finishSession();
      return;
    }
    const trial = this.trials[this.trialIndex]!;
    this.collected = false;
    this.lastOutcome = null;
    this.enterPhase("preparing", this.time.now, trial);
    this.bridge.emit({
      type: "TRIAL_STARTED",
      trialIndex: trial.index,
      blockIndex: trial.blockIndex,
      label: trial.label,
      targetSide: trial.targetSide,
    });
  }

  protected finishSession(): void {
    if (this.fsm.phase === "finished") return;
    this.stage.stopAutoMove();
    this.onSessionStop();
    this.fsm = transitionTo("finished", this.time.now);
    this.flushTelemetry();
    const completed = this.trials.filter((t) => t.outcome != null).length;
    this.bridge.emit({
      type: "SESSION_COMPLETED",
      sessionId: this.sessionId,
      totalScore: this.score,
      successCount: this.successCount,
      trialCount: completed > 0 ? completed : this.trials.length,
    });
    // Stay on the play field; React leaves the session view on SESSION_COMPLETED.
  }

  private stepFsm(now: number, trial: TrialDefinition): void {
    const action = nextCollectorAction(this.fsm, now, trial, this.collected);
    if (action.type === "wait") return;
    if (action.type === "next-trial") {
      this.startNextTrial();
      return;
    }
    if (action.type === "timeout") {
      const scored = scoreForCollection(false, true);
      this.applyOutcome(trial, scored.outcome, scored.delta);
      this.enterPhase(this.phaseAfterControl(), now, trial);
      return;
    }
    const phase = action.phase === "feedback" ? this.phaseAfterControl() : action.phase;
    this.enterPhase(phase, now, trial);
  }

  private phaseAfterControl(): TrialPhase {
    return this.sessionMode === "offline" ? "inter_trial_interval" : "feedback";
  }

  protected enterPhase(phase: TrialPhase, now: number, trial: TrialDefinition): void {
    if (phase !== "control") {
      this.stage.stopAutoMove();
    }
    this.fsm = transitionTo(phase, now);
    this.bridge.emit({
      type: "PHASE_CHANGED",
      phase,
      trialIndex: trial.index,
      blockIndex: trial.blockIndex,
    });
    this.stage.showForPhase(phase, trial.targetSide, this.lastOutcome);
    if (phase === "control") {
      this.onEnterControl(trial);
    }
  }

  private handleCommand(cmd: GameCommand): void {
    if (cmd.type === "STOP_SESSION") {
      this.stage.stopAutoMove();
      this.onSessionStop();
      this.flushTelemetry();
      this.bridge.emit({ type: "SESSION_STOPPED" });
      this.scene.start("BootScene");
    } else if (cmd.type === "UPDATE_SETTINGS") {
      this.settings = cmd.payload.settings;
    } else {
      this.handleExtraCommand(cmd);
    }
  }

  protected applyOutcome(
    trial: TrialDefinition,
    outcome: TrialOutcome,
    delta: number,
  ): void {
    this.lastOutcome = outcome;
    trial.outcome = outcome;
    trial.score = delta;
    this.score += delta;
    if (outcome === "success") this.successCount += 1;
    this.bridge.emit({
      type: "TRIAL_COMPLETED",
      trialIndex: trial.index,
      blockIndex: trial.blockIndex,
      label: trial.label,
      targetSide: trial.targetSide,
      outcome,
      scoreDelta: delta,
      trialScore: this.score,
    });
    this.bridge.emit({ type: "SCORE_CHANGED", totalScore: this.score });
  }

  private flushTelemetry(): void {
    const batch = this.telemetry.flush();
    if (batch.length > 0) {
      this.bridge.emit({ type: "TELEMETRY_BATCH_READY", samples: batch });
    }
  }
}
