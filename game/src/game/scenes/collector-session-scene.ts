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

  protected abstract readonly sessionMode: SessionMode;
  protected abstract readonly bannerLabel: string;

  protected onSessionCreate(): void {}
  protected onSessionStop(): void {}
  protected onEnterControl(_trial: TrialDefinition): void {}
  protected onControlFrame(_now: number, _trial: TrialDefinition): void {}
  protected handleExtraCommand(_cmd: GameCommand): void {}

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
    this.startNextTrial();
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
    if (this.fsm.phase === "finished" || this.trialIndex < 0) {
      return;
    }
    const now = this.time.now;
    const trial = this.trials[this.trialIndex];
    if (!trial) return;

    if (this.telemetry.shouldFlush()) {
      this.flushTelemetry();
    }

    if (this.fsm.phase === "control") {
      this.onControlFrame(now, trial);
    }

    this.stepFsm(now, trial);
  }

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

  private finishSession(): void {
    this.stage.stopAutoMove();
    this.onSessionStop();
    this.fsm = transitionTo("finished", this.time.now);
    this.flushTelemetry();
    this.bridge.emit({
      type: "SESSION_COMPLETED",
      sessionId: this.sessionId,
      totalScore: this.score,
      successCount: this.successCount,
      trialCount: this.trials.length,
    });
    this.scene.start("BootScene");
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
      this.enterPhase("feedback", now, trial);
      return;
    }
    this.enterPhase(action.phase, now, trial);
  }

  private enterPhase(phase: TrialPhase, now: number, trial: TrialDefinition): void {
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

  private applyOutcome(
    trial: TrialDefinition,
    outcome: TrialOutcome,
    delta: number,
  ): void {
    this.lastOutcome = outcome;
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
