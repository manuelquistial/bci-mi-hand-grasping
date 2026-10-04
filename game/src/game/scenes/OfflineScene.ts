import { CollectorSessionScene } from "@/game/scenes/collector-session-scene";
import type { TrialDefinition } from "@/types/domain";

export class OfflineScene extends CollectorSessionScene {
  constructor() {
    super("OfflineScene");
  }

  protected readonly sessionMode = "offline" as const;
  protected readonly bannerLabel = "Calibración offline";

  private lastSamplePush = 0;

  protected onEnterControl(trial: TrialDefinition): void {
    this.stage.startAutoMove(trial.label, this.settings.offline.autoMoveSpeed, () => {
      this.markCollected(trial);
    });
  }

  protected onControlFrame(now: number, trial: TrialDefinition): void {
    if (now - this.lastSamplePush < 33) return;
    this.lastSamplePush = now;
    const left = trial.label === "left" ? 1 : 0;
    const right = trial.label === "right" ? 1 : 0;
    const direction = trial.label === "left" ? -1 : 1;
    this.telemetry.push({
      sessionId: this.sessionId,
      trialId: null,
      timestamp: now,
      leftValue: left,
      rightValue: right,
      controlSignal: right - left,
      smoothedSignal: right - left,
      velocityX: direction * this.settings.offline.autoMoveSpeed,
      avatarX: this.stage.player.getX(),
      phase: this.fsm.phase,
    });
  }
}
