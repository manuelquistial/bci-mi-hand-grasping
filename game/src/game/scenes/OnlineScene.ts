import { createOnlineInputAdapter } from "@/game/adapters/online-adapter-factory";
import type { ControlAdapter } from "@/game/adapters/control-adapter";
import { CollectorSessionScene } from "@/game/scenes/collector-session-scene";
import {
  createInitialPipelineState,
  processControlSample,
} from "@/game/systems/control-pipeline";
import { parseBCIControlSample } from "@/lib/bci/validate-sample";
import type { GameCommand } from "@/types/bridge";
import type { BCIControlSample, TrialDefinition } from "@/types/domain";

export class OnlineScene extends CollectorSessionScene {
  constructor() {
    super("OnlineScene");
  }

  protected readonly sessionMode = "online" as const;
  protected readonly bannerLabel = "Control online";

  private pipeline = createInitialPipelineState();
  private inputAdapter: ControlAdapter | null = null;
  private latestSample: BCIControlSample | null = null;
  private lastSamplePush = 0;

  private readonly adapterCallback = (sample: BCIControlSample) => {
    this.latestSample = sample;
  };

  protected onSessionCreate(): void {
    this.inputAdapter = createOnlineInputAdapter(
      this.settings,
      (status) => {
        this.bridge.emit({
          type: "BCI_INPUT_STATUS",
          kind: status.kind,
          state: status.state,
          detail: status.detail,
        });
      },
      () => this.settings.online.bciDebugLogging,
    );
    this.inputAdapter.subscribe(this.adapterCallback);
    this.inputAdapter.start();
    this.bridge.emit({
      type: "BCI_INPUT_STATUS",
      kind: this.inputAdapter.getInputKind(),
      state: "ready",
      detail: this.inputAdapter.getConnectionSummary(),
    });
  }

  protected onSessionStop(): void {
    this.inputAdapter?.unsubscribe(this.adapterCallback);
    this.inputAdapter?.stop();
    this.inputAdapter = null;
  }

  protected onEnterControl(): void {
    this.pipeline = createInitialPipelineState();
    this.latestSample = null;
  }

  protected onControlFrame(now: number, trial: TrialDefinition): void {
    const sample =
      this.latestSample ??
      ({
        left: 0.5,
        right: 0.5,
        timestamp: now,
      } satisfies BCIControlSample);
    const online = this.settings.online;
    const result = processControlSample(sample, this.pipeline, {
      deadZone: online.deadZone,
      gain: online.gain,
      smoothing: online.smoothing,
      maxHorizontalSpeed: online.maxHorizontalSpeed,
    });
    this.pipeline = result.next;
    const dt = Math.min(0.05, this.game.loop.delta / 1000);
    const avatarX = this.stage.applyHorizontalVelocity(result.velocityX, dt);
    if (this.stage.isAtTarget(trial.targetSide)) {
      this.markCollected(trial);
    }
    if (now - this.lastSamplePush < 16) return;
    this.lastSamplePush = now;
    this.telemetry.push({
      sessionId: this.sessionId,
      trialId: null,
      timestamp: now,
      leftValue: sample.left,
      rightValue: sample.right,
      controlSignal: result.controlSignal,
      smoothedSignal: result.smoothedSignal,
      velocityX: result.velocityX,
      avatarX,
      phase: this.fsm.phase,
    });
  }

  protected handleExtraCommand(cmd: GameCommand): void {
    if (cmd.type !== "PUSH_BCI_SAMPLE") return;
    const parsed = parseBCIControlSample(cmd.payload);
    if (parsed.ok) {
      this.latestSample = parsed.value;
      return;
    }
    this.bridge.emit({
      type: "BCI_INPUT_STATUS",
      kind: "bridge",
      state: "invalid_sample",
      detail: parsed.error,
    });
    if (this.settings.online.bciDebugLogging) {
      console.warn("[BCI bridge command]", parsed.error, cmd.payload);
    }
  }
}
