import { createOnlineInputAdapter } from "@/game/adapters/online-adapter-factory";
import type { ControlAdapter } from "@/game/adapters/control-adapter";
import type { WebSocketControlAdapter } from "@/game/adapters/websocket-control-adapter";
import { CollectorSessionScene } from "@/game/scenes/collector-session-scene";
import {
  ExternalTrialClock,
  type ExternalTrialHost,
} from "@/game/systems/external-trial-clock";
import {
  OnlineMotionController,
  type AvatarMotion,
  type MotionFrame,
} from "@/game/systems/online-motion-controller";
import { WireMessageRouter } from "@/game/systems/wire-message-router";
import { parseBCIControlSample, type BCIWireMessage } from "@/lib/bci/validate-sample";
import type { GameCommand } from "@/types/bridge";
import type { BCIControlSample, TrialDefinition } from "@/types/domain";

/** Online BCI scene: lifecycle + delegation to motion, clock and wire router. */
export class OnlineScene extends CollectorSessionScene {
  constructor() {
    super("OnlineScene");
  }

  protected readonly sessionMode = "online" as const;
  protected readonly bannerLabel = "Control online";

  private inputAdapter: ControlAdapter | null = null;
  private lastSamplePush = 0;
  private readonly motion = new OnlineMotionController();
  private readonly clock = new ExternalTrialClock();

  private readonly clockHost: ExternalTrialHost = {
    enabled: () => this.usesExternalTrialClock(),
    phase: () => this.fsm.phase,
    collected: () => this.collected,
    settings: () => this.settings,
    begin: (trial) => this.beginExternalTrial(trial),
    end: (outcome, delta) => this.endExternalTrial(outcome, delta),
  };

  private readonly router = new WireMessageRouter({
    motion: this.motion,
    clock: this.clock,
    host: this.clockHost,
    collected: () => this.collected,
    debug: () => this.settings.online.bciDebugLogging,
  });

  private readonly avatar: AvatarMotion = {
    getX: () => this.stage.player.getX(),
    applyHorizontalStep: (dx) => this.stage.applyHorizontalStep(dx),
    applyHorizontalVelocity: (vx, dt) => this.stage.applyHorizontalVelocity(vx, dt),
  };

  private readonly adapterCallback = (sample: BCIControlSample) => {
    this.motion.pushSample(sample);
  };

  private readonly wireCallback = (message: BCIWireMessage) => {
    this.router.handle(message);
  };

  protected usesExternalTrialClock(): boolean {
    return this.settings.pipeline.onlineSource === "simulator";
  }

  protected onSessionCreate(): void {
    const adapter = createOnlineInputAdapter(
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
    this.inputAdapter = adapter;
    adapter.subscribe(this.adapterCallback);
    const ws = adapter as WebSocketControlAdapter;
    if (typeof ws.subscribeWire === "function") {
      ws.subscribeWire(this.wireCallback);
    }
    adapter.start();
    this.bridge.emit({
      type: "BCI_INPUT_STATUS",
      kind: adapter.getInputKind(),
      state: "ready",
      detail: adapter.getConnectionSummary(),
    });
  }

  protected onSessionStop(): void {
    const ws = this.inputAdapter as WebSocketControlAdapter | null;
    if (ws && typeof ws.unsubscribeWire === "function") {
      ws.unsubscribeWire(this.wireCallback);
    }
    this.inputAdapter?.unsubscribe(this.adapterCallback);
    this.inputAdapter?.stop();
    this.inputAdapter = null;
    this.motion.reset();
    this.clock.reset();
  }

  protected onEnterControl(): void {
    this.motion.onEnterControl();
  }

  protected markCollected(trial: TrialDefinition): void {
    if (this.collected || this.fsm.phase !== "control") return;
    this.stage.snapToTarget(trial.targetSide);
    this.motion.onCollected();
    if (this.usesExternalTrialClock()) {
      this.clock.onCoinCollected(this.clockHost);
      return;
    }
    super.markCollected(trial);
  }

  protected onExternalFeedbackDone(): void {
    this.clock.onFeedbackDone(this.clockHost);
  }

  protected onControlFrame(now: number, trial: TrialDefinition): void {
    if (this.collected || this.fsm.phase !== "control") return;

    const dt = Math.min(0.05, this.game.loop.delta / 1000);
    const frame = this.motion.step(now, dt, this.avatar, this.settings.online);
    let avatarX = frame.avatarX;
    if (this.stage.isAtTarget(trial.targetSide)) {
      this.markCollected(trial);
      avatarX = this.stage.player.getX();
    }
    this.pushTelemetry(now, frame, avatarX);
  }

  protected handleExtraCommand(cmd: GameCommand): void {
    if (cmd.type !== "PUSH_BCI_SAMPLE") return;
    const parsed = parseBCIControlSample(cmd.payload);
    if (parsed.ok) {
      this.motion.pushSample(parsed.value);
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

  private pushTelemetry(now: number, frame: MotionFrame, avatarX: number): void {
    if (now - this.lastSamplePush < 16) return;
    this.lastSamplePush = now;
    this.telemetry.push({
      sessionId: this.sessionId,
      trialId: null,
      timestamp: now,
      leftValue: frame.sample.left,
      rightValue: frame.sample.right,
      controlSignal: frame.controlSignal,
      smoothedSignal: frame.smoothedSignal,
      velocityX: frame.velocityX,
      avatarX,
      phase: this.fsm.phase,
    });
  }
}
