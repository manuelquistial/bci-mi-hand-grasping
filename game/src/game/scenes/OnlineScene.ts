import { createOnlineInputAdapter } from "@/game/adapters/online-adapter-factory";
import type { ControlAdapter } from "@/game/adapters/control-adapter";
import type { WebSocketControlAdapter } from "@/game/adapters/websocket-control-adapter";
import { CollectorSessionScene } from "@/game/scenes/collector-session-scene";
import {
  createInitialPipelineState,
  processControlCommand,
  processControlSample,
} from "@/game/systems/control-pipeline";
import {
  DISCRETE_STEP_PX,
  discreteStepDirection,
  motionModeForStrategy,
} from "@/game/systems/control-strategy-motion";
import { scoreForCollection } from "@/game/systems/scoring";
import { buildTrialDurations } from "@/game/utils/sequence";
import {
  commandV2ToSample,
  parseBCIControlSample,
  type BCIControlCommandV2,
  type BCITrialEventV2,
  type BCIWireMessage,
} from "@/lib/bci/validate-sample";
import type { GameCommand } from "@/types/bridge";
import type {
  BCIControlSample,
  TargetSide,
  TrialDefinition,
  TrialLabel,
} from "@/types/domain";

type ActiveCommand = {
  control: number;
  trialId: string;
  sequence: number;
  receivedAt: number;
  valid: boolean;
  strategy?: string;
};

function parseTrialIndex(trialId: string): number {
  const match = /^trial_(\d+)$/i.exec(trialId);
  if (!match) return -1;
  return Number.parseInt(match[1]!, 10);
}

export class OnlineScene extends CollectorSessionScene {
  constructor() {
    super("OnlineScene");
  }

  protected readonly sessionMode = "online" as const;
  protected readonly bannerLabel = "Control online";

  private pipeline = createInitialPipelineState();
  private inputAdapter: ControlAdapter | null = null;
  private latestSample: BCIControlSample | null = null;
  private activeCommand: ActiveCommand | null = null;
  private activeBciTrialId: string | null = null;
  private lastSequenceByTrial = new Map<string, number>();
  private lastAppliedDiscreteSequence = -1;
  private lastSamplePush = 0;
  private externalTrialBusy = false;
  private pendingTrialStart: BCITrialEventV2 | null = null;

  private readonly adapterCallback = (sample: BCIControlSample) => {
    this.latestSample = sample;
  };

  private readonly wireCallback = (message: BCIWireMessage) => {
    this.handleWireMessage(message);
  };

  protected usesExternalTrialClock(): boolean {
    return this.settings.pipeline.onlineSource === "simulator";
  }

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
    const ws = this.inputAdapter as WebSocketControlAdapter;
    if (typeof ws.subscribeWire === "function") {
      ws.subscribeWire(this.wireCallback);
    }
    this.inputAdapter.start();
    this.bridge.emit({
      type: "BCI_INPUT_STATUS",
      kind: this.inputAdapter.getInputKind(),
      state: "ready",
      detail: this.inputAdapter.getConnectionSummary(),
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
    this.activeCommand = null;
    this.activeBciTrialId = null;
    this.lastSequenceByTrial.clear();
    this.lastAppliedDiscreteSequence = -1;
    this.externalTrialBusy = false;
    this.pendingTrialStart = null;
  }

  protected onEnterControl(): void {
    this.pipeline = createInitialPipelineState();
    this.latestSample = null;
    this.lastAppliedDiscreteSequence = -1;
    if (
      this.activeCommand &&
      this.activeBciTrialId &&
      this.activeCommand.trialId !== this.activeBciTrialId
    ) {
      this.activeCommand = null;
    }
  }

  protected markCollected(trial: TrialDefinition): void {
    if (this.collected || this.fsm.phase !== "control") return;
    // Prediction drove the avatar to the cued coin → success.
    this.stage.snapToTarget(trial.targetSide);
    this.activeCommand = this.activeCommand
      ? { ...this.activeCommand, control: 0, valid: false }
      : null;
    this.pipeline = createInitialPipelineState();
    this.latestSample = {
      left: 0.5,
      right: 0.5,
      timestamp: performance.now(),
    };
    if (this.usesExternalTrialClock()) {
      const scored = scoreForCollection(true, false);
      this.endExternalTrial(scored.outcome, scored.delta);
      this.externalTrialBusy = false;
      return;
    }
    super.markCollected(trial);
  }

  protected onControlFrame(now: number, trial: TrialDefinition): void {
    const online = this.settings.online;

    if (this.collected || this.fsm.phase !== "control") {
      return;
    }

    const expired =
      this.activeCommand !== null &&
      now - this.activeCommand.receivedAt > online.commandExpiryMs;

    if (expired) {
      this.activeCommand = null;
      this.pipeline = createInitialPipelineState();
    }

    const canUseV2 =
      this.activeCommand !== null &&
      this.activeCommand.valid &&
      (!this.activeBciTrialId ||
        this.activeCommand.trialId === this.activeBciTrialId);

    let result = {
      controlSignal: 0,
      smoothedSignal: 0,
      velocityX: 0,
      next: this.pipeline,
    };
    let sample: BCIControlSample = {
      left: 0.5,
      right: 0.5,
      timestamp: now,
    };
    let avatarX = this.stage.player.getX();

    if (canUseV2 && this.activeCommand) {
      const strategy = this.activeCommand.strategy;
      const mode = motionModeForStrategy(strategy);
      sample = commandV2ToSample({
        version: 2,
        type: "bci_control",
        trialId: this.activeCommand.trialId,
        sequence: this.activeCommand.sequence,
        sourceWindowEnd: null,
        issuedAt: this.activeCommand.receivedAt,
        control: this.activeCommand.control,
        valid: true,
        strategy,
      });

      if (mode === "discrete_step") {
        // B/C: one step per new window command — not continuous full-speed hold.
        if (this.activeCommand.sequence !== this.lastAppliedDiscreteSequence) {
          const dir = discreteStepDirection(this.activeCommand.control);
          if (dir !== 0) {
            avatarX = this.stage.applyHorizontalStep(dir * DISCRETE_STEP_PX);
          }
          this.lastAppliedDiscreteSequence = this.activeCommand.sequence;
        }
        result = {
          controlSignal: this.activeCommand.control,
          smoothedSignal: 0,
          velocityX: 0,
          next: this.pipeline,
        };
      } else {
        // A/D: proportional velocity from control ∈ [-1, 1].
        const temporal = strategy === "temporal_proportional";
        const applySmoothing = !(
          temporal && online.skipPhaserSmoothingForTemporal
        );
        const processed = processControlCommand(
          this.activeCommand.control,
          this.pipeline,
          {
            deadZone: online.deadZone,
            gain: online.gain,
            smoothing: online.smoothing,
            maxHorizontalSpeed: online.maxHorizontalSpeed,
          },
          { applySmoothing },
        );
        result = processed;
        this.pipeline = processed.next;
        const dt = Math.min(0.05, this.game.loop.delta / 1000);
        avatarX = this.stage.applyHorizontalVelocity(processed.velocityX, dt);
      }
    } else {
      sample =
        this.latestSample ??
        ({
          left: 0.5,
          right: 0.5,
          timestamp: now,
        } satisfies BCIControlSample);
      if (!this.latestSample || this.activeCommand?.valid === false) {
        sample = { left: 0.5, right: 0.5, timestamp: now };
      }
      const processed = processControlSample(sample, this.pipeline, {
        deadZone: online.deadZone,
        gain: online.gain,
        smoothing: online.smoothing,
        maxHorizontalSpeed: online.maxHorizontalSpeed,
      });
      // Legacy / invalid: do not keep drifting on stale bias.
      if (this.activeCommand?.valid === false || !this.latestSample) {
        result = {
          controlSignal: 0,
          smoothedSignal: 0,
          velocityX: 0,
          next: createInitialPipelineState(),
        };
        this.pipeline = result.next;
      } else {
        result = processed;
        this.pipeline = processed.next;
        const dt = Math.min(0.05, this.game.loop.delta / 1000);
        avatarX = this.stage.applyHorizontalVelocity(processed.velocityX, dt);
      }
    }

    if (this.stage.isAtTarget(trial.targetSide)) {
      this.markCollected(trial);
      avatarX = this.stage.player.getX();
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

  private handleWireMessage(message: BCIWireMessage): void {
    if (message.kind === "v2_trial") {
      if (message.event.type === "trial_start") {
        this.onBridgeTrialStart(message.event);
      } else {
        this.onBridgeTrialEnd(message.event);
      }
      return;
    }

    if (message.kind === "v2_control") {
      this.acceptCommand(message.command);
      return;
    }

    if (message.kind === "v1") {
      if (!message.valid) {
        this.activeCommand = null;
        this.latestSample = { left: 0.5, right: 0.5, timestamp: performance.now() };
        return;
      }
      if (message.trialId && this.activeBciTrialId && message.trialId !== this.activeBciTrialId) {
        return;
      }
      this.latestSample = message.sample;
    }
  }

  private onBridgeTrialStart(event: BCITrialEventV2): void {
    this.activeBciTrialId = event.trialId;
    this.lastSequenceByTrial.set(event.trialId, -1);
    this.activeCommand = null;
    this.pipeline = createInitialPipelineState();
    if (this.settings.online.bciDebugLogging) {
      console.debug("[BCI] trial_start", event);
    }
    if (!this.usesExternalTrialClock()) {
      return;
    }
    if (this.fsm.phase === "finished") {
      return;
    }
    // Never cut feedback short — queue the next dataset trial.
    if (this.fsm.phase === "feedback") {
      this.pendingTrialStart = event;
      return;
    }
    if (this.externalTrialBusy && this.fsm.phase === "control") {
      return;
    }
    this.startExternalFromEvent(event);
  }

  private startExternalFromEvent(event: BCITrialEventV2): void {
    const cue = event.cue;
    if (cue !== "left" && cue !== "right") {
      return;
    }
    const index = Math.max(0, parseTrialIndex(event.trialId));
    const blockSize = Math.max(1, this.settings.experimental.blockSize);
    const durations = buildTrialDurations(this.settings);
    const label = cue as TrialLabel;
    const targetSide = cue as TargetSide;
    const trial: TrialDefinition = {
      index,
      blockIndex: Math.floor(index / blockSize),
      mode: "online",
      label,
      targetSide,
      ...durations,
      controlDurationMs: 60_000,
      metadata: { bciTrialId: event.trialId, source: "simulator" },
    };
    this.externalTrialBusy = true;
    this.pendingTrialStart = null;
    this.beginExternalTrial(trial);
  }

  private onBridgeTrialEnd(event: BCITrialEventV2): void {
    if (this.activeBciTrialId === event.trialId) {
      this.activeCommand = {
        control: 0,
        trialId: event.trialId,
        sequence: Number.MAX_SAFE_INTEGER,
        receivedAt: performance.now(),
        valid: false,
      };
    }
    if (this.settings.online.bciDebugLogging) {
      console.debug("[BCI] trial_end", event);
    }
    if (!this.usesExternalTrialClock()) {
      return;
    }
    // Already scored by reaching the coin — ignore late trial_end.
    if (this.fsm.phase !== "control" || this.collected) {
      this.activeBciTrialId = null;
      return;
    }
    if (this.activeBciTrialId !== event.trialId && this.fsm.phase !== "control") {
      return;
    }
    // Imagery ended without reaching the coin → miss.
    const scored = scoreForCollection(false, true);
    this.endExternalTrial(scored.outcome, scored.delta);
    this.externalTrialBusy = false;
    this.activeBciTrialId = null;
  }

  protected onExternalFeedbackDone(): void {
    const pending = this.pendingTrialStart;
    if (!pending) {
      // No queued trial — wait briefly for the next dataset onset, then finish.
      return;
    }
    this.pendingTrialStart = null;
    this.startExternalFromEvent(pending);
  }

  private acceptCommand(command: BCIControlCommandV2): void {
    if (this.activeBciTrialId && command.trialId !== this.activeBciTrialId) {
      return;
    }
    const lastSeq = this.lastSequenceByTrial.get(command.trialId) ?? -1;
    if (command.sequence <= lastSeq) {
      return;
    }
    this.lastSequenceByTrial.set(command.trialId, command.sequence);
    if (!this.activeBciTrialId) {
      this.activeBciTrialId = command.trialId;
    }
    // After collecting the coin, ignore further motion commands this trial.
    if (this.collected) {
      this.activeCommand = {
        control: 0,
        trialId: command.trialId,
        sequence: command.sequence,
        receivedAt: performance.now(),
        valid: false,
        strategy: command.strategy,
      };
      return;
    }
    this.activeCommand = {
      control: command.control,
      trialId: command.trialId,
      sequence: command.sequence,
      receivedAt: performance.now(),
      valid: command.valid,
      strategy: command.strategy,
    };
    if (command.valid) {
      this.latestSample = commandV2ToSample(command);
    } else {
      this.latestSample = {
        left: 0.5,
        right: 0.5,
        timestamp: command.issuedAt,
      };
      this.pipeline = createInitialPipelineState();
    }
  }
}
