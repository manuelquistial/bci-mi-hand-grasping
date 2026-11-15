import {
  createInitialPipelineState,
  processControlCommand,
  processControlSample,
  type ControlPipelineState,
} from "@/game/systems/control-pipeline";
import {
  DISCRETE_STEP_PX,
  discreteStepDirection,
  motionModeForStrategy,
} from "@/game/systems/control-strategy-motion";
import {
  commandV2ToSample,
  type BCIControlCommandV2,
  type BCITrialEventV2,
} from "@/lib/bci/validate-sample";
import type { BCIControlSample } from "@/types/domain";
import type { OnlineSettings } from "@/types/settings";

type ActiveCommand = {
  control: number;
  trialId: string;
  sequence: number;
  receivedAt: number;
  valid: boolean;
  strategy?: string;
};

/** Minimal avatar surface the controller drives (implemented by the stage). */
export type AvatarMotion = {
  getX(): number;
  applyHorizontalStep(deltaX: number): number;
  applyHorizontalVelocity(velocityX: number, dt: number): number;
};

export type MotionFrame = {
  sample: BCIControlSample;
  controlSignal: number;
  smoothedSignal: number;
  velocityX: number;
  avatarX: number;
};

const neutralSample = (timestamp: number): BCIControlSample => ({
  left: 0.5,
  right: 0.5,
  timestamp,
});

/** Strategy → avatar X: owns the latest command/sample and the EMA state. */
export class OnlineMotionController {
  private pipeline: ControlPipelineState = createInitialPipelineState();
  private latestSample: BCIControlSample | null = null;
  private activeCommand: ActiveCommand | null = null;
  private activeBciTrialId: string | null = null;
  private lastSequenceByTrial = new Map<string, number>();
  private lastAppliedDiscreteSequence = -1;

  /** Drop all trial/command state (session stop). */
  reset(): void {
    this.activeCommand = null;
    this.activeBciTrialId = null;
    this.lastSequenceByTrial.clear();
    this.lastAppliedDiscreteSequence = -1;
  }

  pushSample(sample: BCIControlSample): void {
    this.latestSample = sample;
  }

  /** Bridge announced a new trial: forget the previous command stream. */
  onTrialStart(event: BCITrialEventV2): void {
    this.activeBciTrialId = event.trialId;
    this.lastSequenceByTrial.set(event.trialId, -1);
    this.activeCommand = null;
    this.pipeline = createInitialPipelineState();
  }

  /** Bridge ended the trial: invalidate its command so the avatar holds. */
  onTrialEnd(event: BCITrialEventV2): void {
    if (this.activeBciTrialId !== event.trialId) return;
    this.activeCommand = {
      control: 0,
      trialId: event.trialId,
      sequence: Number.MAX_SAFE_INTEGER,
      receivedAt: performance.now(),
      valid: false,
    };
  }

  clearActiveTrial(): void {
    this.activeBciTrialId = null;
  }

  onEnterControl(): void {
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

  /** The coin was reached: neutralise motion for the rest of the trial. */
  onCollected(): void {
    this.activeCommand = this.activeCommand
      ? { ...this.activeCommand, control: 0, valid: false }
      : null;
    this.pipeline = createInitialPipelineState();
    this.latestSample = neutralSample(performance.now());
  }

  /** Legacy protocol v1 sample. */
  acceptLegacy(message: {
    sample: BCIControlSample;
    trialId?: string;
    valid: boolean;
  }): void {
    if (!message.valid) {
      this.activeCommand = null;
      this.latestSample = neutralSample(performance.now());
      return;
    }
    if (
      message.trialId &&
      this.activeBciTrialId &&
      message.trialId !== this.activeBciTrialId
    ) {
      return;
    }
    this.latestSample = message.sample;
  }

  acceptCommand(command: BCIControlCommandV2, collected: boolean): void {
    if (this.activeBciTrialId && command.trialId !== this.activeBciTrialId) {
      return;
    }
    const lastSeq = this.lastSequenceByTrial.get(command.trialId) ?? -1;
    if (command.sequence <= lastSeq) return;
    this.lastSequenceByTrial.set(command.trialId, command.sequence);
    if (!this.activeBciTrialId) {
      this.activeBciTrialId = command.trialId;
    }
    const receivedAt = performance.now();
    // After collecting the coin, ignore further motion commands this trial.
    if (collected) {
      this.activeCommand = {
        control: 0,
        trialId: command.trialId,
        sequence: command.sequence,
        receivedAt,
        valid: false,
        strategy: command.strategy,
      };
      return;
    }
    this.activeCommand = {
      control: command.control,
      trialId: command.trialId,
      sequence: command.sequence,
      receivedAt,
      valid: command.valid,
      strategy: command.strategy,
    };
    if (command.valid) {
      this.latestSample = commandV2ToSample(command);
    } else {
      this.latestSample = neutralSample(command.issuedAt);
      this.pipeline = createInitialPipelineState();
    }
  }

  /** Advance the avatar one frame; returns the telemetry-facing values. */
  step(
    now: number,
    dt: number,
    avatar: AvatarMotion,
    online: OnlineSettings,
  ): MotionFrame {
    const cmd = this.activeCommand;
    if (cmd && now - cmd.receivedAt > online.commandExpiryMs) {
      this.activeCommand = null;
      this.pipeline = createInitialPipelineState();
    }
    const active = this.activeCommand;
    const canUseV2 =
      active !== null &&
      active.valid &&
      (!this.activeBciTrialId || active.trialId === this.activeBciTrialId);

    if (canUseV2 && active) {
      return this.stepCommand(active, dt, avatar, online);
    }
    return this.stepLegacy(now, dt, avatar, online);
  }

  private stepCommand(
    cmd: ActiveCommand,
    dt: number,
    avatar: AvatarMotion,
    online: OnlineSettings,
  ): MotionFrame {
    const sample = commandV2ToSample({
      version: 2,
      type: "bci_control",
      trialId: cmd.trialId,
      sequence: cmd.sequence,
      sourceWindowEnd: null,
      issuedAt: cmd.receivedAt,
      control: cmd.control,
      valid: true,
      strategy: cmd.strategy,
    });

    if (motionModeForStrategy(cmd.strategy) === "discrete_step") {
      // B/C: one step per new window command — not continuous full-speed hold.
      let avatarX = avatar.getX();
      if (cmd.sequence !== this.lastAppliedDiscreteSequence) {
        const dir = discreteStepDirection(cmd.control);
        if (dir !== 0) {
          avatarX = avatar.applyHorizontalStep(dir * DISCRETE_STEP_PX);
        }
        this.lastAppliedDiscreteSequence = cmd.sequence;
      }
      return {
        sample,
        controlSignal: cmd.control,
        smoothedSignal: 0,
        velocityX: 0,
        avatarX,
      };
    }

    // A/D: proportional velocity from control ∈ [-1, 1].
    const temporal = cmd.strategy === "temporal_proportional";
    const processed = processControlCommand(
      cmd.control,
      this.pipeline,
      online,
      { applySmoothing: !(temporal && online.skipPhaserSmoothingForTemporal) },
    );
    this.pipeline = processed.next;
    return {
      sample,
      controlSignal: processed.controlSignal,
      smoothedSignal: processed.smoothedSignal,
      velocityX: processed.velocityX,
      avatarX: avatar.applyHorizontalVelocity(processed.velocityX, dt),
    };
  }

  private stepLegacy(
    now: number,
    dt: number,
    avatar: AvatarMotion,
    online: OnlineSettings,
  ): MotionFrame {
    // Legacy / invalid: do not keep drifting on stale bias.
    if (this.activeCommand?.valid === false || !this.latestSample) {
      this.pipeline = createInitialPipelineState();
      return {
        sample: neutralSample(now),
        controlSignal: 0,
        smoothedSignal: 0,
        velocityX: 0,
        avatarX: avatar.getX(),
      };
    }
    const sample = this.latestSample;
    const processed = processControlSample(sample, this.pipeline, online);
    this.pipeline = processed.next;
    return {
      sample,
      controlSignal: processed.controlSignal,
      smoothedSignal: processed.smoothedSignal,
      velocityX: processed.velocityX,
      avatarX: avatar.applyHorizontalVelocity(processed.velocityX, dt),
    };
  }
}
