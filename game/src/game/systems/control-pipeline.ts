import type { BCIControlSample } from "@/types/domain";

export type OnlinePipelineSettings = {
  deadZone: number;
  gain: number;
  smoothing: number;
  maxHorizontalSpeed: number;
};

export type ControlPipelineState = {
  smoothedSignal: number;
};

export function createInitialPipelineState(): ControlPipelineState {
  return { smoothedSignal: 0 };
}

/**
 * Apply a normalized control command ∈ [-1, 1] (negative=left, positive=right).
 * Does not add an extra EMA beyond the configured Phaser-side smoothing.
 */
export function processControlCommand(
  control: number,
  prev: ControlPipelineState,
  settings: OnlinePipelineSettings,
  options?: { applySmoothing?: boolean },
): {
  rawSignal: number;
  controlSignal: number;
  smoothedSignal: number;
  normalized: number;
  velocityX: number;
  next: ControlPipelineState;
} {
  const applySmoothing = options?.applySmoothing !== false;
  const rawSignal = Number.isFinite(control) ? Math.max(-1, Math.min(1, control)) : 0;
  let controlSignal = rawSignal;
  if (Math.abs(controlSignal) <= settings.deadZone) {
    controlSignal = 0;
  }
  controlSignal *= settings.gain;
  const alpha = Math.min(1, Math.max(0, settings.smoothing));
  const smoothedSignal = applySmoothing
    ? alpha * controlSignal + (1 - alpha) * prev.smoothedSignal
    : controlSignal;
  const clamped = Math.max(-1, Math.min(1, smoothedSignal));
  const velocityX = clamped * settings.maxHorizontalSpeed;
  return {
    rawSignal,
    controlSignal,
    smoothedSignal,
    normalized: clamped,
    velocityX,
    next: { smoothedSignal },
  };
}

/** Legacy path: controlSignal = right - left, then dead zone, gain, EMA. */
export function processControlSample(
  sample: BCIControlSample,
  prev: ControlPipelineState,
  settings: OnlinePipelineSettings,
): {
  rawSignal: number;
  controlSignal: number;
  smoothedSignal: number;
  normalized: number;
  velocityX: number;
  next: ControlPipelineState;
} {
  return processControlCommand(sample.right - sample.left, prev, settings, {
    applySmoothing: true,
  });
}
