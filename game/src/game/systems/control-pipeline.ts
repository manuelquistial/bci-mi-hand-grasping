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

/** controlSignal = right - left, then dead zone, gain, EMA smoothing, clamp [-1,1]. */
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
  const rawSignal = sample.right - sample.left;
  let controlSignal = rawSignal;
  if (Math.abs(controlSignal) <= settings.deadZone) {
    controlSignal = 0;
  }
  controlSignal *= settings.gain;
  const alpha = Math.min(1, Math.max(0, settings.smoothing));
  const smoothedSignal =
    alpha * controlSignal + (1 - alpha) * prev.smoothedSignal;
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
