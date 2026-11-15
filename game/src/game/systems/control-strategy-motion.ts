/**
 * How each control strategy should drive Phaser motion.
 *
 * Pipeline decoder → control ∈ [-1, 1]. Strategies differ in how that value
 * is produced; the game must not treat all of them as continuous velocity.
 */

export type MotionMode = "proportional" | "discrete_step";

/** ~4 steps from center to coin (≈390 px / 4). */
export const DISCRETE_STEP_PX = 96;

export function motionModeForStrategy(strategy: string | undefined): MotionMode {
  if (strategy === "discrete" || strategy === "temporal_discrete") {
    return "discrete_step";
  }
  // baseline_proba + temporal_proportional (+ unknown): continuous velocity
  return "proportional";
}

/** Sign of a discrete command: -1 left, +1 right, 0 hold. */
export function discreteStepDirection(control: number): -1 | 0 | 1 {
  if (!Number.isFinite(control) || control === 0) return 0;
  return control < 0 ? -1 : 1;
}
