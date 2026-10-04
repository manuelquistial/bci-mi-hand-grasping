import type { BCIControlSample } from "@/types/domain";

/**
 * Rejects non-finite values and out-of-range class probabilities.
 * `left` and `right` are expected in [0, 1] (motor-imagery / classifier style).
 */
export function parseBCIControlSample(
  raw: unknown,
): { ok: true; value: BCIControlSample } | { ok: false; error: string } {
  if (raw === null || typeof raw !== "object") {
    return { ok: false, error: "Sample must be an object" };
  }
  const o = raw as Record<string, unknown>;
  const left = o.left;
  const right = o.right;
  const ts = o.timestamp;
  if (typeof left !== "number" || !Number.isFinite(left)) {
    return { ok: false, error: "left must be a finite number" };
  }
  if (typeof right !== "number" || !Number.isFinite(right)) {
    return { ok: false, error: "right must be a finite number" };
  }
  if (typeof ts !== "number" || !Number.isFinite(ts)) {
    return { ok: false, error: "timestamp must be a finite number" };
  }
  if (left < 0 || left > 1 || right < 0 || right > 1) {
    return {
      ok: false,
      error: "left and right must be in [0, 1] (class confidences / probabilities)",
    };
  }
  return { ok: true, value: { left, right, timestamp: ts } };
}
