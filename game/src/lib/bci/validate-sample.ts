import type { BCIControlSample } from "@/types/domain";

/** Protocol v2 control command from the pipeline WebSocket hub. */
export type BCIControlCommandV2 = {
  version: 2;
  type: "bci_control";
  trialId: string;
  sequence: number;
  sourceWindowEnd: number | null;
  issuedAt: number;
  control: number;
  valid: boolean;
  reason?: string;
  windowIndex?: number;
  strategy?: string;
};

export type BCITrialEventV2 = {
  version: 2;
  type: "trial_start" | "trial_end";
  trialId: string;
  cue?: "left" | "right";
  /** Gate fused label at imagery end (replay-style scoring). */
  fusedLabel?: "left" | "right";
  /** fused_label == cue when both are known. */
  correct?: boolean;
  issuedAt: number;
  subjectId?: string;
};

export type BCIWireMessage =
  | { kind: "v1"; sample: BCIControlSample; trialId?: string; valid: boolean }
  | { kind: "v2_control"; command: BCIControlCommandV2 }
  | { kind: "v2_trial"; event: BCITrialEventV2 };

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

/** Parse protocol v1 or v2 WebSocket payloads. */
export function parseBCIWireMessage(
  raw: unknown,
): { ok: true; value: BCIWireMessage } | { ok: false; error: string } {
  if (raw === null || typeof raw !== "object") {
    return { ok: false, error: "Message must be an object" };
  }
  const o = raw as Record<string, unknown>;
  const version = o.version;

  if (version === 2) {
    const type = o.type;
    if (type === "trial_start" || type === "trial_end") {
      const trialId = o.trial_id;
      const issuedAt = o.issued_at;
      if (typeof trialId !== "string" || !trialId) {
        return { ok: false, error: "trial_id must be a non-empty string" };
      }
      if (typeof issuedAt !== "number" || !Number.isFinite(issuedAt)) {
        return { ok: false, error: "issued_at must be a finite number" };
      }
      const cue = o.cue;
      if (
        cue !== undefined &&
        cue !== "left" &&
        cue !== "right"
      ) {
        return { ok: false, error: "cue must be left or right when present" };
      }
      const fused = o.fused_label;
      if (
        fused !== undefined &&
        fused !== null &&
        fused !== "left" &&
        fused !== "right"
      ) {
        return { ok: false, error: "fused_label must be left or right when present" };
      }
      const correct = o.correct;
      if (correct !== undefined && correct !== null && typeof correct !== "boolean") {
        return { ok: false, error: "correct must be a boolean when present" };
      }
      return {
        ok: true,
        value: {
          kind: "v2_trial",
          event: {
            version: 2,
            type,
            trialId,
            cue: cue as "left" | "right" | undefined,
            fusedLabel:
              fused === "left" || fused === "right" ? fused : undefined,
            correct: typeof correct === "boolean" ? correct : undefined,
            issuedAt,
            subjectId:
              typeof o.subject_id === "string" ? o.subject_id : undefined,
          },
        },
      };
    }
    if (type !== "bci_control") {
      return { ok: false, error: "Unknown v2 type" };
    }
    const trialId = o.trial_id;
    const sequence = o.sequence;
    const control = o.control;
    const valid = o.valid;
    const issuedAt = o.issued_at;
    const sourceWindowEnd = o.source_window_end;
    if (typeof trialId !== "string" || !trialId) {
      return { ok: false, error: "trial_id must be a non-empty string" };
    }
    if (typeof sequence !== "number" || !Number.isFinite(sequence)) {
      return { ok: false, error: "sequence must be a finite number" };
    }
    if (typeof control !== "number" || !Number.isFinite(control)) {
      return { ok: false, error: "control must be a finite number" };
    }
    if (control < -1 || control > 1) {
      return { ok: false, error: "control must be in [-1, 1]" };
    }
    if (typeof valid !== "boolean") {
      return { ok: false, error: "valid must be a boolean" };
    }
    if (typeof issuedAt !== "number" || !Number.isFinite(issuedAt)) {
      return { ok: false, error: "issued_at must be a finite number" };
    }
    if (
      sourceWindowEnd !== null &&
      sourceWindowEnd !== undefined &&
      (typeof sourceWindowEnd !== "number" || !Number.isFinite(sourceWindowEnd))
    ) {
      return { ok: false, error: "source_window_end must be null or finite" };
    }
    return {
      ok: true,
      value: {
        kind: "v2_control",
        command: {
          version: 2,
          type: "bci_control",
          trialId,
          sequence,
          sourceWindowEnd:
            sourceWindowEnd === undefined || sourceWindowEnd === null
              ? null
              : sourceWindowEnd,
          issuedAt,
          control,
          valid,
          reason: typeof o.reason === "string" ? o.reason : undefined,
          windowIndex:
            typeof o.window_index === "number" ? o.window_index : undefined,
          strategy: typeof o.strategy === "string" ? o.strategy : undefined,
        },
      },
    };
  }

  const v1 = parseBCIControlSample(raw);
  if (!v1.ok) return v1;
  const trialId = typeof o.trial_id === "string" ? o.trial_id : undefined;
  const valid = typeof o.valid === "boolean" ? o.valid : true;
  return {
    ok: true,
    value: { kind: "v1", sample: v1.value, trialId, valid },
  };
}

/** Convert a v2 control command into the legacy left/right sample shape. */
export function commandV2ToSample(
  command: BCIControlCommandV2,
): BCIControlSample {
  const right = 0.5 + 0.5 * command.control;
  const left = 1 - right;
  return {
    left,
    right,
    timestamp: command.issuedAt,
  };
}
