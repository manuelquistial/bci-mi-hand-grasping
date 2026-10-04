import type { SessionMode } from "@/types/domain";
import type { SessionSettings } from "@/types/settings";
import type { TelemetrySamplePayload } from "@/types/domain";

const jsonHeaders = { "Content-Type": "application/json" };

export async function apiCreateSession(mode: SessionMode, settings: SessionSettings) {
  const res = await fetch("/api/sessions", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({
      mode,
      settings: settings as unknown as Record<string, unknown>,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{
    id: string;
    mode: SessionMode;
    startedAt: string;
    settingsJson: unknown;
  }>;
}

/** `score` is the points earned on that trial (e.g. +1 / -1), not cumulative session score. */
export async function apiPostTrial(body: {
  sessionId: string;
  trialIndex: number;
  blockIndex: number;
  label: string;
  targetSide: string;
  phaseTimestampsJson?: Record<string, number>;
  outcome: string | null;
  score: number;
}) {
  const res = await fetch("/api/trials", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ id: string; trialIndex: number }>;
}

export async function apiPostTelemetry(
  sessionId: string,
  samples: TelemetrySamplePayload[],
) {
  const res = await fetch("/api/telemetry", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({
      sessionId,
      samples: samples.map((s) => ({
        trialId: s.trialId,
        timestamp: s.timestamp,
        leftValue: s.leftValue,
        rightValue: s.rightValue,
        controlSignal: s.controlSignal,
        smoothedSignal: s.smoothedSignal,
        velocityX: s.velocityX,
        avatarX: s.avatarX,
        phase: s.phase,
      })),
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ ok: boolean; count: number }>;
}

export async function apiFinalizeSession(
  sessionId: string,
  payload: { totalScore: number; successCount: number; trialCount: number },
) {
  const successRate =
    payload.trialCount > 0 ? payload.successCount / payload.trialCount : 0;
  const res = await fetch(`/api/sessions/${sessionId}`, {
    method: "PATCH",
    headers: jsonHeaders,
    body: JSON.stringify({
      endedAt: new Date().toISOString(),
      totalScore: payload.totalScore,
      successRate,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ ok: boolean }>;
}

export async function apiListSessions(limit = 20, offset = 0) {
  const res = await fetch(
    `/api/sessions?limit=${limit}&offset=${offset}`,
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{
    sessions: {
      id: string;
      mode: string;
      startedAt: string;
      endedAt: string | null;
      totalScore: number;
      successRate: number | null;
      createdAt: string;
    }[];
  }>;
}
