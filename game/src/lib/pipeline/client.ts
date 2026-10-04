import type { SessionSettings } from "@/types/settings";

export type PipelineSettings = SessionSettings["pipeline"];

async function postJson(
  baseUrl: string,
  path: string,
  body: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const url = `${baseUrl.replace(/\/$/, "")}${path}`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error(
      `Pipeline is not reachable at ${baseUrl}. Start it with \`uv run pipeline serve\` inside pipeline/.`,
    );
  }
  const text = await response.text();
  const payload = text ? (JSON.parse(text) as { error?: string }) : {};
  if (!response.ok) {
    throw new Error(payload.error || `Pipeline responded ${response.status}`);
  }
  return payload as Record<string, unknown>;
}

export function pipelineRecordStart(
  settings: PipelineSettings,
  sessionId: string,
): Promise<Record<string, unknown>> {
  return postJson(settings.baseUrl, "/v1/record/start", {
    session_id: sessionId,
  });
}

export function pipelineRecordStop(
  settings: PipelineSettings,
): Promise<Record<string, unknown>> {
  return postJson(settings.baseUrl, "/v1/record/stop", {});
}

export function pipelineMarker(
  settings: PipelineSettings,
  marker: { kind: string; code: string; label: string },
): Promise<Record<string, unknown>> {
  return postJson(settings.baseUrl, "/v1/record/marker", marker);
}

export function pipelineOnlineStart(
  settings: PipelineSettings,
): Promise<Record<string, unknown>> {
  return postJson(settings.baseUrl, "/v1/online/start", {
    source: settings.onlineSource,
    lsl_name: settings.onlineSource === "cyton" ? "" : settings.lslName,
    dataset: settings.dataset,
    session_dir: settings.sessionDir,
  });
}

export function pipelineOnlineStop(
  settings: PipelineSettings,
): Promise<Record<string, unknown>> {
  return postJson(settings.baseUrl, "/v1/online/stop", {});
}

export function trialMarker(label: string, kind: "trial_start" | "trial_end") {
  const code = label === "left" ? "T1" : label === "right" ? "T2" : "event";
  return { kind, code, label };
}
