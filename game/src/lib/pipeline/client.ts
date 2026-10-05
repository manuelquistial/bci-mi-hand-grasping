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

export type SimulatorDataset = {
  id: string;
  label: string;
  dataset: string;
  sessionDir: string;
};

export async function pipelineListDatasets(baseUrl: string): Promise<SimulatorDataset[]> {
  const url = `${baseUrl.replace(/\/$/, "")}/v1/datasets`;
  let response: Response;
  try {
    response = await fetch(url);
  } catch {
    throw new Error(
      `Pipeline is not reachable at ${baseUrl}. Start it with \`uv run pipeline serve\` inside pipeline/.`,
    );
  }
  const text = await response.text();
  const payload = text
    ? (JSON.parse(text) as {
        error?: string;
        datasets?: Array<Partial<SimulatorDataset> & { id: string; label: string }>;
      })
    : {};
  if (!response.ok) {
    throw new Error(payload.error || `Pipeline responded ${response.status}`);
  }
  return (payload.datasets ?? []).map((item) => ({
    id: item.id,
    label: item.label,
    dataset: item.dataset || item.id,
    sessionDir: item.sessionDir ?? "",
  }));
}

export type SimulatorSubject = {
  id: string;
  label: string;
};

export async function pipelineListSubjects(
  baseUrl: string,
  dataset: string,
): Promise<SimulatorSubject[]> {
  const url = `${baseUrl.replace(/\/$/, "")}/v1/subjects?dataset=${encodeURIComponent(dataset)}`;
  let response: Response;
  try {
    response = await fetch(url);
  } catch {
    throw new Error(
      `Pipeline is not reachable at ${baseUrl}. Start it with \`uv run pipeline serve\` inside pipeline/.`,
    );
  }
  const text = await response.text();
  const payload = text
    ? (JSON.parse(text) as {
        error?: string;
        subjects?: Array<Partial<SimulatorSubject> & { id: string }>;
      })
    : {};
  if (!response.ok) {
    throw new Error(payload.error || `Pipeline responded ${response.status}`);
  }
  return (payload.subjects ?? []).map((item) => ({
    id: item.id,
    label: item.label || item.id,
  }));
}

export function pipelineOnlineStart(
  settings: PipelineSettings,
): Promise<Record<string, unknown>> {
  return postJson(settings.baseUrl, "/v1/online/start", {
    source: settings.onlineSource,
    dataset: settings.dataset,
    session_dir: settings.sessionDir,
    ...(settings.subject ? { subject: settings.subject } : {}),
  });
}

export type SignalMarker = {
  kind: string;
  label: string;
  position: number;
};

export type PipelineSignal = {
  active: boolean;
  channels: string[];
  sampleRateHz: number;
  samples: number[][];
  markers: SignalMarker[];
};

export async function pipelineSignal(baseUrl: string): Promise<PipelineSignal> {
  const url = `${baseUrl.replace(/\/$/, "")}/v1/signal`;
  let response: Response;
  try {
    response = await fetch(url);
  } catch {
    throw new Error(
      `Pipeline is not reachable at ${baseUrl}. Start it with \`uv run pipeline serve\` inside pipeline/.`,
    );
  }
  const text = await response.text();
  const payload = text
    ? (JSON.parse(text) as {
        error?: string;
        active?: boolean;
        channels?: string[];
        sample_rate_hz?: number;
        samples?: number[][];
        markers?: SignalMarker[];
      })
    : {};
  if (!response.ok) {
    throw new Error(payload.error || `Pipeline responded ${response.status}`);
  }
  return {
    active: Boolean(payload.active),
    channels: payload.channels ?? [],
    sampleRateHz: Number(payload.sample_rate_hz) || 0,
    samples: payload.samples ?? [],
    markers: payload.markers ?? [],
  };
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
