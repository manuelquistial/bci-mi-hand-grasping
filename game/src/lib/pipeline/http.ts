import type { SessionSettings } from "@/types/settings";

export type PipelineSettings = SessionSettings["pipeline"];

function unreachableMessage(baseUrl: string): string {
  return `Pipeline is not reachable at ${baseUrl}. Start it with \`uv run pipeline serve\` inside pipeline/.`;
}

async function requestJson(
  baseUrl: string,
  path: string,
  init?: RequestInit,
): Promise<Record<string, unknown>> {
  const url = `${baseUrl.replace(/\/$/, "")}${path}`;
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new Error(unreachableMessage(baseUrl));
  }
  const text = await response.text();
  const payload = text ? (JSON.parse(text) as { error?: string }) : {};
  if (!response.ok) {
    throw new Error(payload.error || `Pipeline responded ${response.status}`);
  }
  return payload as Record<string, unknown>;
}

export function getJson(baseUrl: string, path: string): Promise<Record<string, unknown>> {
  return requestJson(baseUrl, path);
}

export function postJson(
  baseUrl: string,
  path: string,
  body: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  return requestJson(baseUrl, path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
