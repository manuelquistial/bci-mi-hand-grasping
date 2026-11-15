import { postJson, type PipelineSettings } from "./http";

export type PipelineMarker = { kind: string; code: string; label: string };

export function pipelineMarker(
  settings: PipelineSettings,
  marker: PipelineMarker,
): Promise<Record<string, unknown>> {
  return postJson(settings.baseUrl, "/v1/record/marker", marker);
}

export function pipelineOnlineMarker(
  settings: PipelineSettings,
  marker: PipelineMarker,
): Promise<Record<string, unknown>> {
  return postJson(settings.baseUrl, "/v1/online/marker", marker);
}

function codeForLabel(label: string): string {
  return label === "left" ? "T1" : label === "right" ? "T2" : "event";
}

export function trialMarker(label: string, kind: "trial_start" | "trial_end") {
  return { kind, code: codeForLabel(label), label };
}

export function imageryMarker(label: string, kind: "imagery_start" | "imagery_end") {
  return { kind, code: codeForLabel(label), label };
}
