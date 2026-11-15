import { postJson, type PipelineSettings } from "./http";

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
