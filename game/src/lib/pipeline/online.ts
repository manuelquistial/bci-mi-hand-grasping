import { postJson, type PipelineSettings } from "./http";

export function pipelineOnlineStart(
  settings: PipelineSettings,
): Promise<Record<string, unknown>> {
  return postJson(settings.baseUrl, "/v1/online/start", {
    source: settings.onlineSource,
    dataset: settings.dataset,
    session_dir: settings.sessionDir,
    subject: settings.subject || undefined,
    registry: settings.registry || undefined,
    control_ws_port: settings.controlWsPort || 8090,
    control_strategy: settings.controlStrategy || "baseline_proba",
    control_window_k: settings.controlWindowK || 3,
    control_proportional_scale: settings.controlProportionalScale || 1.0,
  });
}

export function pipelineOnlineStop(
  settings: PipelineSettings,
): Promise<Record<string, unknown>> {
  return postJson(settings.baseUrl, "/v1/online/stop", {});
}
