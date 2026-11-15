import type { CollectorStartPayload } from "@/game/scenes/collector-session-types";
import { generateTrialSequence } from "@/game/utils/sequence";
import { apiCreateSession } from "@/lib/api/client";
import {
  pipelineOnlineStart,
  pipelineOnlineStop,
  pipelineRecordStart,
  pipelineRecordStop,
} from "@/lib/pipeline/client";
import { useSettingsStore } from "@/state/settings-store";
import type { SessionMode } from "@/types/domain";
import type { SessionSettings } from "@/types/settings";

export type StartedSession = {
  sessionId: string;
  payload: CollectorStartPayload;
};

/**
 * Create the persisted session, start the pipeline side (record or online),
 * and build the payload handed to Phaser. Throws on any failure; the caller
 * owns UI state and error reporting.
 */
export async function startSessionRuntime(
  mode: SessionMode,
  settings: SessionSettings,
): Promise<StartedSession> {
  const session = await apiCreateSession(mode, settings);
  if (mode === "offline") {
    await pipelineRecordStart(settings.pipeline, session.id);
  } else {
    const online = await pipelineOnlineStart(settings.pipeline);
    const wsUrl = online.control_ws_url;
    if (typeof wsUrl === "string" && wsUrl) {
      useSettingsStore.getState().setNested("online", {
        inputAdapter: "websocket",
        websocketUrl: wsUrl,
      });
    }
  }
  // Fresh settings after the WS URL update (do not reuse the stale snapshot).
  const live = useSettingsStore.getState().settings;
  const trials =
    mode === "online" && live.pipeline.onlineSource === "simulator"
      ? []
      : generateTrialSequence(live, mode);
  return {
    sessionId: session.id,
    payload: { sessionId: session.id, mode, settings: live, trials },
  };
}

/** Stop the pipeline side matching the session mode. */
export function stopSessionPipeline(
  mode: SessionMode,
  settings: SessionSettings,
): Promise<Record<string, unknown>> {
  return mode === "offline"
    ? pipelineRecordStop(settings.pipeline)
    : pipelineOnlineStop(settings.pipeline);
}
