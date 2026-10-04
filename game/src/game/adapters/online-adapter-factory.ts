import { MockControlAdapter } from "@/game/adapters/mock-control-adapter";
import { WindowEventControlAdapter } from "@/game/adapters/window-event-control-adapter";
import { WebSocketControlAdapter } from "@/game/adapters/websocket-control-adapter";
import type { BciInputKind, ControlAdapter } from "@/game/adapters/control-adapter";
import type { SessionSettings } from "@/types/settings";

export type BciStatusPayload = {
  kind: BciInputKind;
  state: string;
  detail?: string;
};

export type BciStatusEmitter = (p: BciStatusPayload) => void;

export function createOnlineInputAdapter(
  settings: SessionSettings,
  emit: BciStatusEmitter,
  getBciDebug: () => boolean,
): ControlAdapter {
  const kind = settings.online.inputAdapter;
  if (kind === "window") {
    return new WindowEventControlAdapter(emit, getBciDebug);
  }
  if (kind === "websocket") {
    return new WebSocketControlAdapter(
      {
        url: settings.online.websocketUrl,
        maxReconnectAttempts: settings.online.wsMaxReconnectAttempts,
        baseDelayMs: settings.online.wsReconnectBaseMs,
        maxDelayMs: settings.online.wsReconnectMaxMs,
        getDebug: getBciDebug,
      },
      emit,
    );
  }
  return new MockControlAdapter(settings, settings.online.mockPattern);
}
