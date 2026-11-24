/** Observability client for the online pipeline monitor (WS :8091). */

export type ObserveGate = {
  command?: string;
  stable?: boolean;
  direction?: string | null;
  valid_count?: number;
  majority_count?: number;
  consecutive?: number;
  reason?: string;
};

export type ObserveMarker = {
  t: number;
  kind: string;
  label: string;
  role: string;
};

export type ObserveWindowBand = {
  index: number;
  start: number;
  end: number;
};

export type ObserveHeadPrediction = {
  classifier: string;
  label: string | null;
  score: number | null;
  p_left: number | null;
  p_right: number | null;
};

export type PipelineStageMessage = {
  version: number;
  type: "pipeline_stage";
  trial_id: string;
  cue: string | null;
  window_index: number;
  window_start: number;
  window_end: number;
  imagery_onset: number | null;
  channels: string[];
  raw: number[][];
  filtered: number[][];
  aligned: number[][];
  csp_traces: number[][];
  csp_channels: string[];
  features: number[];
  predictions: ObserveHeadPrediction[];
  p_left: number | null;
  p_right: number | null;
  label: string | null;
  margin: number | null;
  control: number;
  valid: boolean;
  strategy: string;
  gate: ObserveGate;
  emg_rejected: boolean;
  reason: string;
  classify_ms: number;
};

export type StreamFrameMessage = {
  version: number;
  type: "stream_frame";
  channels: string[];
  sample_rate_hz: number;
  t_start: number;
  t_end: number;
  imagery_onset: number | null;
  markers: ObserveMarker[];
  windows: ObserveWindowBand[];
  raw: number[][];
  filtered: number[][];
};

export type ObserveMessage = PipelineStageMessage | StreamFrameMessage;

export type OnlineStatusSnapshot = {
  source?: string | null;
  running?: boolean;
  samples_seen?: number;
  control_samples?: number | null;
  registry?: string | null;
  control_ws_url?: string | null;
  observe_ws_url?: string | null;
  last_control_sample?: Record<string, unknown> | null;
  last_pipeline_stage?: PipelineStageMessage | null;
  last_stream_frame?: StreamFrameMessage | null;
  error?: string | null;
};

function asNumberGrid(value: unknown): number[][] {
  if (!Array.isArray(value)) return [];
  return value.map((row) =>
    Array.isArray(row) ? row.map((cell) => Number(cell) || 0) : [],
  );
}

export function parseObserveMessage(raw: unknown): ObserveMessage | null {
  if (!raw || typeof raw !== "object") return null;
  const data = raw as Record<string, unknown>;
  const type = data.type;
  if (type === "stream_frame") {
    const markers = Array.isArray(data.markers)
      ? data.markers
          .filter((item): item is Record<string, unknown> => !!item && typeof item === "object")
          .map((item) => ({
            t: Number(item.t) || 0,
            kind: String(item.kind ?? ""),
            label: String(item.label ?? ""),
            role: String(item.role ?? "other"),
          }))
      : [];
    const windows = Array.isArray(data.windows)
      ? data.windows
          .filter((item): item is Record<string, unknown> => !!item && typeof item === "object")
          .map((item) => ({
            index: Number(item.index) || 0,
            start: Number(item.start) || 0,
            end: Number(item.end) || 0,
          }))
      : [];
    return {
      version: Number(data.version) || 1,
      type: "stream_frame",
      channels: Array.isArray(data.channels)
        ? data.channels.map(String)
        : [],
      sample_rate_hz: Number(data.sample_rate_hz) || 0,
      t_start: Number(data.t_start) || 0,
      t_end: Number(data.t_end) || 0,
      imagery_onset:
        data.imagery_onset == null ? null : Number(data.imagery_onset),
      markers,
      windows,
      raw: asNumberGrid(data.raw),
      filtered: asNumberGrid(data.filtered),
    };
  }
  if (type === "pipeline_stage") {
    const gate =
      data.gate && typeof data.gate === "object"
        ? (data.gate as ObserveGate)
        : {};
    return {
      version: Number(data.version) || 1,
      type: "pipeline_stage",
      trial_id: String(data.trial_id ?? ""),
      cue: data.cue == null ? null : String(data.cue),
      window_index: Number(data.window_index) || 0,
      window_start: Number(data.window_start) || 0,
      window_end: Number(data.window_end) || 0,
      imagery_onset:
        data.imagery_onset == null ? null : Number(data.imagery_onset),
      channels: Array.isArray(data.channels)
        ? data.channels.map(String)
        : [],
      raw: asNumberGrid(data.raw),
      filtered: asNumberGrid(data.filtered),
      aligned: asNumberGrid(data.aligned),
      csp_traces: asNumberGrid(data.csp_traces),
      csp_channels: Array.isArray(data.csp_channels)
        ? data.csp_channels.map(String)
        : [],
      features: Array.isArray(data.features)
        ? data.features.map((item) => Number(item) || 0)
        : [],
      predictions: Array.isArray(data.predictions)
        ? data.predictions
            .filter(
              (item): item is Record<string, unknown> =>
                !!item && typeof item === "object",
            )
            .map((item) => ({
              classifier: String(item.classifier ?? ""),
              label: item.label == null ? null : String(item.label),
              score: item.score == null ? null : Number(item.score),
              p_left: item.p_left == null ? null : Number(item.p_left),
              p_right: item.p_right == null ? null : Number(item.p_right),
            }))
        : [],
      p_left: data.p_left == null ? null : Number(data.p_left),
      p_right: data.p_right == null ? null : Number(data.p_right),
      label: data.label == null ? null : String(data.label),
      margin: data.margin == null ? null : Number(data.margin),
      control: Number(data.control) || 0,
      valid: Boolean(data.valid),
      strategy: String(data.strategy ?? ""),
      gate,
      emg_rejected: Boolean(data.emg_rejected),
      reason: String(data.reason ?? ""),
      classify_ms: Number(data.classify_ms) || 0,
    };
  }
  return null;
}

export function observeWsUrlFromControl(
  controlWsUrl: string | null | undefined,
  fallbackPort = 8091,
): string {
  if (!controlWsUrl) return `ws://127.0.0.1:${fallbackPort}`;
  try {
    const url = new URL(controlWsUrl);
    const port = Number(url.port || "8090");
    url.port = String(port + 1);
    return url.toString().replace(/\/$/, "");
  } catch {
    return `ws://127.0.0.1:${fallbackPort}`;
  }
}

export async function fetchOnlineStatus(
  baseUrl: string,
): Promise<OnlineStatusSnapshot | null> {
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/v1/status`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`Pipeline status ${response.status}`);
  }
  const payload = (await response.json()) as {
    online?: OnlineStatusSnapshot | null;
  };
  return payload.online ?? null;
}

export type ObserveSocketHandlers = {
  onMessage: (message: ObserveMessage) => void;
  onStatus?: (status: "connecting" | "open" | "closed" | "error") => void;
};

/** Reconnecting WebSocket for pipeline_stage / stream_frame events. */
export function connectObserveSocket(
  url: string,
  handlers: ObserveSocketHandlers,
): { close: () => void } {
  let socket: WebSocket | null = null;
  let closed = false;
  let retry: ReturnType<typeof setTimeout> | null = null;
  let attempt = 0;

  const connect = () => {
    if (closed) return;
    handlers.onStatus?.("connecting");
    socket = new WebSocket(url);
    socket.onopen = () => {
      attempt = 0;
      handlers.onStatus?.("open");
    };
    socket.onmessage = (event) => {
      try {
        const parsed = parseObserveMessage(JSON.parse(String(event.data)));
        if (parsed) handlers.onMessage(parsed);
      } catch {
        // ignore malformed frames
      }
    };
    socket.onerror = () => {
      handlers.onStatus?.("error");
    };
    socket.onclose = () => {
      handlers.onStatus?.("closed");
      socket = null;
      if (closed) return;
      const delay = Math.min(8000, 400 * 2 ** attempt);
      attempt += 1;
      retry = setTimeout(connect, delay);
    };
  };

  connect();
  return {
    close: () => {
      closed = true;
      if (retry) clearTimeout(retry);
      socket?.close();
      socket = null;
    },
  };
}
