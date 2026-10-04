import { parseBCIControlSample } from "@/lib/bci/validate-sample";
import type { BCIControlSample } from "@/types/domain";

import type { BciStatusEmitter } from "./online-adapter-factory";
import type {
  BciInputKind,
  ControlAdapter,
  ControlAdapterStatus,
} from "./control-adapter";

export type WebSocketAdapterConfig = {
  url: string;
  maxReconnectAttempts: number;
  baseDelayMs: number;
  maxDelayMs: number;
  getDebug: () => boolean;
};

/**
 * Ingests JSON lines like `{ "left": 0.2, "right": 0.8, "timestamp": 1234 }`
 * (same shape as {@link BCIControlSample} after validation).
 */
export class WebSocketControlAdapter implements ControlAdapter {
  private socket: WebSocket | null = null;

  private readonly listeners = new Set<(s: BCIControlSample) => void>();

  private status: ControlAdapterStatus = "idle";

  private userInitiatedClose = false;

  /** Increments on each post-close scheduled reconnect; reset on `open`. */
  private reconnectCount = 0;

  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  private lastInvalidLogAt = 0;

  private invalidBurst = 0;

  constructor(
    private readonly config: WebSocketAdapterConfig,
    private readonly emit: BciStatusEmitter,
  ) {}

  start(): void {
    this.userInitiatedClose = false;
    this.reconnectCount = 0;
    this.clearReconnectTimer();
    this.openSocket("initial");
  }

  stop(): void {
    this.userInitiatedClose = true;
    this.clearReconnectTimer();
    this.socket?.close();
    this.socket = null;
    this.status = "idle";
    this.emit({ kind: "websocket", state: "disconnected" });
  }

  subscribe(callback: (sample: BCIControlSample) => void): void {
    this.listeners.add(callback);
  }

  unsubscribe(callback: (sample: BCIControlSample) => void): void {
    this.listeners.delete(callback);
  }

  getStatus(): ControlAdapterStatus {
    return this.status;
  }

  getInputKind(): BciInputKind {
    return "websocket";
  }

  getConnectionSummary(): string {
    const u = this.config.url;
    if (this.userInitiatedClose) return "WebSocket (stopped)";
    if (!u) return "WebSocket (no URL)";
    if (this.socket?.readyState === WebSocket.OPEN) return `WebSocket: ${u}`;
    if (this.socket?.readyState === WebSocket.CONNECTING) return "Connecting…";
    return "WebSocket: offline / retrying";
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private openSocket(phase: "initial" | "reconnect"): void {
    if (this.userInitiatedClose) return;
    this.clearReconnectTimer();
    if (!this.config.url) {
      this.emit({
        kind: "websocket",
        state: "error",
        detail: "Missing WebSocket URL in settings",
      });
      this.status = "idle";
      return;
    }

    this.status = "running";
    if (this.socket) {
      this.socket.onopen = null;
      this.socket.onmessage = null;
      this.socket.onerror = null;
      this.socket.onclose = null;
      try {
        this.socket.close();
      } catch {
        /* ignore */
      }
      this.socket = null;
    }

    this.emit({
      kind: "websocket",
      state: phase === "initial" ? "connecting" : "reconnecting",
      detail:
        phase === "initial"
          ? this.config.url
          : `retry #${this.reconnectCount} → ${this.config.url}`,
    });
    if (this.config.getDebug()) {
      console.debug(`[BCI ws] openSocket`, phase, this.config.url);
    }

    try {
      this.socket = new WebSocket(this.config.url);
    } catch (e) {
      this.emit({
        kind: "websocket",
        state: "error",
        detail: e instanceof Error ? e.message : "WebSocket constructor failed",
      });
      this.scheduleReconnect();
      return;
    }

    this.socket.onopen = () => {
      this.reconnectCount = 0;
      this.emit({ kind: "websocket", state: "open", detail: this.config.url });
    };

    this.socket.onmessage = (ev) => {
      this.handleMessage(ev);
    };

    this.socket.onerror = () => {
      this.emit({
        kind: "websocket",
        state: "error",
        detail: "WebSocket error (see browser devtools network)",
      });
    };

    this.socket.onclose = () => {
      this.emit({
        kind: "websocket",
        state: "closed",
        detail: "socket closed",
      });
      this.socket = null;
      if (!this.userInitiatedClose) {
        this.scheduleReconnect();
      }
    };
  }

  private handleMessage(ev: MessageEvent): void {
    if (this.userInitiatedClose) return;
    let raw: unknown;
    try {
      raw = JSON.parse(String(ev.data));
    } catch {
      this.warnInvalid("JSON parse error");
      return;
    }
    if (Array.isArray(raw) && raw.length > 0) {
      raw = raw[0];
    }
    const parsed = parseBCIControlSample(raw);
    if (!parsed.ok) {
      this.warnInvalid(parsed.error, raw);
      return;
    }
    if (this.config.getDebug()) {
      console.debug("[BCI ws] sample", parsed.value);
    }
    for (const cb of this.listeners) cb(parsed.value);
  }

  private warnInvalid(msg: string, data?: unknown): void {
    this.invalidBurst += 1;
    const now = performance.now();
    if (now - this.lastInvalidLogAt > 900) {
      this.lastInvalidLogAt = now;
      this.emit({
        kind: "websocket",
        state: "invalid_sample",
        detail: `${msg} (${this.invalidBurst} since last notice)`,
      });
      this.invalidBurst = 0;
      if (this.config.getDebug()) {
        console.warn("[BCI ws] invalid", msg, data);
      }
    }
  }

  private scheduleReconnect(): void {
    if (this.userInitiatedClose) return;
    if (this.reconnectCount >= this.config.maxReconnectAttempts) {
      this.emit({
        kind: "websocket",
        state: "exhausted",
        detail: `Stopped after ${this.config.maxReconnectAttempts} reconnect attempt(s)`,
      });
      this.status = "idle";
      return;
    }
    this.reconnectCount += 1;
    const e = this.reconnectCount - 1;
    const delay = Math.min(
      this.config.maxDelayMs,
      this.config.baseDelayMs * 2 ** e,
    );
    this.emit({
      kind: "websocket",
      state: "reconnect_scheduled",
      detail: `in ${Math.round(delay)}ms (try ${this.reconnectCount}/${this.config.maxReconnectAttempts})`,
    });
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.openSocket("reconnect");
    }, delay);
  }
}
