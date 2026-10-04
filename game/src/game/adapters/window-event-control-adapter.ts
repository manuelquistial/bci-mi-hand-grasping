import { parseBCIControlSample } from "@/lib/bci/validate-sample";
import type { BCIControlSample } from "@/types/domain";

import type { BciStatusEmitter } from "./online-adapter-factory";
import type {
  BciInputKind,
  ControlAdapter,
  ControlAdapterStatus,
} from "./control-adapter";

const EVENT_NAME = "collector:bci-sample";

const INVALID_LOG_MS = 900;

/**
 * Ingests external BCI samples from `window` via
 * `CustomEvent` with `type === "collector:bci-sample"`.
 * `event.detail` must pass {@link parseBCIControlSample}.
 */
export class WindowEventControlAdapter implements ControlAdapter {
  private readonly listeners = new Set<(s: BCIControlSample) => void>();

  private status: ControlAdapterStatus = "idle";

  private lastInvalidLogAt = 0;

  private invalidBurst = 0;

  private readonly bound = (ev: Event) => this.onWindowEvent(ev);

  constructor(
    private readonly emit: BciStatusEmitter,
    private readonly getBciDebug: () => boolean,
  ) {}

  start(): void {
    if (this.status === "running") return;
    window.addEventListener(EVENT_NAME, this.bound);
    this.status = "running";
    this.emit({
      kind: "window",
      state: "listening",
      detail: `Event "${EVENT_NAME}" (CustomEvent, bubbles: false)`,
    });
  }

  stop(): void {
    window.removeEventListener(EVENT_NAME, this.bound);
    this.status = "idle";
    this.emit({ kind: "window", state: "idle" });
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
    return "window";
  }

  getConnectionSummary(): string {
    if (this.status !== "running") return "Window events (stopped)";
    return `Listening: ${EVENT_NAME}`;
  }

  private onWindowEvent(ev: Event): void {
    const ce = ev as CustomEvent<unknown>;
    const parsed = parseBCIControlSample(ce.detail);
    if (!parsed.ok) {
      this.invalidBurst += 1;
      const now = performance.now();
      if (now - this.lastInvalidLogAt > INVALID_LOG_MS) {
        this.lastInvalidLogAt = now;
        this.emit({
          kind: "window",
          state: "invalid_sample",
          detail: `${parsed.error} (${this.invalidBurst} since last notice)`,
        });
        this.invalidBurst = 0;
        if (this.getBciDebug()) {
          console.warn("[BCI window]", parsed.error, ce.detail);
        }
      }
      return;
    }
    if (this.getBciDebug()) {
      console.debug("[BCI window] sample", parsed.value);
    }
    for (const cb of this.listeners) cb(parsed.value);
  }
}
