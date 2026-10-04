import type { BCIControlSample } from "@/types/domain";
import type { SessionSettings } from "@/types/settings";

import type {
  BciInputKind,
  ControlAdapter,
  ControlAdapterStatus,
} from "./control-adapter";

export type MockPattern = SessionSettings["online"]["mockPattern"];

export class MockControlAdapter implements ControlAdapter {
  private interval: ReturnType<typeof setInterval> | null = null;

  private readonly listeners = new Set<(s: BCIControlSample) => void>();

  private status: ControlAdapterStatus = "idle";

  private t = 0;

  constructor(
    private readonly settings: SessionSettings,
    private readonly pattern: MockPattern = "alternating",
  ) {}

  start(): void {
    if (this.interval) return;
    this.status = "running";
    const ms = Math.max(16, this.settings.online.mockSampleIntervalMs);
    this.interval = setInterval(() => this.tick(), ms);
  }

  stop(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    this.status = "idle";
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
    return "mock";
  }

  getConnectionSummary(): string {
    return this.status === "running"
      ? `Mock stream (${this.pattern}, ${this.settings.online.mockSampleIntervalMs}ms)`
      : "Mock (stopped)";
  }

  private tick(): void {
    this.t += 1;
    const sample = this.makeSample();
    for (const cb of this.listeners) cb(sample);
  }

  private makeSample(): BCIControlSample {
    const noise = this.settings.online.mockNoiseLevel;
    const n = () => (Math.random() - 0.5) * 2 * noise;
    const p = this.pattern;
    let left = 0.5;
    let right = 0.5;

    if (p === "strong_left") {
      left = 0.85 + n();
      right = 0.15 + n();
    } else if (p === "strong_right") {
      left = 0.15 + n();
      right = 0.85 + n();
    } else if (p === "noisy") {
      left = 0.5 + (Math.random() - 0.5) * 0.9;
      right = 1 - left;
    } else if (p === "weak_ambiguous") {
      left = 0.48 + n() * 0.1;
      right = 1 - left;
    } else {
      const phase = Math.floor(this.t / 40) % 2;
      if (phase === 0) {
        left = 0.72 + n();
        right = 0.28 + n();
      } else {
        left = 0.28 + n();
        right = 0.72 + n();
      }
    }

    left = Math.max(0, Math.min(1, left));
    right = Math.max(0, Math.min(1, right));
    const sum = left + right;
    left /= sum;
    right /= sum;

    return { left, right, timestamp: performance.now() };
  }
}
