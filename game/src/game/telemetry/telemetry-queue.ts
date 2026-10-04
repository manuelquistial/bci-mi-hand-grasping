import type { TelemetrySamplePayload } from "@/types/domain";

const DEFAULT_MAX = 200;

export class TelemetryQueue {
  private buffer: TelemetrySamplePayload[] = [];

  constructor(private readonly maxBatch = DEFAULT_MAX) {}

  push(sample: TelemetrySamplePayload): void {
    this.buffer.push(sample);
  }

  /** Returns and clears buffer if at least one sample, or empty if under batch threshold for optional flush */
  flush(): TelemetrySamplePayload[] {
    const out = this.buffer;
    this.buffer = [];
    return out;
  }

  /** Peek without clearing — for diagnostics */
  snapshot(): TelemetrySamplePayload[] {
    return [...this.buffer];
  }

  shouldFlush(): boolean {
    return this.buffer.length >= this.maxBatch;
  }

  size(): number {
    return this.buffer.length;
  }
}
