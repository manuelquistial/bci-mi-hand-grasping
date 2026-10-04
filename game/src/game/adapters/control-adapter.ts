import type { BCIControlSample } from "@/types/domain";

export type BciInputKind = "mock" | "window" | "websocket";

export type ControlAdapterStatus = "idle" | "running";

export interface ControlAdapter {
  start(): void;
  stop(): void;
  subscribe(callback: (sample: BCIControlSample) => void): void;
  unsubscribe(callback: (sample: BCIControlSample) => void): void;
  getStatus(): ControlAdapterStatus;
  getInputKind(): BciInputKind;
  /** One line for HUD (connection / listening / reconnect, etc.) */
  getConnectionSummary(): string;
}
