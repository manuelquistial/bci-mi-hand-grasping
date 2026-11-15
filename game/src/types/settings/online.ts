export type TargetSpawnMode = "random" | "alternate" | "balanced";

export type OnlineSettings = {
  /** Where Phaser subscribes for BCI stream samples in online mode. */
  inputAdapter: "mock" | "window" | "websocket";
  websocketUrl: string;
  bciDebugLogging: boolean;
  wsMaxReconnectAttempts: number;
  wsReconnectBaseMs: number;
  wsReconnectMaxMs: number;
  deadZone: number;
  gain: number;
  smoothing: number;
  maxHorizontalSpeed: number;
  /** Drop BCI commands older than this many ms (stale-hold prevention). */
  commandExpiryMs: number;
  /** When true, temporal strategies skip Phaser EMA (decoder already integrates). */
  skipPhaserSmoothingForTemporal: boolean;
  mockSampleIntervalMs: number;
  targetSpawnMode: TargetSpawnMode;
  mockPattern:
    | "strong_left"
    | "strong_right"
    | "noisy"
    | "weak_ambiguous"
    | "alternating";
  mockNoiseLevel: number;
};

export const defaultOnlineSettings = (): OnlineSettings => ({
  inputAdapter: "websocket",
  websocketUrl: "ws://127.0.0.1:8090",
  bciDebugLogging: false,
  wsMaxReconnectAttempts: 8,
  wsReconnectBaseMs: 1000,
  wsReconnectMaxMs: 30_000,
  deadZone: 0.08,
  gain: 1.2,
  smoothing: 0.25,
  maxHorizontalSpeed: 280,
  commandExpiryMs: 750,
  skipPhaserSmoothingForTemporal: true,
  mockSampleIntervalMs: 33,
  targetSpawnMode: "balanced",
  mockPattern: "alternating",
  mockNoiseLevel: 0.05,
});
