import type { SessionMode } from "./domain";

export type TargetSpawnMode = "random" | "alternate" | "balanced";

export type SequenceMode = "random_balanced" | "predefined" | "random";

export type SessionSettings = {
  general: {
    fullscreenPlaceholder: boolean;
    audioPlaceholder: boolean;
  };
  offline: {
    fixationDurationMs: number;
    cueDurationMs: number;
    controlDurationMs: number;
    feedbackDurationMs: number;
    itiDurationMs: number;
    autoMoveSpeed: number;
  };
  online: {
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
  experimental: {
    numberOfTrials: number;
    blockSize: number;
    randomSeed: number | null;
    sequenceMode: SequenceMode;
  };
  pipeline: {
    baseUrl: string;
    lslName: string;
    onlineSource: "simulator" | "cyton";
    dataset: string;
    sessionDir: string;
  };
};

export const defaultSessionSettings = (): SessionSettings => ({
  general: {
    fullscreenPlaceholder: false,
    audioPlaceholder: false,
  },
  offline: {
    fixationDurationMs: 800,
    cueDurationMs: 1000,
    controlDurationMs: 4000,
    feedbackDurationMs: 700,
    itiDurationMs: 500,
    autoMoveSpeed: 220,
  },
  online: {
    inputAdapter: "mock",
    websocketUrl: "ws://127.0.0.1:8090",
    bciDebugLogging: false,
    wsMaxReconnectAttempts: 8,
    wsReconnectBaseMs: 1000,
    wsReconnectMaxMs: 30_000,
    deadZone: 0.08,
    gain: 1.2,
    smoothing: 0.25,
    maxHorizontalSpeed: 280,
    mockSampleIntervalMs: 33,
    targetSpawnMode: "balanced",
    mockPattern: "alternating",
    mockNoiseLevel: 0.05,
  },
  experimental: {
    numberOfTrials: 8,
    blockSize: 4,
    randomSeed: 42,
    sequenceMode: "random_balanced",
  },
  pipeline: {
    baseUrl: "http://127.0.0.1:8765",
    lslName: "OpenBCI_EEG",
    onlineSource: "simulator",
    dataset: "bci_iv2a",
    sessionDir: "",
  },
});

export function settingsWithModeSnapshot(
  settings: SessionSettings,
  mode: SessionMode,
): Record<string, unknown> {
  return { ...settings, mode, snapshotAt: new Date().toISOString() };
}
