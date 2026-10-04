import type {
  BCIControlSample,
  SessionMode,
  TelemetrySamplePayload,
  TrialDefinition,
  TrialPhase,
} from "./domain";
import type { SessionSettings } from "./settings";

export type GameCommand =
  | {
      type: "START_SESSION";
      payload: {
        sessionId: string;
        mode: SessionMode;
        settings: SessionSettings;
        trials: TrialDefinition[];
      };
    }
  | { type: "STOP_SESSION" }
  | { type: "UPDATE_SETTINGS"; payload: { settings: SessionSettings } }
  | {
      type: "LOAD_TRIAL_SEQUENCE";
      payload: { trials: TrialDefinition[] };
    }
  | { type: "PUSH_BCI_SAMPLE"; payload: BCIControlSample }
  | { type: "SWITCH_MODE"; payload: { mode: SessionMode } };

export type GameEvent =
  | { type: "SESSION_STARTED"; sessionId: string; mode: SessionMode }
  | { type: "SESSION_STOPPED" }
  | {
      type: "PHASE_CHANGED";
      phase: TrialPhase;
      trialIndex: number;
      blockIndex: number;
    }
  | {
      type: "TRIAL_STARTED";
      trialIndex: number;
      blockIndex: number;
      label: string;
      targetSide: string;
    }
  | {
      type: "TRIAL_COMPLETED";
      trialIndex: number;
      blockIndex: number;
      label: string;
      targetSide: string;
      outcome: string;
      scoreDelta: number;
      trialScore: number;
    }
  | { type: "SCORE_CHANGED"; totalScore: number }
  | {
      type: "SESSION_COMPLETED";
      sessionId: string;
      totalScore: number;
      successCount: number;
      trialCount: number;
    }
  | {
      type: "TELEMETRY_BATCH_READY";
      samples: TelemetrySamplePayload[];
    }
  | {
      type: "ERROR";
      message: string;
      detail?: unknown;
    }
  | {
      type: "BCI_INPUT_STATUS";
      kind: "mock" | "window" | "websocket" | "bridge";
      state: string;
      detail?: string;
    };

export type GameEventListener = (event: GameEvent) => void;
export type GameCommandListener = (command: GameCommand) => void;
