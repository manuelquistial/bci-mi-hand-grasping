export type TrialPhase =
  | "idle"
  | "preparing"
  | "fixation"
  | "cue"
  | "control"
  | "feedback"
  | "inter_trial_interval"
  | "finished";

export type SessionMode = "offline" | "online";

export type TrialOutcome = "success" | "failure" | "timeout" | "aborted" | "pending";

export type TargetSide = "left" | "right";

export type TrialLabel = "left" | "right";

export type BCIControlSample = {
  left: number;
  right: number;
  timestamp: number;
};

export type TelemetrySamplePayload = {
  sessionId: string;
  trialId: string | null;
  timestamp: number;
  leftValue: number;
  rightValue: number;
  controlSignal: number;
  smoothedSignal: number;
  velocityX: number;
  avatarX: number;
  phase: TrialPhase;
};

export type PhaseTimestamps = Partial<Record<TrialPhase, number>>;

export type TrialDefinition = {
  id?: string;
  sessionId?: string;
  index: number;
  blockIndex: number;
  mode: SessionMode;
  label: TrialLabel;
  targetSide: TargetSide;
  fixationDurationMs: number;
  cueDurationMs: number;
  controlDurationMs: number;
  feedbackDurationMs: number;
  itiDurationMs: number;
  startedAt?: number;
  endedAt?: number;
  outcome?: TrialOutcome;
  score?: number;
  metadata?: Record<string, unknown>;
};
