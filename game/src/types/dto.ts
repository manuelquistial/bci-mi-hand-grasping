import type { SessionMode } from "./domain";

export type SessionCreateResponse = {
  id: string;
  mode: SessionMode;
  startedAt: string;
  settingsJson: unknown;
};

export type SessionListItem = {
  id: string;
  mode: string;
  startedAt: string;
  endedAt: string | null;
  totalScore: number;
  successRate: number | null;
  createdAt: string;
};

export type SessionDetailResponse = {
  session: SessionListItem & { settingsJson: unknown };
  trials: {
    id: string;
    trialIndex: number;
    blockIndex: number;
    label: string;
    targetSide: string;
    outcome: string | null;
    score: number;
  }[];
  summary: {
    trialCount: number;
    successCount: number;
  };
};

export type TelemetryBatchRequest = {
  sessionId: string;
  samples: {
    trialId: string | null;
    timestamp: number;
    leftValue: number;
    rightValue: number;
    controlSignal: number;
    smoothedSignal: number;
    velocityX: number;
    avatarX: number;
    phase: string;
  }[];
};
