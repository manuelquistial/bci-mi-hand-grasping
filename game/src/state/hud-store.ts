import { create } from "zustand";

import type { TrialPhase } from "@/types/domain";
import type { SessionMode } from "@/types/domain";

type HudState = {
  sessionId: string | null;
  mode: SessionMode | null;
  phase: TrialPhase | null;
  trialIndex: number;
  blockIndex: number;
  label: string | null;
  score: number;
  leftProb: number | null;
  rightProb: number | null;
  controlSignal: number | null;
  bciStatusLine: string | null;
  sessionStatus: "idle" | "running" | "completed";
  reset: () => void;
};

export const useHudStore = create<HudState>((set) => ({
  sessionId: null,
  mode: null,
  phase: null,
  trialIndex: 0,
  blockIndex: 0,
  label: null,
  score: 0,
  leftProb: null,
  rightProb: null,
  controlSignal: null,
  bciStatusLine: null,
  sessionStatus: "idle",
  reset: () =>
    set({
      sessionId: null,
      mode: null,
      phase: null,
      trialIndex: 0,
      blockIndex: 0,
      label: null,
      score: 0,
      leftProb: null,
      rightProb: null,
      controlSignal: null,
      bciStatusLine: null,
      sessionStatus: "idle",
    }),
}));
