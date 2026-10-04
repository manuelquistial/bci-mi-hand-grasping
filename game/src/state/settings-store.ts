import { create } from "zustand";

import {
  defaultSessionSettings,
  type SessionSettings,
} from "@/types/settings";
import type { SessionMode } from "@/types/domain";

type SettingsState = {
  settings: SessionSettings;
  mode: SessionMode;
  setSettings: (partial: Partial<SessionSettings>) => void;
  setNested: <K extends keyof SessionSettings>(
    key: K,
    partial: Partial<SessionSettings[K]>,
  ) => void;
  setMode: (mode: SessionMode) => void;
  reset: () => void;
};

export const useSettingsStore = create<SettingsState>((set) => ({
  settings: defaultSessionSettings(),
  mode: "offline",
  setSettings: (partial) =>
    set((s) => ({
      settings: { ...s.settings, ...partial } as SessionSettings,
    })),
  setNested: (key, partial) =>
    set((s) => ({
      settings: {
        ...s.settings,
        [key]: { ...s.settings[key], ...partial },
      },
    })),
  setMode: (mode) => set({ mode }),
  reset: () => set({ settings: defaultSessionSettings(), mode: "offline" }),
}));
