import { create } from "zustand";

type PersistenceState = {
  lastError: string | null;
  setLastError: (msg: string | null) => void;
};

export const usePersistenceStore = create<PersistenceState>((set) => ({
  lastError: null,
  setLastError: (lastError) => set({ lastError }),
}));
