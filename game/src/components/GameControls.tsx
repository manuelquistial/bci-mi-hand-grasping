"use client";

import type { SessionMode } from "@/types/domain";

import styles from "./game-controls.module.css";

type Props = {
  onStart: () => void | Promise<void>;
  onStop: () => void;
  onPause: () => void;
  onResume: () => void;
  mode: SessionMode;
  onModeChange: (m: SessionMode) => void;
};

export function GameControls({
  onStart,
  onStop,
  onPause,
  onResume,
  mode,
  onModeChange,
}: Props) {
  return (
    <div className={styles.wrap}>
      <div className={styles.mode}>
        <span className={styles.label}>Mode</span>
        <label className={styles.radio}>
          <input
            type="radio"
            name="mode"
            checked={mode === "offline"}
            onChange={() => onModeChange("offline")}
          />
          Offline
        </label>
        <label className={styles.radio}>
          <input
            type="radio"
            name="mode"
            checked={mode === "online"}
            onChange={() => onModeChange("online")}
          />
          Online
        </label>
      </div>
      <div className={styles.buttons}>
        <button type="button" className={styles.primary} onClick={() => void onStart()}>
          Start session
        </button>
        <button type="button" className={styles.secondary} onClick={onPause}>
          Pause
        </button>
        <button type="button" className={styles.secondary} onClick={onResume}>
          Resume
        </button>
        <button type="button" className={styles.danger} onClick={onStop}>
          Stop
        </button>
      </div>
    </div>
  );
}
