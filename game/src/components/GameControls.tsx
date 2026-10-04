"use client";

import styles from "./game-controls.module.css";

type Props = {
  onStart: () => void | Promise<void>;
  starting: boolean;
};

export function GameControls({ onStart, starting }: Props) {
  return (
    <div className={styles.buttons}>
      <button
        type="button"
        className={styles.primary}
        onClick={() => void onStart()}
        disabled={starting}
      >
        {starting ? "Starting…" : "Start session"}
      </button>
    </div>
  );
}
