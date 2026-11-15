"use client";

import styles from "./game-controls.module.css";

type Props = {
  onStart: () => void | Promise<void>;
  starting: boolean;
  label?: string;
  onOpenMonitor?: () => void;
  showMonitor?: boolean;
};

export function GameControls({
  onStart,
  starting,
  label = "Start session",
  onOpenMonitor,
  showMonitor = false,
}: Props) {
  return (
    <div className={styles.buttons}>
      {showMonitor && onOpenMonitor ? (
        <button
          type="button"
          className={styles.secondary}
          onClick={onOpenMonitor}
          disabled={starting}
        >
          Abrir monitor
        </button>
      ) : null}
      <button
        type="button"
        className={styles.primary}
        onClick={() => void onStart()}
        disabled={starting}
      >
        {starting ? "Starting…" : label}
      </button>
    </div>
  );
}
