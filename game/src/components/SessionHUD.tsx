"use client";

import { useHudStore } from "@/state/hud-store";

import styles from "./session-hud.module.css";

export function SessionHUD() {
  const sessionId = useHudStore((s) => s.sessionId);
  const mode = useHudStore((s) => s.mode);
  const phase = useHudStore((s) => s.phase);
  const trialIndex = useHudStore((s) => s.trialIndex);
  const blockIndex = useHudStore((s) => s.blockIndex);
  const label = useHudStore((s) => s.label);
  const score = useHudStore((s) => s.score);
  const status = useHudStore((s) => s.sessionStatus);
  const bciStatusLine = useHudStore((s) => s.bciStatusLine);

  return (
    <div className={styles.hud} aria-live="polite">
      <div className={styles.row}>
        <span className={styles.k}>Status</span>
        <span>{status}</span>
      </div>
      {mode === "online" && bciStatusLine ? (
        <div className={styles.row}>
          <span className={styles.k}>BCI input</span>
          <span className={styles.wrap}>{bciStatusLine}</span>
        </div>
      ) : null}
      <div className={styles.row}>
        <span className={styles.k}>Mode</span>
        <span>{mode ?? "—"}</span>
      </div>
      <div className={styles.row}>
        <span className={styles.k}>Session</span>
        <span className={styles.mono}>{sessionId ?? "—"}</span>
      </div>
      <div className={styles.row}>
        <span className={styles.k}>Phase</span>
        <span>{phase === "inter_trial_interval" ? "rest" : (phase ?? "—")}</span>
      </div>
      <div className={styles.row}>
        <span className={styles.k}>Trial</span>
        <span>
          #{trialIndex + 1} (block {blockIndex})
        </span>
      </div>
      <div className={styles.row}>
        <span className={styles.k}>Cue</span>
        <span>{label ?? "—"}</span>
      </div>
      <div className={styles.row}>
        <span className={styles.k}>Score</span>
        <span>{score}</span>
      </div>
    </div>
  );
}
