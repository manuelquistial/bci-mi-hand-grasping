"use client";

import { useSettingsStore } from "@/state/settings-store";

import styles from "../settings-panel.module.css";
import { SettingsFieldset } from "./SettingsShell";

/** Trial phase timings shared by the offline and real screens. */
export function TimingsFieldset({
  showFeedback = false,
  showAutoMoveSpeed = false,
}: {
  showFeedback?: boolean;
  showAutoMoveSpeed?: boolean;
}) {
  const offline = useSettingsStore((s) => s.settings.offline);
  const setNested = useSettingsStore((s) => s.setNested);
  return (
    <SettingsFieldset legend="Timings (ms)">
      <label className={styles.row}>
        Fixation
        <input
          type="number"
          value={offline.fixationDurationMs}
          onChange={(e) => setNested("offline", { fixationDurationMs: Number(e.target.value) })}
        />
      </label>
      <label className={styles.row}>
        Cue
        <input
          type="number"
          value={offline.cueDurationMs}
          onChange={(e) => setNested("offline", { cueDurationMs: Number(e.target.value) })}
        />
      </label>
      <label className={styles.row}>
        Control
        <input
          type="number"
          value={offline.controlDurationMs}
          onChange={(e) => setNested("offline", { controlDurationMs: Number(e.target.value) })}
        />
      </label>
      {showFeedback ? (
        <label className={styles.row}>
          Feedback
          <input
            type="number"
            value={offline.feedbackDurationMs}
            onChange={(e) => setNested("offline", { feedbackDurationMs: Number(e.target.value) })}
          />
        </label>
      ) : null}
      <label className={styles.row}>
        Rest
        <input
          type="number"
          value={offline.itiDurationMs}
          onChange={(e) => setNested("offline", { itiDurationMs: Number(e.target.value) })}
        />
      </label>
      {showAutoMoveSpeed ? (
        <label className={styles.row}>
          Auto move speed
          <input
            type="number"
            value={offline.autoMoveSpeed}
            onChange={(e) => setNested("offline", { autoMoveSpeed: Number(e.target.value) })}
          />
        </label>
      ) : null}
    </SettingsFieldset>
  );
}

/** Trial count, blocking and sequence options. */
export function SessionFieldset() {
  const experimental = useSettingsStore((s) => s.settings.experimental);
  const setNested = useSettingsStore((s) => s.setNested);
  return (
    <SettingsFieldset legend="Session">
      <label className={styles.row}>
        Trials (simulator: dataset MI count, e.g. 14)
        <input
          type="number"
          min={1}
          value={experimental.numberOfTrials}
          onChange={(e) => setNested("experimental", { numberOfTrials: Number(e.target.value) })}
        />
      </label>
      <label className={styles.row}>
        Block size
        <input
          type="number"
          min={1}
          value={experimental.blockSize}
          onChange={(e) => setNested("experimental", { blockSize: Number(e.target.value) })}
        />
      </label>
      <label className={styles.row}>
        Random seed
        <input
          type="number"
          value={experimental.randomSeed ?? ""}
          placeholder="null"
          onChange={(e) =>
            setNested("experimental", {
              randomSeed: e.target.value === "" ? null : Number(e.target.value),
            })
          }
        />
      </label>
      <label className={styles.row}>
        Sequence
        <select
          value={experimental.sequenceMode}
          onChange={(e) =>
            setNested("experimental", {
              sequenceMode: e.target.value as typeof experimental.sequenceMode,
            })
          }
        >
          <option value="random_balanced">random_balanced</option>
          <option value="random">random</option>
          <option value="predefined">predefined</option>
        </select>
      </label>
    </SettingsFieldset>
  );
}
