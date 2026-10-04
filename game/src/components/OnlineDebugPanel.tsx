"use client";

import { useCallback } from "react";

import type { GameBridge } from "@/game/bridge/game-bridge";
import { useSettingsStore } from "@/state/settings-store";
import { useHudStore } from "@/state/hud-store";

import styles from "./online-debug.module.css";

type Props = {
  bridge: GameBridge;
};

const EVENT_NAME = "collector:bci-sample";

export function OnlineDebugPanel({ bridge }: Props) {
  const bciStatusLine = useHudStore((s) => s.bciStatusLine);
  const bciDebugLogging = useSettingsStore((s) => s.settings.online.bciDebugLogging);
  const inputAdapter = useSettingsStore((s) => s.settings.online.inputAdapter);
  const setNested = useSettingsStore((s) => s.setNested);

  const pushSample = useCallback(
    (left: number, right: number) => {
      const sample = { left, right, timestamp: performance.now() };
      bridge.dispatchCommand({ type: "PUSH_BCI_SAMPLE", payload: sample });
      window.dispatchEvent(
        new CustomEvent(EVENT_NAME, { detail: sample }),
      );
    },
    [bridge],
  );

  return (
    <div className={styles.panel}>
      <h3 className={styles.title}>Online BCI</h3>
      <p className={styles.hint}>
        <strong>Source:</strong> {inputAdapter} — set under Settings &gt; Online
        BCI input. Dead zone, gain, and smoothing run in the game after samples
        arrive.
      </p>
      {bciStatusLine ? (
        <p className={styles.status} role="status">
          {bciStatusLine}
        </p>
      ) : null}
      <label className={styles.check}>
        <input
          type="checkbox"
          checked={bciDebugLogging}
          onChange={(e) =>
            setNested("online", { bciDebugLogging: e.target.checked })
          }
        />
        BCI debug logging (see browser console)
      </label>
      <p className={styles.hint}>
        <strong>Test buttons</strong> push valid [0,1] samples via{" "}
        <code className={styles.code}>PUSH_BCI_SAMPLE</code> and dispatch{" "}
        <code className={styles.code}>{EVENT_NAME}</code> (for window
        adapter).
      </p>
      <div className={styles.buttons}>
        <button type="button" onClick={() => pushSample(0.8, 0.2)}>
          Strong left
        </button>
        <button type="button" onClick={() => pushSample(0.2, 0.8)}>
          Strong right
        </button>
        <button type="button" onClick={() => pushSample(0.5, 0.5)}>
          Neutral
        </button>
        <button
          type="button"
          onClick={() =>
            window.dispatchEvent(
              new CustomEvent(EVENT_NAME, {
                detail: { left: 2, right: 0, timestamp: performance.now() },
              }),
            )
          }
        >
          Fire invalid (window test)
        </button>
      </div>
    </div>
  );
}
