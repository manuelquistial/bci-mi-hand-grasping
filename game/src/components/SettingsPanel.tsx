"use client";

import { useSettingsStore } from "@/state/settings-store";

import styles from "./settings-panel.module.css";

export function SettingsPanel() {
  const settings = useSettingsStore((s) => s.settings);
  const setNested = useSettingsStore((s) => s.setNested);

  return (
    <section className={styles.panel} aria-labelledby="settings-heading">
      <h2 id="settings-heading" className={styles.title}>
        Settings
      </h2>

      <fieldset className={styles.fieldset}>
        <legend>General</legend>
        <label className={styles.row}>
          Width
          <input
            type="number"
            min={480}
            max={1920}
            value={settings.general.width}
            onChange={(e) =>
              setNested("general", { width: Number(e.target.value) })
            }
          />
        </label>
        <label className={styles.row}>
          Height
          <input
            type="number"
            min={320}
            max={1080}
            value={settings.general.height}
            onChange={(e) =>
              setNested("general", { height: Number(e.target.value) })
            }
          />
        </label>
      </fieldset>

      <fieldset className={styles.fieldset}>
        <legend>Offline timings (ms)</legend>
        <label className={styles.row}>
          Fixation
          <input
            type="number"
            value={settings.offline.fixationDurationMs}
            onChange={(e) =>
              setNested("offline", {
                fixationDurationMs: Number(e.target.value),
              })
            }
          />
        </label>
        <label className={styles.row}>
          Cue
          <input
            type="number"
            value={settings.offline.cueDurationMs}
            onChange={(e) =>
              setNested("offline", { cueDurationMs: Number(e.target.value) })
            }
          />
        </label>
        <label className={styles.row}>
          Control
          <input
            type="number"
            value={settings.offline.controlDurationMs}
            onChange={(e) =>
              setNested("offline", {
                controlDurationMs: Number(e.target.value),
              })
            }
          />
        </label>
        <label className={styles.row}>
          Feedback
          <input
            type="number"
            value={settings.offline.feedbackDurationMs}
            onChange={(e) =>
              setNested("offline", {
                feedbackDurationMs: Number(e.target.value),
              })
            }
          />
        </label>
        <label className={styles.row}>
          ITI
          <input
            type="number"
            value={settings.offline.itiDurationMs}
            onChange={(e) =>
              setNested("offline", { itiDurationMs: Number(e.target.value) })
            }
          />
        </label>
        <label className={styles.row}>
          Auto move speed
          <input
            type="number"
            value={settings.offline.autoMoveSpeed}
            onChange={(e) =>
              setNested("offline", { autoMoveSpeed: Number(e.target.value) })
            }
          />
        </label>
      </fieldset>

      <fieldset className={styles.fieldset}>
        <legend>Online BCI input</legend>
        <label className={styles.row}>
          Source
          <select
            value={settings.online.inputAdapter}
            onChange={(e) =>
              setNested("online", {
                inputAdapter: e.target.value as typeof settings.online.inputAdapter,
              })
            }
          >
            <option value="mock">Mock (built-in stream)</option>
            <option value="window">Window events (CustomEvent)</option>
            <option value="websocket">WebSocket JSON</option>
          </select>
        </label>
        {settings.online.inputAdapter === "websocket" ? (
          <label className={styles.row}>
            WebSocket URL
            <input
              type="text"
              value={settings.online.websocketUrl}
              onChange={(e) =>
                setNested("online", { websocketUrl: e.target.value })
              }
            />
          </label>
        ) : null}
        {settings.online.inputAdapter === "websocket" ? (
          <>
            <label className={styles.row}>
              Max reconnects
              <input
                type="number"
                min={0}
                max={50}
                value={settings.online.wsMaxReconnectAttempts}
                onChange={(e) =>
                  setNested("online", {
                    wsMaxReconnectAttempts: Number(e.target.value),
                  })
                }
              />
            </label>
            <label className={styles.row}>
              Reconnect base (ms)
              <input
                type="number"
                min={200}
                value={settings.online.wsReconnectBaseMs}
                onChange={(e) =>
                  setNested("online", {
                    wsReconnectBaseMs: Number(e.target.value),
                  })
                }
              />
            </label>
            <label className={styles.row}>
              Reconnect cap (ms)
              <input
                type="number"
                min={1000}
                value={settings.online.wsReconnectMaxMs}
                onChange={(e) =>
                  setNested("online", {
                    wsReconnectMaxMs: Number(e.target.value),
                  })
                }
              />
            </label>
          </>
        ) : null}
        <label className={styles.rowCheck}>
          <input
            type="checkbox"
            checked={settings.online.bciDebugLogging}
            onChange={(e) =>
              setNested("online", { bciDebugLogging: e.target.checked })
            }
          />
          BCI debug logging (console)
        </label>
      </fieldset>

      <fieldset className={styles.fieldset}>
        <legend>Online control (signal path)</legend>
        <label className={styles.row}>
          Dead zone
          <input
            type="number"
            step="0.01"
            value={settings.online.deadZone}
            onChange={(e) =>
              setNested("online", { deadZone: Number(e.target.value) })
            }
          />
        </label>
        <label className={styles.row}>
          Gain
          <input
            type="number"
            step="0.05"
            value={settings.online.gain}
            onChange={(e) =>
              setNested("online", { gain: Number(e.target.value) })
            }
          />
        </label>
        <label className={styles.row}>
          Smoothing
          <input
            type="number"
            step="0.05"
            value={settings.online.smoothing}
            onChange={(e) =>
              setNested("online", { smoothing: Number(e.target.value) })
            }
          />
        </label>
        <label className={styles.row}>
          Max speed
          <input
            type="number"
            value={settings.online.maxHorizontalSpeed}
            onChange={(e) =>
              setNested("online", {
                maxHorizontalSpeed: Number(e.target.value),
              })
            }
          />
        </label>
        <label className={styles.row}>
          Mock interval (ms)
          <input
            type="number"
            value={settings.online.mockSampleIntervalMs}
            onChange={(e) =>
              setNested("online", {
                mockSampleIntervalMs: Number(e.target.value),
              })
            }
          />
        </label>
        <label className={styles.row}>
          Mock pattern
          <select
            value={settings.online.mockPattern}
            onChange={(e) =>
              setNested("online", {
                mockPattern: e.target.value as typeof settings.online.mockPattern,
              })
            }
          >
            <option value="strong_left">strong_left</option>
            <option value="strong_right">strong_right</option>
            <option value="noisy">noisy</option>
            <option value="weak_ambiguous">weak_ambiguous</option>
            <option value="alternating">alternating</option>
          </select>
        </label>
      </fieldset>

      <fieldset className={styles.fieldset}>
        <legend>Experimental</legend>
        <label className={styles.row}>
          Trials
          <input
            type="number"
            min={1}
            value={settings.experimental.numberOfTrials}
            onChange={(e) =>
              setNested("experimental", {
                numberOfTrials: Number(e.target.value),
              })
            }
          />
        </label>
        <label className={styles.row}>
          Block size
          <input
            type="number"
            min={1}
            value={settings.experimental.blockSize}
            onChange={(e) =>
              setNested("experimental", {
                blockSize: Number(e.target.value),
              })
            }
          />
        </label>
        <label className={styles.row}>
          Random seed
          <input
            type="number"
            value={settings.experimental.randomSeed ?? ""}
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
            value={settings.experimental.sequenceMode}
            onChange={(e) =>
              setNested("experimental", {
                sequenceMode: e.target.value as typeof settings.experimental.sequenceMode,
              })
            }
          >
            <option value="random_balanced">random_balanced</option>
            <option value="random">random</option>
            <option value="predefined">predefined</option>
          </select>
        </label>
      </fieldset>
    </section>
  );
}
