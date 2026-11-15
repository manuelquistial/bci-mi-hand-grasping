"use client";

import type { OnlineModelPack, SimulatorDataset } from "@/lib/pipeline/client";
import { useSettingsStore } from "@/state/settings-store";

import styles from "../settings-panel.module.css";

export function ServiceUrlField() {
  const baseUrl = useSettingsStore((s) => s.settings.pipeline.baseUrl);
  const setNested = useSettingsStore((s) => s.setNested);
  return (
    <label className={styles.row}>
      Service URL
      <input
        type="text"
        value={baseUrl}
        onChange={(e) => setNested("pipeline", { baseUrl: e.target.value })}
      />
    </label>
  );
}

function selectedDatasetId(
  datasets: SimulatorDataset[],
  pipeline: { dataset: string; sessionDir: string },
): string {
  const match = datasets.find((item) =>
    item.dataset === "recorded_session"
      ? pipeline.dataset === "recorded_session" && item.sessionDir === pipeline.sessionDir
      : item.id === pipeline.dataset,
  );
  return match?.id ?? pipeline.dataset;
}

export function DatasetField({
  datasets,
  error,
}: {
  datasets: SimulatorDataset[];
  error: string | null;
}) {
  const pipeline = useSettingsStore((s) => s.settings.pipeline);
  const setNested = useSettingsStore((s) => s.setNested);
  return (
    <>
      <label className={styles.row}>
        Dataset
        <select
          value={selectedDatasetId(datasets, pipeline)}
          onChange={(e) => {
            const item = datasets.find((entry) => entry.id === e.target.value);
            if (!item) return;
            setNested("pipeline", {
              dataset: item.dataset,
              sessionDir: item.sessionDir,
            });
          }}
        >
          {datasets.length === 0 ? (
            <option value={pipeline.dataset}>{pipeline.dataset}</option>
          ) : (
            datasets.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))
          )}
        </select>
      </label>
      {error ? <p className={styles.note}>{error}</p> : null}
    </>
  );
}

/** Subject, LOSO registry, control strategy and BCI transport (simulator and real). */
export function OnlineControlFields({
  models,
  error,
}: {
  models: OnlineModelPack[];
  error: string | null;
}) {
  const settings = useSettingsStore((s) => s.settings);
  const setNested = useSettingsStore((s) => s.setNested);
  return (
    <>
      <label className={styles.row}>
        Subject
        <input
          type="text"
          value={settings.pipeline.subject}
          placeholder="S001"
          onChange={(e) => setNested("pipeline", { subject: e.target.value })}
        />
      </label>
      <label className={styles.row}>
        Online model registry (LOSO)
        <select
          value={settings.pipeline.registry}
          onChange={(e) => setNested("pipeline", { registry: e.target.value })}
        >
          {models.length === 0 ? (
            <option value={settings.pipeline.registry}>
              {settings.pipeline.registry || "No LOSO packs found"}
            </option>
          ) : (
            models.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))
          )}
        </select>
      </label>
      {error ? <p className={styles.note}>{error}</p> : null}
      <label className={styles.row}>
        Control strategy
        <select
          value={settings.pipeline.controlStrategy}
          onChange={(e) =>
            setNested("pipeline", {
              controlStrategy: e.target.value as typeof settings.pipeline.controlStrategy,
            })
          }
        >
          <option value="baseline_proba">A · baseline (proba / margin)</option>
          <option value="discrete">B · discrete ±1</option>
          <option value="temporal_discrete">C · temporal discrete</option>
          <option value="temporal_proportional">D · temporal proportional</option>
        </select>
      </label>
      <label className={styles.row}>
        BCI input
        <select
          value={settings.online.inputAdapter}
          onChange={(e) =>
            setNested("online", {
              inputAdapter: e.target.value as typeof settings.online.inputAdapter,
            })
          }
        >
          <option value="websocket">WebSocket (frozen model)</option>
          <option value="mock">Mock</option>
          <option value="window">Window event</option>
        </select>
      </label>
      <label className={styles.row}>
        WebSocket URL
        <input
          type="text"
          value={settings.online.websocketUrl}
          onChange={(e) => setNested("online", { websocketUrl: e.target.value })}
        />
      </label>
    </>
  );
}
