"use client";

import { useEffect, useRef, useState } from "react";

import {
  pipelineListDatasets,
  pipelineListSubjects,
  type SimulatorDataset,
  type SimulatorSubject,
} from "@/lib/pipeline/client";
import { useSettingsStore } from "@/state/settings-store";

import styles from "./settings-panel.module.css";

export type SetupScreen = "offline" | "simulator" | "signal" | "real";

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

export function SettingsPanel({ screen }: { screen: SetupScreen }) {
  const settings = useSettingsStore((s) => s.settings);
  const setNested = useSettingsStore((s) => s.setNested);
  const title =
    screen === "offline"
      ? "Offline"
      : screen === "simulator"
        ? "Simulator"
        : screen === "signal"
          ? "Online simulated signal"
          : "Real";
  const showsDataset = screen === "simulator" || screen === "signal";
  const [datasets, setDatasets] = useState<SimulatorDataset[]>([]);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [subjects, setSubjects] = useState<SimulatorSubject[]>([]);
  const [subjectError, setSubjectError] = useState<string | null>(null);
  const subjectsDataset = useRef(settings.pipeline.dataset);

  useEffect(() => {
    if (screen !== "simulator" && screen !== "signal") return;
    let cancelled = false;
    const baseUrl = useSettingsStore.getState().settings.pipeline.baseUrl;
    pipelineListDatasets(baseUrl)
      .then((listed) => {
        if (cancelled) return;
        setDatasets(listed);
        setDatasetError(null);
        const current = useSettingsStore.getState().settings.pipeline.dataset;
        if (listed.length > 0 && !listed.some((item) => item.id === current)) {
          setNested("pipeline", { dataset: listed[0].id });
        }
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setDatasetError(error instanceof Error ? error.message : "Could not load datasets");
      });
    return () => {
      cancelled = true;
    };
  }, [screen, setNested]);

  useEffect(() => {
    if (screen !== "signal") return;
    const dataset = settings.pipeline.dataset;
    const datasetChanged = subjectsDataset.current !== dataset;
    subjectsDataset.current = dataset;
    if (!dataset || dataset === "recorded_session") {
      setSubjects([]);
      setSubjectError(null);
      if (useSettingsStore.getState().settings.pipeline.subject) {
        setNested("pipeline", { subject: "" });
      }
      return;
    }
    if (datasetChanged) {
      setSubjects([]);
      setSubjectError(null);
      setNested("pipeline", { subject: "" });
    }
    let cancelled = false;
    pipelineListSubjects(settings.pipeline.baseUrl, dataset)
      .then((listed) => {
        if (cancelled) return;
        setSubjects(listed);
        setSubjectError(null);
        const current = useSettingsStore.getState().settings.pipeline.subject;
        if (listed.length > 0 && !listed.some((item) => item.id === current)) {
          setNested("pipeline", { subject: listed[0].id });
        }
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setSubjects([]);
        setSubjectError(error instanceof Error ? error.message : "Could not load participants");
      });
    return () => {
      cancelled = true;
    };
  }, [screen, settings.pipeline.baseUrl, settings.pipeline.dataset, setNested]);

  return (
    <section className={styles.panel} aria-labelledby="settings-heading">
      <h2 id="settings-heading" className={styles.title}>
        {title}
      </h2>

      <fieldset className={styles.fieldset}>
        <legend>Pipeline</legend>
        <label className={styles.row}>
          Service URL
          <input
            type="text"
            value={settings.pipeline.baseUrl}
            onChange={(e) => setNested("pipeline", { baseUrl: e.target.value })}
          />
        </label>
        {showsDataset ? (
          <>
            <label className={styles.row}>
              Dataset
              <select
                value={selectedDatasetId(datasets, settings.pipeline)}
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
                  <option value={settings.pipeline.dataset}>{settings.pipeline.dataset}</option>
                ) : (
                  datasets.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))
                )}
              </select>
            </label>
            {datasetError ? <p className={styles.note}>{datasetError}</p> : null}
            {screen === "signal" && subjects.length > 0 ? (
              <label className={styles.row}>
                Participant
                <select
                  value={settings.pipeline.subject}
                  onChange={(e) => setNested("pipeline", { subject: e.target.value })}
                >
                  {subjects.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            {screen === "signal" && subjectError ? (
              <p className={styles.note}>{subjectError}</p>
            ) : null}
          </>
        ) : null}
      </fieldset>

      {showsDataset ? null : (
      <>
      <fieldset className={styles.fieldset}>
        <legend>Timings (ms)</legend>
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
        {screen === "real" ? (
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
        ) : null}
        <label className={styles.row}>
          Rest
          <input
            type="number"
            value={settings.offline.itiDurationMs}
            onChange={(e) =>
              setNested("offline", { itiDurationMs: Number(e.target.value) })
            }
          />
        </label>
        {screen === "offline" ? (
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
        ) : null}
      </fieldset>

      <fieldset className={styles.fieldset}>
        <legend>Session</legend>
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
      </>
      )}
    </section>
  );
}
