"use client";

import { useEffect, useState } from "react";

import {
  pipelineListDatasets,
  pipelineListOnlineModels,
  type OnlineModelPack,
  type SimulatorDataset,
} from "@/lib/pipeline/client";
import { useSettingsStore } from "@/state/settings-store";

import styles from "./settings-panel.module.css";

export type SetupScreen = "offline" | "pre-model" | "simulator" | "real";

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
        : screen === "pre-model"
          ? "Pre model"
          : "Real";
  const showsDataset = screen === "simulator" || screen === "pre-model";
  const [datasets, setDatasets] = useState<SimulatorDataset[]>([]);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [losoModels, setLosoModels] = useState<OnlineModelPack[]>([]);
  const [registryError, setRegistryError] = useState<string | null>(null);

  useEffect(() => {
    if (screen !== "simulator" && screen !== "pre-model") return;
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
    if (screen !== "simulator" && screen !== "real") return;
    let cancelled = false;
    const baseUrl = useSettingsStore.getState().settings.pipeline.baseUrl;
    pipelineListOnlineModels(baseUrl, { losoOnly: true })
      .then((listed) => {
        if (cancelled) return;
        setLosoModels(listed);
        setRegistryError(null);
        const current = useSettingsStore.getState().settings.pipeline.registry;
        if (listed.length > 0 && !listed.some((item) => item.id === current)) {
          setNested("pipeline", { registry: listed[0].id });
        }
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setRegistryError(
          error instanceof Error ? error.message : "Could not load LOSO models",
        );
      });
    return () => {
      cancelled = true;
    };
  }, [screen, setNested, settings.pipeline.baseUrl]);

  return (
    <section className={styles.panel} aria-labelledby="settings-heading">
      <h2 id="settings-heading" className={styles.title}>
        {title}
      </h2>
      {screen === "pre-model" ? (
        <p className={styles.help}>
          Offline, before epoching: notch, then bandpass. The default band is 8–30 Hz for
          left/right hand imagery. A spatial step can follow that filter.
        </p>
      ) : null}
      {screen === "simulator" ? (
        <p className={styles.help}>
          Activity times come from the dataset. The frozen LOSO pack classifies each
          2 s window and streams left/right probabilities to the game over WebSocket.
        </p>
      ) : null}
      {screen === "real" ? (
        <p className={styles.help}>
          Each trial comes from the session. During activity the pipeline classifies
          2 s windows and drives the avatar through the same WebSocket bridge.
        </p>
      ) : null}

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
          </>
        ) : null}
        {screen === "simulator" || screen === "real" ? (
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
                {losoModels.length === 0 ? (
                  <option value={settings.pipeline.registry}>
                    {settings.pipeline.registry || "No LOSO packs found"}
                  </option>
                ) : (
                  losoModels.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))
                )}
              </select>
            </label>
            {registryError ? <p className={styles.note}>{registryError}</p> : null}
            <label className={styles.row}>
              Control strategy
              <select
                value={settings.pipeline.controlStrategy}
                onChange={(e) =>
                  setNested("pipeline", {
                    controlStrategy:
                      e.target.value as typeof settings.pipeline.controlStrategy,
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
        ) : null}
      </fieldset>

      {screen === "pre-model" ? (
        <>
        <fieldset className={styles.fieldset}>
          <legend>Filter</legend>
          <label className={styles.row}>
            Notch (Hz)
            <input
              type="number"
              min={1}
              step={1}
              value={settings.preModel.notchHz}
              onChange={(e) => setNested("preModel", { notchHz: Number(e.target.value) })}
            />
          </label>
          <label className={styles.row}>
            Low (Hz)
            <input
              type="number"
              min={0.1}
              step={0.5}
              value={settings.preModel.bandLowHz}
              onChange={(e) => setNested("preModel", { bandLowHz: Number(e.target.value) })}
            />
          </label>
          <label className={styles.row}>
            High (Hz)
            <input
              type="number"
              min={0.1}
              step={0.5}
              value={settings.preModel.bandHighHz}
              onChange={(e) => setNested("preModel", { bandHighHz: Number(e.target.value) })}
            />
          </label>
          <label className={styles.row}>
            Notch Q
            <input
              type="number"
              min={0.1}
              step={1}
              value={settings.preModel.notchQ ?? 30}
              onChange={(e) => setNested("preModel", { notchQ: Number(e.target.value) })}
            />
          </label>
          <label className={styles.row}>
            Order
            <input
              type="number"
              min={1}
              max={8}
              step={1}
              value={settings.preModel.order}
              onChange={(e) => setNested("preModel", { order: Number(e.target.value) })}
            />
          </label>
        </fieldset>
        <fieldset className={styles.fieldset}>
          <legend>Spatial</legend>
          <label className={styles.row}>
            Method
            <select
              value={settings.preModel.spatial}
              onChange={(e) =>
                setNested("preModel", {
                  spatial: e.target.value as typeof settings.preModel.spatial,
                })
              }
            >
              <option value="none">None</option>
              <option value="car">CAR</option>
              <option value="hjorth">Hjorth</option>
              <option value="large_laplacian">Large Laplacian</option>
              <option value="spherical_spline">Spherical spline</option>
            </select>
          </label>
          {settings.preModel.spatial === "hjorth" ? (
            <NumberField
              label="Neighborhood (°)"
              value={settings.preModel.hjorthDegrees ?? 36}
              step={1}
              min={1}
              onChange={(value) => setNested("preModel", { hjorthDegrees: value })}
            />
          ) : null}
          {settings.preModel.spatial === "large_laplacian" ? (
            <>
              <NumberField
                label="Radius (°)"
                value={settings.preModel.laplacianRadiusDegrees ?? 70}
                step={1}
                min={1}
                onChange={(value) => setNested("preModel", { laplacianRadiusDegrees: value })}
              />
              <NumberField
                label="Sigma (°)"
                value={settings.preModel.laplacianSigmaDegrees ?? 30}
                step={1}
                min={1}
                onChange={(value) => setNested("preModel", { laplacianSigmaDegrees: value })}
              />
            </>
          ) : null}
          {settings.preModel.spatial === "spherical_spline" ? (
            <>
              <NumberField
                label="Legendre m"
                value={settings.preModel.splineOrder ?? 4}
                step={1}
                min={2}
                onChange={(value) => setNested("preModel", { splineOrder: value })}
              />
              <NumberField
                label="Ridge"
                value={settings.preModel.splineRidge ?? 0.00001}
                step={0.00001}
                min={0}
                onChange={(value) => setNested("preModel", { splineRidge: value })}
              />
            </>
          ) : null}
        </fieldset>
        <fieldset className={styles.fieldset}>
          <legend>Alignment</legend>
          <label className={styles.row}>
            Method
            <select
              value={settings.preModel.alignment}
              onChange={(e) =>
                setNested("preModel", {
                  alignment: e.target.value as typeof settings.preModel.alignment,
                })
              }
            >
              <option value="none">None</option>
              <option value="euclidean">Euclidean alignment</option>
            </select>
          </label>
        </fieldset>
        <fieldset className={styles.fieldset}>
          <legend>Features</legend>
          <label className={styles.row}>
            Method
            <select
              value={settings.preModel.features}
              onChange={(e) =>
                setNested("preModel", {
                  features: e.target.value as typeof settings.preModel.features,
                })
              }
            >
              <option value="none">None</option>
              <option value="ts">TS</option>
              <option value="tdp">TDP</option>
              <option value="psd">PSD</option>
              <option value="csp">CSP</option>
              <option value="fbcsp">FBCSP</option>
            </select>
          </label>
          {settings.preModel.features === "ts" ? (
            <NumberField
              label="Points"
              value={settings.preModel.tsPoints ?? 32}
              step={1}
              min={2}
              onChange={(value) => setNested("preModel", { tsPoints: value })}
            />
          ) : null}
          {settings.preModel.features === "psd" ? (
            <>
              <NumberField label="Theta low (Hz)" value={settings.preModel.thetaLowHz ?? 4} step={0.5} min={0} onChange={(value) => setNested("preModel", { thetaLowHz: value })} />
              <NumberField label="Theta high (Hz)" value={settings.preModel.thetaHighHz ?? 8} step={0.5} min={0} onChange={(value) => setNested("preModel", { thetaHighHz: value })} />
              <NumberField label="Mu low (Hz)" value={settings.preModel.muLowHz ?? 8} step={0.5} min={0} onChange={(value) => setNested("preModel", { muLowHz: value })} />
              <NumberField label="Mu high (Hz)" value={settings.preModel.muHighHz ?? 12} step={0.5} min={0} onChange={(value) => setNested("preModel", { muHighHz: value })} />
              <NumberField label="Beta low (Hz)" value={settings.preModel.betaLowHz ?? 13} step={0.5} min={0} onChange={(value) => setNested("preModel", { betaLowHz: value })} />
              <NumberField label="Beta high (Hz)" value={settings.preModel.betaHighHz ?? 30} step={0.5} min={0} onChange={(value) => setNested("preModel", { betaHighHz: value })} />
            </>
          ) : null}
          {settings.preModel.features === "csp" || settings.preModel.features === "fbcsp" ? (
            <NumberField
              label="CSP pairs"
              value={settings.preModel.cspPairs ?? 2}
              step={1}
              min={1}
              onChange={(value) => setNested("preModel", { cspPairs: value })}
            />
          ) : null}
          {settings.preModel.features === "fbcsp" ? (
            <>
              <NumberField label="Band from (Hz)" value={settings.preModel.fbcspLowHz ?? 4} step={1} min={0} onChange={(value) => setNested("preModel", { fbcspLowHz: value })} />
              <NumberField label="Band to (Hz)" value={settings.preModel.fbcspHighHz ?? 40} step={1} min={1} onChange={(value) => setNested("preModel", { fbcspHighHz: value })} />
              <NumberField label="Band width (Hz)" value={settings.preModel.fbcspBandHz ?? 4} step={1} min={1} onChange={(value) => setNested("preModel", { fbcspBandHz: value })} />
              <NumberField label="Band order" value={settings.preModel.fbcspOrder ?? 4} step={1} min={1} onChange={(value) => setNested("preModel", { fbcspOrder: value })} />
              <NumberField label="Features kept" value={settings.preModel.fbcspKeep ?? 4} step={1} min={1} onChange={(value) => setNested("preModel", { fbcspKeep: value })} />
            </>
          ) : null}
        </fieldset>
        <fieldset className={styles.fieldset}>
          <legend>Classifier</legend>
          <label className={styles.row}>
            Method
            <select
              value={settings.preModel.classifier}
              onChange={(e) => {
                const classifier = e.target.value as typeof settings.preModel.classifier;
                setNested("preModel", { classifier, ...classifierDefaults(classifier) });
              }}
            >
              <option value="none">None</option>
              <option value="lda">LDA</option>
              <option value="slda">Shrinkage LDA</option>
              <option value="logistic_regression">Logistic regression</option>
              <option value="linear_svm">Linear SVM</option>
              <option value="knn">k-NN</option>
              <option value="gaussian_nb">Gaussian Naive Bayes</option>
              <option value="decision_tree">Decision tree</option>
              <option value="svm_rbf">SVM-RBF</option>
              <option value="random_forest">Random forest</option>
              <option value="random_subspace_knn">Random subspace k-NN</option>
              <option value="boosted_trees">Boosted trees</option>
              <option value="bagged_trees">Bagged trees</option>
              <option value="rusboosted_trees">RUSBoosted trees</option>
            </select>
          </label>
          {showsRegularization(settings.preModel.classifier) ? (
            <NumberField
              label="C"
              value={settings.preModel.regularizationC ?? 1}
              step={0.1}
              min={0.0001}
              onChange={(value) => setNested("preModel", { regularizationC: value })}
            />
          ) : null}
          {settings.preModel.classifier === "slda" ? (
            <NumberField
              label="Shrinkage"
              value={settings.preModel.shrinkage ?? ""}
              step={0.05}
              min={0}
              max={1}
              placeholder="auto"
              onChange={(value) =>
                setNested("preModel", { shrinkage: Number.isNaN(value) ? null : value })
              }
            />
          ) : null}
          {showsNeighbors(settings.preModel.classifier) ? (
            <NumberField
              label="Neighbors"
              value={settings.preModel.neighbors ?? 5}
              step={1}
              min={1}
              onChange={(value) => setNested("preModel", { neighbors: value })}
            />
          ) : null}
          {settings.preModel.classifier === "svm_rbf" ? (
            <label className={styles.row}>
              Gamma
              <input
                type="text"
                value={settings.preModel.gamma ?? "scale"}
                onChange={(e) => setNested("preModel", { gamma: e.target.value })}
              />
            </label>
          ) : null}
          {showsDepth(settings.preModel.classifier) ? (
            <NumberField
              label="Max depth"
              value={settings.preModel.maxDepth ?? ""}
              step={1}
              min={1}
              placeholder="none"
              onChange={(value) =>
                setNested("preModel", { maxDepth: Number.isNaN(value) ? null : value })
              }
            />
          ) : null}
          {showsEstimators(settings.preModel.classifier) ? (
            <NumberField
              label="Estimators"
              value={settings.preModel.nEstimators ?? 80}
              step={1}
              min={1}
              onChange={(value) => setNested("preModel", { nEstimators: value })}
            />
          ) : null}
          {settings.preModel.classifier === "random_subspace_knn" ? (
            <NumberField
              label="Max features"
              value={settings.preModel.maxFeatures ?? 0.5}
              step={0.05}
              min={0.05}
              max={1}
              onChange={(value) => setNested("preModel", { maxFeatures: value })}
            />
          ) : null}
          <label className={styles.row}>
            Evaluation
            <select
              value={settings.preModel.evaluation}
              onChange={(e) =>
                setNested("preModel", {
                  evaluation: e.target.value as typeof settings.preModel.evaluation,
                })
              }
            >
              <option value="nested">Nested cross-subject</option>
              <option value="nested_loso">Nested LOSO</option>
              <option value="split">Train / validation / test</option>
              <option value="cross_validation">Cross-validation</option>
              <option value="loso">LOSO</option>
            </select>
          </label>
          {settings.preModel.evaluation === "nested" ||
          settings.preModel.evaluation === "cross_validation" ? (
            <NumberField
              label="Folds"
              value={settings.preModel.folds ?? 5}
              step={1}
              min={2}
              onChange={(value) => setNested("preModel", { folds: value })}
            />
          ) : null}
          {settings.preModel.evaluation === "nested" ||
          settings.preModel.evaluation === "nested_loso" ? (
            <>
              <NumberField
                label="Inner folds"
                value={settings.preModel.innerFolds ?? 3}
                step={1}
                min={2}
                onChange={(value) => setNested("preModel", { innerFolds: value })}
              />
              <NumberField
                label="Search budget"
                value={settings.preModel.searchBudget ?? 4}
                step={1}
                min={1}
                onChange={(value) => setNested("preModel", { searchBudget: value })}
              />
            </>
          ) : null}
          {settings.preModel.evaluation === "split" ? (
            <>
              <NumberField
                label="Test fraction"
                value={settings.preModel.testFraction ?? 0.2}
                step={0.05}
                min={0.05}
                max={0.8}
                onChange={(value) => setNested("preModel", { testFraction: value })}
              />
              <NumberField
                label="Validation fraction"
                value={settings.preModel.validationFraction ?? 0.2}
                step={0.05}
                min={0.05}
                max={0.8}
                onChange={(value) => setNested("preModel", { validationFraction: value })}
              />
            </>
          ) : null}
        </fieldset>
        </>
      ) : null}

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
          Trials (simulator: dataset MI count, e.g. 14)
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

function NumberField({
  label,
  value,
  onChange,
  step,
  min,
  max,
  placeholder,
}: {
  label: string;
  value: number | "";
  onChange: (value: number) => void;
  step?: number;
  min?: number;
  max?: number;
  placeholder?: string;
}) {
  return (
    <label className={styles.row}>
      {label}
      <input
        type="number"
        step={step}
        min={min}
        max={max}
        placeholder={placeholder}
        value={value}
        onChange={(event) => onChange(event.target.value === "" ? Number.NaN : Number(event.target.value))}
      />
    </label>
  );
}

function classifierDefaults(method: string): {
  nEstimators?: number | null;
  maxDepth?: number | null;
} {
  if (method === "random_forest") return { nEstimators: 80, maxDepth: null };
  if (method === "random_subspace_knn") return { nEstimators: 15, maxDepth: null };
  if (method === "boosted_trees") return { nEstimators: 40, maxDepth: 2 };
  if (method === "bagged_trees") return { nEstimators: 25, maxDepth: null };
  if (method === "rusboosted_trees") return { nEstimators: 30, maxDepth: 2 };
  if (method === "decision_tree") return { maxDepth: null };
  return {};
}

function showsRegularization(method: string): boolean {
  return ["logistic_regression", "linear_svm", "svm_rbf"].includes(method);
}

function showsNeighbors(method: string): boolean {
  return method === "knn" || method === "random_subspace_knn";
}

function showsDepth(method: string): boolean {
  return ["decision_tree", "random_forest", "boosted_trees", "bagged_trees", "rusboosted_trees"].includes(
    method,
  );
}

function showsEstimators(method: string): boolean {
  return [
    "random_forest",
    "random_subspace_knn",
    "boosted_trees",
    "bagged_trees",
    "rusboosted_trees",
  ].includes(method);
}
