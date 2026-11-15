"use client";

import type { FilterPreview } from "@/lib/pipeline/client";

import { TALL_TRACE_LAYOUT, TraceCanvas } from "./charts/TraceCanvas";
import styles from "./filter-charts.module.css";

function Chart({
  title,
  detail,
  channels,
  samples,
  label,
}: {
  title: string;
  detail: string;
  channels: string[];
  samples: number[][];
  label: string;
}) {
  return (
    <figure className={styles.chart}>
      <figcaption>
        <span className={styles.title}>{title}</span>
        <span className={styles.detail}>{detail}</span>
      </figcaption>
      <TraceCanvas
        channels={channels}
        samples={samples}
        label={label}
        layout={TALL_TRACE_LAYOUT}
      />
    </figure>
  );
}

export function FilterCharts({ preview }: { preview: FilterPreview }) {
  const who = preview.subject
    ? `${preview.dataset} · example ${preview.subject}`
    : preview.dataset;
  const windowLabel = `${who} · first ${preview.seconds}s`;
  const spatial = preview.spatial !== "none" && preview.spatialSignal.length > 0;
  return (
    <div className={styles.charts}>
      <Chart
        title="Raw"
        detail={windowLabel}
        channels={preview.channels}
        samples={preview.raw}
        label="Raw signal"
      />
      <Chart
        title="Filtered"
        detail={`notch ${preview.notchHz} Hz, then ${preview.bandLowHz}–${preview.bandHighHz} Hz`}
        channels={preview.channels}
        samples={preview.filtered}
        label="Filtered signal"
      />
      {spatial ? (
        <Chart
          title={spatialTitle(preview.spatial)}
          detail={preview.spatialChannels.length + " channels"}
          channels={preview.spatialChannels}
          samples={preview.spatialSignal}
          label="Spatial signal"
        />
      ) : null}
      {preview.alignment === "euclidean" && preview.alignmentSignal.length > 0 ? (
        <Chart
          title="Euclidean alignment"
          detail={`${preview.alignmentTrials} trials`}
          channels={preview.alignmentChannels}
          samples={preview.alignmentSignal}
          label="Aligned signal"
        />
      ) : null}
      {preview.features !== "none" && preview.featureSignal.length > 0 ? (
        <Chart
          title={featureTitle(preview.features)}
          detail={preview.featureDetail}
          channels={preview.featureChannels}
          samples={preview.featureSignal}
          label="Feature signal"
        />
      ) : null}
      {preview.classifier !== "none" && preview.confusionMatrix.length > 0 ? (
        <MetricsPanel
          title={classifierTitle(preview.classifier)}
          detail={preview.classifierDetail}
          labels={preview.confusionLabels}
          matrix={preview.confusionMatrix}
          metrics={preview.metrics}
        />
      ) : null}
    </div>
  );
}

function classifierTitle(method: string): string {
  if (method === "lda") return "LDA";
  if (method === "slda") return "Shrinkage LDA";
  if (method === "logistic_regression") return "Logistic regression";
  if (method === "linear_svm") return "Linear SVM";
  if (method === "knn") return "k-NN";
  if (method === "gaussian_nb") return "Gaussian Naive Bayes";
  if (method === "decision_tree") return "Decision tree";
  if (method === "svm_rbf") return "SVM-RBF";
  if (method === "random_forest") return "Random forest";
  if (method === "random_subspace_knn") return "Random subspace k-NN";
  if (method === "boosted_trees") return "Boosted trees";
  if (method === "bagged_trees") return "Bagged trees";
  if (method === "rusboosted_trees") return "RUSBoosted trees";
  return method;
}

const METRIC_ORDER = [
  ["balanced_accuracy", "Balanced accuracy"],
  ["accuracy", "Accuracy"],
  ["sensitivity", "Sensitivity"],
  ["specificity", "Specificity"],
  ["precision", "Precision"],
  ["recall", "Recall"],
  ["f1", "F1"],
  ["kappa", "Kappa"],
  ["auc", "AUC"],
  ["mcc", "MCC"],
] as const;

const PERCENT = new Set([
  "balanced_accuracy",
  "accuracy",
  "sensitivity",
  "specificity",
  "precision",
  "recall",
  "f1",
  "auc",
]);

function MetricsPanel({
  title,
  detail,
  labels,
  matrix,
  metrics,
}: {
  title: string;
  detail: string;
  labels: string[];
  matrix: number[][];
  metrics: Record<string, number | null>;
}) {
  return (
    <figure className={styles.chart}>
      <figcaption>
        <span className={styles.title}>{title}</span>
        <span className={styles.detail}>{detail}</span>
      </figcaption>
      <dl className={styles.metrics}>
        {METRIC_ORDER.map(([key, label]) => (
          <div key={key}>
            <dt>{label}</dt>
            <dd>{formatMetric(key, metrics[key])}</dd>
          </div>
        ))}
      </dl>
      <table className={styles.matrix}>
        <caption>Confusion matrix</caption>
        <thead>
          <tr>
            <th scope="col">True</th>
            {labels.map((label) => (
              <th key={label} scope="col">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, index) => (
            <tr key={labels[index] ?? index}>
              <th scope="row">{labels[index]}</th>
              {row.map((count, column) => (
                <td key={`${labels[index]}-${labels[column]}`}>{count}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}

function formatMetric(name: string, value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (PERCENT.has(name)) return `${Math.round(value * 100)}%`;
  return value.toFixed(2);
}

function featureTitle(method: string): string {
  if (method === "ts") return "TS";
  if (method === "tdp") return "TDP";
  if (method === "psd") return "PSD";
  if (method === "csp") return "CSP";
  if (method === "fbcsp") return "FBCSP";
  return method;
}

function spatialTitle(method: string): string {
  if (method === "car") return "CAR";
  if (method === "hjorth") return "Hjorth";
  if (method === "large_laplacian") return "Large Laplacian";
  if (method === "spherical_spline") return "Spherical spline";
  return method;
}
