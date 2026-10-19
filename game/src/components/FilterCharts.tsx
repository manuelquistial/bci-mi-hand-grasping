"use client";

import { useEffect, useRef } from "react";

import type { FilterPreview } from "@/lib/pipeline/client";

import styles from "./filter-charts.module.css";

const TRACE = ["#d6d3d1", "#b7c4a8", "#e7d7c1", "#e7b2a8", "#a8a29e", "#c4b5a0"];

function drawTraces(
  canvas: HTMLCanvasElement | null,
  channels: string[],
  samples: number[][],
) {
  if (!canvas) return;
  const height = Math.min(720, Math.max(240, channels.length * 18 + 16));
  canvas.style.width = "100%";
  canvas.style.height = `${height}px`;
  const width = Math.max(160, canvas.clientWidth);
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#1c1917";
  context.fillRect(0, 0, width, height);
  if (samples.length === 0) return;
  const count = channels.length || samples.length;
  const lane = (height - 16) / count;
  const left = 52;
  const plotWidth = width - left - 8;
  context.font = "11px system-ui, sans-serif";
  context.textBaseline = "middle";
  for (let channel = 0; channel < count; channel += 1) {
    const series = samples[channel] ?? [];
    const mid = 8 + lane * channel + lane / 2;
    context.strokeStyle = "#44403c";
    context.beginPath();
    context.moveTo(left, mid);
    context.lineTo(width - 8, mid);
    context.stroke();
    context.fillStyle = "#a8a29e";
    context.fillText(channels[channel] || `Ch${channel + 1}`, 6, mid);
    if (series.length < 2) continue;
    let peak = 0;
    for (const value of series) peak = Math.max(peak, Math.abs(value));
    const scale = peak > 0 ? (lane * 0.4) / peak : 1;
    context.strokeStyle = TRACE[channel % TRACE.length];
    context.lineWidth = 1;
    context.beginPath();
    series.forEach((value, index) => {
      const x = left + (index / (series.length - 1)) * plotWidth;
      const y = mid - value * scale;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  }
}

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
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const draw = () => drawTraces(canvas, channels, samples);
    draw();
    const parent = canvas?.parentElement;
    if (!parent || typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", draw);
      return () => window.removeEventListener("resize", draw);
    }
    const observer = new ResizeObserver(draw);
    observer.observe(parent);
    return () => observer.disconnect();
  }, [channels, samples]);
  return (
    <figure className={styles.chart}>
      <figcaption>
        <span className={styles.title}>{title}</span>
        <span className={styles.detail}>{detail}</span>
      </figcaption>
      <canvas ref={canvasRef} aria-label={label} />
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
  if (method === "riemann") return "Riemannian SVM";
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
