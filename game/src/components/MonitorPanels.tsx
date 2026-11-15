"use client";

import type { PipelineStageMessage } from "@/lib/pipeline/observe";

import { COMPACT_TRACE_LAYOUT, TraceCanvas } from "./charts/TraceCanvas";
import styles from "./monitor-page.module.css";

export function TracePanel({
  title,
  detail,
  channels,
  samples,
}: {
  title: string;
  detail: string;
  channels: string[];
  samples: number[][];
}) {
  return (
    <figure className={styles.panel}>
      <figcaption>
        <span className={styles.title}>{title}</span>
        <span className={styles.detail}>{detail}</span>
      </figcaption>
      <TraceCanvas
        channels={channels}
        samples={samples}
        label={title}
        layout={COMPACT_TRACE_LAYOUT}
      />
    </figure>
  );
}

export function FeatureBars({ features }: { features: number[] }) {
  const peak = Math.max(1e-6, ...features.map((value) => Math.abs(value)));
  return (
    <figure className={styles.panel}>
      <figcaption>
        <span className={styles.title}>CSP log-var features</span>
        <span className={styles.detail}>{features.length} dims</span>
      </figcaption>
      {features.length === 0 ? (
        <p className={styles.empty}>Waiting for a classified window…</p>
      ) : (
        <div className={styles.bars} aria-label="Feature magnitudes">
          {features.map((value, index) => (
            <div
              key={`f-${index}`}
              className={styles.bar}
              style={{ height: `${Math.max(4, (Math.abs(value) / peak) * 100)}%` }}
              title={`f${index + 1}=${value.toFixed(3)}`}
            />
          ))}
        </div>
      )}
    </figure>
  );
}

export function formatProb(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(3);
}

export function DecisionPanel({
  stage,
  controlLabel,
}: {
  stage: PipelineStageMessage | null;
  controlLabel: string;
}) {
  return (
    <figure className={styles.panel} style={{ marginTop: "0.85rem" }}>
      <figcaption>
        <span className={styles.title}>Decision</span>
        <span className={styles.detail}>
          {stage
            ? `${stage.strategy} · ${stage.reason || "—"} · ${stage.classify_ms.toFixed(1)} ms`
            : "no window yet"}
        </span>
      </figcaption>
      <div className={styles.probs}>
        <div className={styles.prob}>
          <div className={styles.probLabel}>P(left)</div>
          <div className={styles.probValue}>{formatProb(stage?.p_left)}</div>
        </div>
        <div className={styles.prob}>
          <div className={styles.probLabel}>P(right)</div>
          <div className={styles.probValue}>{formatProb(stage?.p_right)}</div>
        </div>
      </div>
      <div className={styles.controlRow}>
        <span>
          label <strong>{stage?.label ?? "—"}</strong>
        </span>
        <span>
          margin <strong>{formatProb(stage?.margin)}</strong>
        </span>
        <span>
          control <strong>{controlLabel}</strong>
        </span>
        <span>
          gate{" "}
          <strong>
            {stage?.gate?.command ?? "—"}
            {stage?.gate?.reason ? ` (${stage.gate.reason})` : ""}
          </strong>
        </span>
        {stage?.emg_rejected ? (
          <span>
            <strong>EMG rejected</strong>
          </span>
        ) : null}
      </div>
    </figure>
  );
}
