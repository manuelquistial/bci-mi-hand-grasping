"use client";

import type {
  ObserveMarker,
  ObserveWindowBand,
  PipelineStageMessage,
} from "@/lib/pipeline/observe";
import {
  formatRelativeSeconds,
  secondsAfterOnset,
} from "@/lib/pipeline/monitor-time";

import { COMPACT_TRACE_LAYOUT, TraceCanvas } from "./charts/TraceCanvas";
import styles from "./monitor-page.module.css";

export function TracePanel({
  title,
  detail,
  channels,
  samples,
  timeRange = null,
  markers = [],
  windows = [],
  imageryOnset = null,
}: {
  title: string;
  detail: string;
  channels: string[];
  samples: number[][];
  timeRange?: { start: number; end: number } | null;
  markers?: ObserveMarker[];
  windows?: ObserveWindowBand[];
  imageryOnset?: number | null;
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
        timeRange={timeRange}
        markers={markers}
        windows={windows}
        imageryOnset={imageryOnset}
      />
    </figure>
  );
}

export function formatProb(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(3);
}

export function PipelineCascade({
  hasStream,
  hasStage,
  stage,
}: {
  hasStream: boolean;
  hasStage: boolean;
  stage: PipelineStageMessage | null;
}) {
  const lag = secondsAfterOnset(stage?.window_end, stage?.imagery_onset);
  const steps = [
    { id: "1", label: "Raw", ready: hasStream },
    { id: "2", label: "Filter", ready: hasStream },
    { id: "3", label: "EA", ready: hasStage && (stage?.aligned?.length ?? 0) > 0 },
    { id: "4", label: "CSP", ready: hasStage && (stage?.csp_traces?.length ?? 0) > 0 },
    {
      id: "5",
      label: "Heads",
      ready: hasStage && (stage?.predictions?.length ?? 0) > 0,
    },
  ];
  return (
    <figure className={`${styles.panel} ${styles.cascade}`}>
      <figcaption>
        <span className={styles.title}>Pipeline cascade</span>
        <span className={styles.detail}>
          EA/CSP/heads fill in as each 2 s window closes
          {lag == null ? "" : ` · last window closed ${formatRelativeSeconds(lag)} after MI`}
        </span>
      </figcaption>
      <ol className={styles.cascadeList}>
        {steps.map((step, index) => (
          <li
            key={step.id}
            className={step.ready ? styles.cascadeReady : styles.cascadeWait}
          >
            <span className={styles.cascadeStep}>{step.id}</span>
            <span>{step.label}</span>
            {index < steps.length - 1 ? (
              <span className={styles.cascadeArrow} aria-hidden>
                →
              </span>
            ) : null}
          </li>
        ))}
      </ol>
    </figure>
  );
}

export function TimelineLegend({
  markers,
  windows,
  imageryOnset,
}: {
  markers: ObserveMarker[];
  windows: ObserveWindowBand[];
  imageryOnset: number | null;
}) {
  const onset =
    imageryOnset ??
    markers.find((marker) => marker.role === "imagery_onset")?.t ??
    null;
  return (
    <figure className={styles.panel}>
      <figcaption>
        <span className={styles.title}>Timeline</span>
        <span className={styles.detail}>
          Axis 0 = MI onset · windows: 0–2, 0.5–2.5, 1–3, 1.5–3.5, 2–4 s
        </span>
      </figcaption>
      <div className={styles.controlRow}>
        <span>
          MI onset in view{" "}
          <strong>
            {onset == null ? "—" : formatRelativeSeconds(onset)}
          </strong>
        </span>
        <span>
          active windows <strong>{windows.length}</strong>
        </span>
        {markers
          .filter((marker) => marker.role === "imagery_onset")
          .slice(-2)
          .map((marker) => (
            <span key={`${marker.t}-${marker.label}`}>
              {marker.kind}:{marker.label} @ {formatRelativeSeconds(marker.t)}
            </span>
          ))}
      </div>
    </figure>
  );
}

export function DecisionPanel({
  stage,
  controlLabel,
}: {
  stage: PipelineStageMessage | null;
  controlLabel: string;
}) {
  const windowLag = secondsAfterOnset(stage?.window_end, stage?.imagery_onset);
  const windowRel =
    stage && stage.imagery_onset != null
      ? {
          start: stage.window_start - stage.imagery_onset,
          end: stage.window_end - stage.imagery_onset,
        }
      : null;
  return (
    <figure className={styles.panel}>
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
          window vs MI{" "}
          <strong>
            {windowRel
              ? `${formatRelativeSeconds(windowRel.start)} → ${formatRelativeSeconds(windowRel.end)}`
              : "—"}
          </strong>
        </span>
        <span>
          closed after MI{" "}
          <strong>{formatRelativeSeconds(windowLag)}</strong>
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
