"use client";

import { useMemo } from "react";

import { useObserveSession } from "@/hooks/useObserveSession";

import { DecisionPanel, FeatureBars, TracePanel } from "./MonitorPanels";
import styles from "./monitor-page.module.css";

export function MonitorPageClient() {
  const { observeUrl, socketStatus, stage, stream, onlineMeta } =
    useObserveSession();

  const channels = stage?.channels ?? stream?.channels ?? [];
  const statusClass =
    socketStatus === "open"
      ? styles.dotOpen
      : socketStatus === "error"
        ? styles.dotError
        : styles.dotClosed;

  const controlLabel = useMemo(() => {
    if (!stage) return "—";
    const sign = stage.control >= 0 ? "+" : "";
    return `${sign}${stage.control.toFixed(3)} (${stage.valid ? "valid" : "hold"})`;
  }, [stage]);

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Online monitor</h1>
        <div className={styles.meta}>
          <span>
            <span className={`${styles.dot} ${statusClass}`} />
            {socketStatus} · {observeUrl}
          </span>
          <span>
            source <strong>{onlineMeta.source ?? "—"}</strong>
          </span>
          <span>
            samples <strong>{onlineMeta.samples ?? "—"}</strong>
          </span>
          <span>
            control windows <strong>{onlineMeta.controlSamples ?? "—"}</strong>
          </span>
          {onlineMeta.registry ? (
            <span>
              registry <strong>{onlineMeta.registry}</strong>
            </span>
          ) : null}
          {onlineMeta.error ? (
            <span>
              error <strong>{onlineMeta.error}</strong>
            </span>
          ) : null}
        </div>
      </header>

      <div className={styles.grid}>
        <TracePanel
          title="Raw EEG (stream)"
          detail={stream ? `${stream.sample_rate_hz} Hz` : "waiting"}
          channels={stream?.channels ?? channels}
          samples={stream?.raw ?? []}
        />
        <TracePanel
          title="Filtered EEG (stream)"
          detail="causal notch+bandpass"
          channels={stream?.channels ?? channels}
          samples={stream?.filtered ?? []}
        />
        <TracePanel
          title="Window filtered"
          detail={
            stage
              ? `trial ${stage.trial_id} · w${stage.window_index}`
              : "waiting for imagery"
          }
          channels={stage?.channels ?? channels}
          samples={stage?.filtered ?? []}
        />
        <TracePanel
          title="EA-aligned"
          detail={stage?.cue ? `cue ${stage.cue}` : "EA reference"}
          channels={stage?.channels ?? channels}
          samples={stage?.aligned ?? []}
        />
        <TracePanel
          title="CSP components"
          detail={stage?.csp_channels?.join(" · ") || "frozen filters"}
          channels={stage?.csp_channels ?? []}
          samples={stage?.csp_traces ?? []}
        />
        <FeatureBars features={stage?.features ?? []} />
      </div>

      <DecisionPanel stage={stage} controlLabel={controlLabel} />
    </div>
  );
}
