"use client";

import { useMemo } from "react";

import { useObserveSession } from "@/hooks/useObserveSession";
import { toRelativeTimeline } from "@/lib/pipeline/monitor-time";
import { stitchFieldAcrossWindows } from "@/lib/pipeline/stage-history";

import { ClassifierHeadsChart } from "./ClassifierHeadsChart";
import {
  DecisionPanel,
  PipelineCascade,
  TimelineLegend,
  TracePanel,
} from "./MonitorPanels";
import styles from "./monitor-page.module.css";

export function MonitorPageClient() {
  const { observeUrl, socketStatus, stage, stages, stream, onlineMeta } =
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

  const streamAbs =
    stream && stream.t_end > stream.t_start
      ? { start: stream.t_start, end: stream.t_end }
      : null;

  const streamTimeline = toRelativeTimeline(
    streamAbs,
    stream?.markers ?? [],
    stream?.windows ?? [],
    stream?.imagery_onset ?? null,
  );
  const eaStitch = stitchFieldAcrossWindows(stages, "aligned");
  const cspStitch = stitchFieldAcrossWindows(stages, "csp_traces");

  const spanLabel = streamTimeline
    ? `view ${streamTimeline.range.end.toFixed(1)} s`
    : "waiting";

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
          <span>
            stages <strong>{stages.length}/5</strong>
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

      <PipelineCascade
        hasStream={Boolean(stream?.raw?.length)}
        hasStage={stages.length > 0}
        stage={stage}
      />

      <TimelineLegend
        markers={streamTimeline?.markers ?? []}
        windows={streamTimeline?.windows ?? []}
        imageryOnset={streamTimeline?.imageryOnset ?? null}
      />

      <div className={styles.grid}>
        <TracePanel
          title="1 · Raw EEG (stream)"
          detail={`${stream?.sample_rate_hz ?? "?"} Hz · ${spanLabel} · live`}
          channels={stream?.channels ?? channels}
          samples={stream?.raw ?? []}
          timeRange={streamTimeline?.range ?? null}
          markers={streamTimeline?.markers ?? []}
          windows={streamTimeline?.windows ?? []}
          imageryOnset={streamTimeline?.imageryOnset ?? null}
        />
        <TracePanel
          title="2 · Filtered EEG (stream)"
          detail="Same clock as raw · ribbons w0–w4 = five 2 s windows"
          channels={stream?.channels ?? channels}
          samples={stream?.filtered ?? []}
          timeRange={streamTimeline?.range ?? null}
          markers={streamTimeline?.markers ?? []}
          windows={streamTimeline?.windows ?? []}
          imageryOnset={streamTimeline?.imageryOnset ?? null}
        />
        <TracePanel
          title="3 · EA-aligned"
          detail={
            eaStitch
              ? `builds across windows (${stages.length}/5) · cue ${stage?.cue ?? "—"}`
              : "fills in as each 2 s window closes"
          }
          channels={eaStitch?.channels ?? channels}
          samples={eaStitch?.samples ?? []}
          timeRange={eaStitch?.range ?? null}
          windows={eaStitch?.windows ?? []}
          imageryOnset={eaStitch?.imageryOnset ?? null}
        />
        <TracePanel
          title="4 · CSP components"
          detail={
            cspStitch
              ? `${cspStitch.channels.join(" · ")} · updates with each window`
              : "CSP projection appears window by window"
          }
          channels={cspStitch?.channels ?? []}
          samples={cspStitch?.samples ?? []}
          timeRange={cspStitch?.range ?? null}
          windows={cspStitch?.windows ?? []}
          imageryOnset={cspStitch?.imageryOnset ?? null}
        />
        <ClassifierHeadsChart stages={stages} />
      </div>

      <DecisionPanel stage={stage} controlLabel={controlLabel} />
    </div>
  );
}
