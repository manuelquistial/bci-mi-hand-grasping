"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import {
  connectObserveSocket,
  fetchOnlineStatus,
  observeWsUrlFromControl,
  type PipelineStageMessage,
  type StreamFrameMessage,
} from "@/lib/pipeline/observe";
import { useSettingsStore } from "@/state/settings-store";

import styles from "./monitor-page.module.css";

const TRACE = ["#d6d3d1", "#b7c4a8", "#e7d7c1", "#e7b2a8", "#a8a29e", "#c4b5a0"];

function drawTraces(
  canvas: HTMLCanvasElement | null,
  channels: string[],
  samples: number[][],
) {
  if (!canvas) return;
  const height = Math.min(420, Math.max(160, channels.length * 16 + 12));
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
  const lane = (height - 12) / count;
  const left = 48;
  const plotWidth = width - left - 8;
  context.font = "10px system-ui, sans-serif";
  context.textBaseline = "middle";
  for (let channel = 0; channel < count; channel += 1) {
    const series = samples[channel] ?? [];
    const mid = 6 + lane * channel + lane / 2;
    context.strokeStyle = "#44403c";
    context.beginPath();
    context.moveTo(left, mid);
    context.lineTo(width - 8, mid);
    context.stroke();
    context.fillStyle = "#a8a29e";
    context.fillText(channels[channel] || `Ch${channel + 1}`, 4, mid);
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

function TracePanel({
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
    <figure className={styles.panel}>
      <figcaption>
        <span className={styles.title}>{title}</span>
        <span className={styles.detail}>{detail}</span>
      </figcaption>
      <canvas ref={canvasRef} aria-label={title} />
    </figure>
  );
}

function FeatureBars({ features }: { features: number[] }) {
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

function formatProb(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(3);
}

function readMonitorQuery(): { baseUrl: string; controlPort: number } {
  if (typeof window === "undefined") {
    return { baseUrl: "http://127.0.0.1:8765", controlPort: 8090 };
  }
  const params = new URLSearchParams(window.location.search);
  const fromStore = useSettingsStore.getState().settings.pipeline;
  const baseUrl =
    params.get("baseUrl") || fromStore.baseUrl || "http://127.0.0.1:8765";
  const controlPort = Number(
    params.get("controlPort") || fromStore.controlWsPort || 8090,
  );
  return {
    baseUrl,
    controlPort: Number.isFinite(controlPort) ? controlPort : 8090,
  };
}

export function MonitorPageClient() {
  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8765");
  const [observeUrl, setObserveUrl] = useState("ws://127.0.0.1:8091");
  const [socketStatus, setSocketStatus] = useState<
    "connecting" | "open" | "closed" | "error"
  >("connecting");

  useEffect(() => {
    const query = readMonitorQuery();
    setBaseUrl(query.baseUrl);
    setObserveUrl(`ws://127.0.0.1:${query.controlPort + 1}`);
  }, []);
  const [stage, setStage] = useState<PipelineStageMessage | null>(null);
  const [stream, setStream] = useState<StreamFrameMessage | null>(null);
  const [onlineMeta, setOnlineMeta] = useState<{
    source?: string | null;
    samples?: number;
    controlSamples?: number | null;
    registry?: string | null;
    error?: string | null;
  }>({});

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const online = await fetchOnlineStatus(baseUrl);
        if (cancelled) return;
        if (!online) {
          setOnlineMeta({});
          return;
        }
        setOnlineMeta({
          source: online.source,
          samples: online.samples_seen,
          controlSamples: online.control_samples,
          registry: online.registry,
          error: online.error,
        });
        if (online.observe_ws_url) {
          setObserveUrl(online.observe_ws_url);
        } else if (online.control_ws_url) {
          setObserveUrl(observeWsUrlFromControl(online.control_ws_url));
        }
        if (online.last_pipeline_stage) {
          setStage((prev) => prev ?? online.last_pipeline_stage ?? null);
        }
        if (online.last_stream_frame) {
          setStream((prev) => prev ?? online.last_stream_frame ?? null);
        }
      } catch {
        if (!cancelled) {
          setOnlineMeta((prev) => ({ ...prev, error: "status unreachable" }));
        }
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 2000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [baseUrl]);

  useEffect(() => {
    const connection = connectObserveSocket(observeUrl, {
      onMessage: (message) => {
        if (message.type === "pipeline_stage") setStage(message);
        else setStream(message);
      },
      onStatus: setSocketStatus,
    });
    return () => connection.close();
  }, [observeUrl]);

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
    </div>
  );
}
