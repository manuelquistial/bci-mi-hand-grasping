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
  const parent = canvas.parentElement;
  const pad = parent ? getComputedStyle(parent) : null;
  const inset = pad
    ? Number.parseFloat(pad.paddingLeft) + Number.parseFloat(pad.paddingRight)
    : 0;
  const width = Math.max(160, (parent?.clientWidth ?? 640) - inset);
  const height = Math.min(720, Math.max(240, channels.length * 18 + 16));
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
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
    const draw = () => drawTraces(canvasRef.current, channels, samples);
    draw();
    window.addEventListener("resize", draw);
    return () => window.removeEventListener("resize", draw);
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
    ? `${preview.dataset} · ${preview.subject}`
    : preview.dataset;
  const windowLabel = `${who} · first ${preview.seconds}s`;
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
    </div>
  );
}
