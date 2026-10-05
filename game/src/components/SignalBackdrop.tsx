"use client";

import { useEffect, useRef } from "react";

import { pipelineSignal, type PipelineSignal } from "@/lib/pipeline/client";

import styles from "./game-page.module.css";

const TRACE = [
  "#b7c4a8",
  "#e7d7c1",
  "#e7b2a8",
  "#d6c4a8",
  "#c9d4c2",
  "#e6cbb3",
  "#c4b7a8",
  "#d4cfc4",
];

const BACKGROUND = "#1c1917";
const LABEL = "#a8a29e";
const GRID = "#44403c";

function drawSignal(canvas: HTMLCanvasElement | null, frame: PipelineSignal | null) {
  if (!canvas) return;
  const parent = canvas.parentElement;
  if (!parent) return;
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, parent.clientWidth);
  const height = Math.max(1, parent.clientHeight);
  if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) {
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
  }
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.fillStyle = BACKGROUND;
  context.fillRect(0, 0, width, height);

  const channels = frame?.samples ?? [];
  const names = frame?.channels ?? [];
  if (!frame?.active || channels.length === 0 || channels.every((row) => row.length === 0)) {
    context.fillStyle = LABEL;
    context.font = "16px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText("Waiting for the signal…", width / 2, height / 2);
    return;
  }

  const labelWidth = 56;
  const pad = 12;
  const band = (height - pad * 2) / channels.length;
  channels.forEach((series, index) => {
    const mid = pad + band * index + band / 2;
    context.strokeStyle = GRID;
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(labelWidth, mid);
    context.lineTo(width - pad, mid);
    context.stroke();

    context.fillStyle = LABEL;
    context.font = "12px ui-monospace, monospace";
    context.textAlign = "left";
    context.textBaseline = "middle";
    context.fillText(names[index] || `Ch${index + 1}`, 10, mid);

    if (series.length === 0) return;
    let mean = 0;
    for (const value of series) mean += value;
    mean /= series.length;
    let deviation = 0;
    for (const value of series) deviation = Math.max(deviation, Math.abs(value - mean));
    const span = Math.max(deviation * 1.25, 1e-6);
    const amplitude = band * 0.42;
    context.strokeStyle = TRACE[index % TRACE.length];
    context.lineWidth = 1.4;
    context.beginPath();
    series.forEach((value, sample) => {
      const x =
        series.length === 1
          ? labelWidth
          : labelWidth + (sample / (series.length - 1)) * (width - labelWidth - pad);
      const y = mid - ((value - mean) / span) * amplitude;
      if (sample === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  });

  const plotLeft = labelWidth;
  const plotWidth = width - labelWidth - pad;
  for (const marker of frame.markers ?? []) {
    const x = plotLeft + Math.min(1, Math.max(0, marker.position)) * plotWidth;
    const color = markerColor(marker.kind, marker.label);
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(x, pad);
    context.lineTo(x, height - pad);
    context.stroke();
    context.fillStyle = color;
    context.font = "13px system-ui, sans-serif";
    context.textAlign = "left";
    context.textBaseline = "top";
    context.fillText(marker.label, Math.min(x + 6, width - 72), pad);
  }
}

function markerColor(kind: string, label: string): string {
  const key = label.trim().toLowerCase();
  if (key === "left" || key === "t1") return "#8fbf7a";
  if (key === "right" || key === "t2") return "#e7b2a8";
  if (kind === "rest" || key === "rest" || key === "t0") return "#d6d3d1";
  return "#e7d7c1";
}

export function SignalBackdrop({ baseUrl }: { baseUrl: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<PipelineSignal | null>(null);

  useEffect(() => {
    let stopped = false;
    const paint = () => drawSignal(canvasRef.current, frameRef.current);
    const tick = async () => {
      try {
        const frame = await pipelineSignal(baseUrl);
        if (stopped) return;
        frameRef.current = frame;
        paint();
      } catch {
        paint();
      }
    };
    void tick();
    const timer = window.setInterval(() => void tick(), 100);
    const parent = canvasRef.current?.parentElement;
    const observer = parent ? new ResizeObserver(paint) : null;
    if (parent) observer?.observe(parent);
    return () => {
      stopped = true;
      window.clearInterval(timer);
      observer?.disconnect();
    };
  }, [baseUrl]);

  return (
    <div className={styles.signal} aria-hidden="true">
      <canvas ref={canvasRef} />
    </div>
  );
}
