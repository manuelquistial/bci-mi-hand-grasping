"use client";

import { useEffect, useRef } from "react";

const TRACE = ["#d6d3d1", "#b7c4a8", "#e7d7c1", "#e7b2a8", "#a8a29e", "#c4b5a0"];

export type TraceLayout = {
  minHeight: number;
  maxHeight: number;
  /** Vertical pixels reserved per channel lane when sizing the canvas. */
  laneHeight: number;
  /** Extra vertical padding added to the computed height and used as top/bottom margin. */
  pad: number;
  /** Left gutter reserved for channel labels. */
  gutter: number;
  fontPx: number;
  labelX: number;
};

export const COMPACT_TRACE_LAYOUT: TraceLayout = {
  minHeight: 160,
  maxHeight: 420,
  laneHeight: 16,
  pad: 12,
  gutter: 48,
  fontPx: 10,
  labelX: 4,
};

export const TALL_TRACE_LAYOUT: TraceLayout = {
  minHeight: 240,
  maxHeight: 720,
  laneHeight: 18,
  pad: 16,
  gutter: 52,
  fontPx: 11,
  labelX: 6,
};

export function drawTraces(
  canvas: HTMLCanvasElement | null,
  channels: string[],
  samples: number[][],
  layout: TraceLayout,
) {
  if (!canvas) return;
  const { pad, gutter } = layout;
  const height = Math.min(
    layout.maxHeight,
    Math.max(layout.minHeight, channels.length * layout.laneHeight + pad),
  );
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
  const lane = (height - pad) / count;
  const plotWidth = width - gutter - 8;
  context.font = `${layout.fontPx}px system-ui, sans-serif`;
  context.textBaseline = "middle";
  for (let channel = 0; channel < count; channel += 1) {
    const series = samples[channel] ?? [];
    const mid = pad / 2 + lane * channel + lane / 2;
    context.strokeStyle = "#44403c";
    context.beginPath();
    context.moveTo(gutter, mid);
    context.lineTo(width - 8, mid);
    context.stroke();
    context.fillStyle = "#a8a29e";
    context.fillText(channels[channel] || `Ch${channel + 1}`, layout.labelX, mid);
    if (series.length < 2) continue;
    let peak = 0;
    for (const value of series) peak = Math.max(peak, Math.abs(value));
    const scale = peak > 0 ? (lane * 0.4) / peak : 1;
    context.strokeStyle = TRACE[channel % TRACE.length];
    context.lineWidth = 1;
    context.beginPath();
    series.forEach((value, index) => {
      const x = gutter + (index / (series.length - 1)) * plotWidth;
      const y = mid - value * scale;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  }
}

/** Multi-channel trace canvas that redraws on data change and container resize. */
export function TraceCanvas({
  channels,
  samples,
  label,
  layout,
}: {
  channels: string[];
  samples: number[][];
  label: string;
  layout: TraceLayout;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const draw = () => drawTraces(canvas, channels, samples, layout);
    draw();
    const parent = canvas?.parentElement;
    if (!parent || typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", draw);
      return () => window.removeEventListener("resize", draw);
    }
    const observer = new ResizeObserver(draw);
    observer.observe(parent);
    return () => observer.disconnect();
  }, [channels, samples, layout]);
  return <canvas ref={canvasRef} aria-label={label} />;
}
