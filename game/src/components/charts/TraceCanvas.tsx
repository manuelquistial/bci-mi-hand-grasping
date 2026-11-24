"use client";

import { useEffect, useRef } from "react";

const TRACE = ["#d6d3d1", "#b7c4a8", "#e7d7c1", "#e7b2a8", "#a8a29e", "#c4b5a0"];

export type TraceLayout = {
  minHeight: number;
  maxHeight: number;
  laneHeight: number;
  pad: number;
  gutter: number;
  fontPx: number;
  labelX: number;
  /** Extra bottom strip reserved for the time axis. */
  axisHeight: number;
};

export const COMPACT_TRACE_LAYOUT: TraceLayout = {
  minHeight: 200,
  maxHeight: 480,
  laneHeight: 16,
  pad: 48, // room for five window ribbons above the traces
  gutter: 48,
  fontPx: 10,
  labelX: 4,
  axisHeight: 18,
};

export const TALL_TRACE_LAYOUT: TraceLayout = {
  minHeight: 256,
  maxHeight: 740,
  laneHeight: 18,
  pad: 16,
  gutter: 52,
  fontPx: 11,
  labelX: 6,
  axisHeight: 20,
};

export type TraceMarker = {
  t: number;
  label?: string;
  role?: string;
};

export type TraceWindow = {
  index: number;
  start: number;
  end: number;
};

export type TraceTimeRange = {
  start: number;
  end: number;
};

function xForTime(
  t: number,
  range: TraceTimeRange,
  gutter: number,
  plotWidth: number,
): number {
  const span = Math.max(1e-6, range.end - range.start);
  const ratio = (t - range.start) / span;
  return gutter + Math.min(1, Math.max(0, ratio)) * plotWidth;
}

function formatTime(value: number): string {
  // View-relative axis: left edge is 0 after monitor-time normalization.
  const sign = value < 0 ? "-" : "+";
  return `${sign}${Math.abs(value).toFixed(1)}s`;
}

export function drawTraces(
  canvas: HTMLCanvasElement | null,
  channels: string[],
  samples: number[][],
  layout: TraceLayout,
  timeRange?: TraceTimeRange | null,
  markers: TraceMarker[] = [],
  windows: TraceWindow[] = [],
  imageryOnset: number | null = null,
) {
  if (!canvas) return;
  const { pad, gutter, axisHeight } = layout;
  const plotPad = pad + axisHeight;
  const height = Math.min(
    layout.maxHeight,
    Math.max(layout.minHeight, channels.length * layout.laneHeight + plotPad),
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
  const plotBottom = height - axisHeight;
  const lane = (plotBottom - pad) / count;
  const plotWidth = width - gutter - 8;
  const range =
    timeRange && timeRange.end > timeRange.start
      ? timeRange
      : { start: 0, end: 1 };

  // Draw each of the five 2 s windows as its own thin ribbon so overlaps stay readable.
  if (windows.length > 0) {
    const ribbon = Math.min(10, Math.max(6, (pad / 2 + 2) / Math.max(1, windows.length)));
    const colors = ["#38bdf8", "#22d3ee", "#2dd4bf", "#a3e635", "#fbbf24"];
    windows.forEach((windowBand, order) => {
      const x0 = xForTime(windowBand.start, range, gutter, plotWidth);
      const x1 = xForTime(windowBand.end, range, gutter, plotWidth);
      const y = 2 + order * ribbon;
      const color = colors[windowBand.index % colors.length];
      context.fillStyle = color;
      context.globalAlpha = 0.55;
      context.fillRect(x0, y, Math.max(2, x1 - x0), ribbon - 1);
      context.globalAlpha = 1;
      context.fillStyle = color;
      context.font = `${layout.fontPx}px system-ui, sans-serif`;
      context.textBaseline = "middle";
      context.fillText(`w${windowBand.index}`, Math.min(x0 + 2, width - 28), y + (ribbon - 1) / 2);
      // Light vertical guides at window edges across the traces.
      context.strokeStyle = color;
      context.globalAlpha = 0.25;
      context.beginPath();
      context.moveTo(x0, pad / 2);
      context.lineTo(x0, plotBottom);
      context.moveTo(x1, pad / 2);
      context.lineTo(x1, plotBottom);
      context.stroke();
      context.globalAlpha = 1;
    });
  }

  const onset = imageryOnset ?? markers.find((m) => m.role === "imagery_onset")?.t;
  if (onset != null && onset >= range.start && onset <= range.end) {
    const x = xForTime(onset, range, gutter, plotWidth);
    context.strokeStyle = "#fbbf24";
    context.lineWidth = 1.5;
    context.setLineDash([4, 3]);
    context.beginPath();
    context.moveTo(x, pad / 2);
    context.lineTo(x, plotBottom);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = "#fbbf24";
    context.font = `${layout.fontPx}px system-ui, sans-serif`;
    context.textBaseline = "top";
    context.fillText("MI onset", x + 3, pad / 2 + 2);
  }

  for (const marker of markers) {
    if (marker.role === "imagery_onset") continue;
    if (marker.t < range.start || marker.t > range.end) continue;
    const x = xForTime(marker.t, range, gutter, plotWidth);
    context.strokeStyle =
      marker.role === "rest"
        ? "#a8a29e"
        : marker.role === "imagery_end"
          ? "#fb7185"
          : "#d6d3d1";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(x, pad / 2);
    context.lineTo(x, plotBottom);
    context.stroke();
  }

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

  context.strokeStyle = "#57534e";
  context.beginPath();
  context.moveTo(gutter, plotBottom);
  context.lineTo(width - 8, plotBottom);
  context.stroke();
  context.fillStyle = "#a8a29e";
  context.textBaseline = "top";
  const ticks = 5;
  for (let i = 0; i < ticks; i += 1) {
    const ratio = i / (ticks - 1);
    const t = range.start + ratio * (range.end - range.start);
    const x = gutter + ratio * plotWidth;
    context.strokeStyle = "#44403c";
    context.beginPath();
    context.moveTo(x, plotBottom);
    context.lineTo(x, plotBottom + 4);
    context.stroke();
    const label = formatTime(t);
    const metrics = context.measureText(label);
    const textX = Math.min(
      width - 8 - metrics.width,
      Math.max(gutter, x - metrics.width / 2),
    );
    context.fillStyle = "#a8a29e";
    context.fillText(label, textX, plotBottom + 5);
  }
}

/** Multi-channel trace canvas that redraws on data change and container resize. */
export function TraceCanvas({
  channels,
  samples,
  label,
  layout,
  timeRange = null,
  markers = [],
  windows = [],
  imageryOnset = null,
}: {
  channels: string[];
  samples: number[][];
  label: string;
  layout: TraceLayout;
  timeRange?: TraceTimeRange | null;
  markers?: TraceMarker[];
  windows?: TraceWindow[];
  imageryOnset?: number | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Keep dependency slots scalar — nested sample arrays once poisoned HMR/React.
  const channelKey = channels.join("|");
  const sampleKey = `${samples.length}:${samples[0]?.length ?? 0}:${samples[0]?.[0] ?? 0}:${samples[0]?.[(samples[0]?.length ?? 1) - 1] ?? 0}`;
  const markerKey = `${markers.length}:${markers[0]?.t ?? ""}:${markers[markers.length - 1]?.t ?? ""}`;
  const windowKey = `${windows.length}:${windows[0]?.start ?? ""}:${windows[windows.length - 1]?.end ?? ""}`;
  const rangeKey = `${timeRange?.start ?? ""}:${timeRange?.end ?? ""}`;
  useEffect(() => {
    const canvas = canvasRef.current;
    const draw = () =>
      drawTraces(
        canvas,
        channels,
        samples,
        layout,
        timeRange,
        markers,
        windows,
        imageryOnset,
      );
    draw();
    const parent = canvas?.parentElement;
    if (!parent || typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", draw);
      return () => window.removeEventListener("resize", draw);
    }
    const observer = new ResizeObserver(draw);
    observer.observe(parent);
    return () => observer.disconnect();
    // channels/samples/markers/windows are read inside draw; keys above track changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [channelKey, sampleKey, markerKey, windowKey, rangeKey, imageryOnset, layout]);
  return <canvas ref={canvasRef} aria-label={label} />;
}
