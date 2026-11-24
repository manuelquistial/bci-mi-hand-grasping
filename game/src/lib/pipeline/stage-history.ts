/** Accumulate per-window observe stages and stitch traces across a trial. */

import type { PipelineStageMessage } from "./observe";

const ACTIVITY_SECONDS = 4;

export function upsertStageHistory(
  previous: PipelineStageMessage[],
  next: PipelineStageMessage,
): PipelineStageMessage[] {
  if (previous.length > 0 && previous[0]?.trial_id !== next.trial_id) {
    return [next];
  }
  const without = previous.filter(
    (stage) => stage.window_index !== next.window_index,
  );
  return [...without, next].sort((a, b) => a.window_index - b.window_index);
}

export function stitchFieldAcrossWindows(
  stages: PipelineStageMessage[],
  field: "aligned" | "csp_traces",
  sampleCount = 240,
): {
  channels: string[];
  samples: number[][];
  range: { start: number; end: number };
  windows: { index: number; start: number; end: number }[];
  imageryOnset: number;
} | null {
  if (stages.length === 0) return null;
  const onset = stages.find((stage) => stage.imagery_onset != null)?.imagery_onset;
  if (onset == null) return null;
  const latest = stages[stages.length - 1];
  const channels =
    field === "csp_traces"
      ? latest?.csp_channels ?? []
      : latest?.channels ?? [];
  if (channels.length === 0) return null;

  const grid = channels.map(() => Array.from({ length: sampleCount }, () => 0));
  const filled = channels.map(() =>
    Array.from({ length: sampleCount }, () => false),
  );

  for (const stage of stages) {
    const series = stage[field];
    if (!series.length) continue;
    const start = stage.window_start - onset;
    const end = stage.window_end - onset;
    const span = Math.max(1e-6, end - start);
    for (let channel = 0; channel < channels.length; channel += 1) {
      const row = series[channel] ?? [];
      if (row.length < 2) continue;
      for (let i = 0; i < row.length; i += 1) {
        const t = start + (i / (row.length - 1)) * span;
        const index = Math.round((t / ACTIVITY_SECONDS) * (sampleCount - 1));
        if (index < 0 || index >= sampleCount) continue;
        grid[channel]![index] = row[i] ?? 0;
        filled[channel]![index] = true;
      }
    }
  }

  for (let channel = 0; channel < channels.length; channel += 1) {
    let last = 0;
    let seen = false;
    for (let i = 0; i < sampleCount; i += 1) {
      if (filled[channel]![i]) {
        last = grid[channel]![i] ?? 0;
        seen = true;
      } else if (seen) {
        grid[channel]![i] = last;
      }
    }
  }

  return {
    channels,
    samples: grid,
    range: { start: 0, end: ACTIVITY_SECONDS },
    windows: stages.map((stage) => ({
      index: stage.window_index,
      start: stage.window_start - onset,
      end: stage.window_end - onset,
    })),
    imageryOnset: 0,
  };
}
