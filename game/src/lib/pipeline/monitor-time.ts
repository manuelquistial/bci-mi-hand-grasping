/** Convert absolute LSL times to a view-relative axis starting at 0. */

import type { ObserveMarker, ObserveWindowBand } from "./observe";

export type AbsoluteRange = { start: number; end: number };

export type RelativeTimeline = {
  range: AbsoluteRange;
  markers: ObserveMarker[];
  windows: ObserveWindowBand[];
  imageryOnset: number | null;
  origin: number;
};

export function toRelativeTimeline(
  range: AbsoluteRange | null,
  markers: ObserveMarker[],
  windows: ObserveWindowBand[],
  imageryOnset: number | null,
  /** Prefer MI onset as 0 so the five 2 s windows land on +0…+4 s. */
  originOverride?: number | null,
): RelativeTimeline | null {
  if (!range || !(range.end > range.start)) return null;
  const origin =
    originOverride != null
      ? originOverride
      : imageryOnset != null
        ? imageryOnset
        : range.start;
  const shift = (t: number) => t - origin;
  return {
    origin,
    range: { start: range.start - origin, end: range.end - origin },
    markers: markers.map((marker) => ({ ...marker, t: shift(marker.t) })),
    windows: windows.map((window) => ({
      ...window,
      start: shift(window.start),
      end: shift(window.end),
    })),
    imageryOnset:
      imageryOnset == null ? null : shift(imageryOnset),
  };
}

export function formatRelativeSeconds(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  const sign = value < 0 ? "-" : "+";
  return `${sign}${Math.abs(value).toFixed(2)} s`;
}

export function secondsAfterOnset(
  absolute: number | null | undefined,
  onset: number | null | undefined,
): number | null {
  if (absolute == null || onset == null) return null;
  return absolute - onset;
}
