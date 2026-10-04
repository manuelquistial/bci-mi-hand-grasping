/** Pure helpers for summaries / analysis */

export function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export function maxAbs(values: number[]): number {
  if (values.length === 0) return 0;
  return Math.max(...values.map((v) => Math.abs(v)));
}

export function countSuccess(outcomes: string[]): number {
  return outcomes.filter((o) => o === "success").length;
}
