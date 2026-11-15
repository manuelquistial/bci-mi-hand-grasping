export type OfflineSettings = {
  fixationDurationMs: number;
  cueDurationMs: number;
  controlDurationMs: number;
  feedbackDurationMs: number;
  itiDurationMs: number;
  autoMoveSpeed: number;
};

export const defaultOfflineSettings = (): OfflineSettings => ({
  fixationDurationMs: 800,
  cueDurationMs: 1000,
  controlDurationMs: 4000,
  feedbackDurationMs: 700,
  itiDurationMs: 500,
  autoMoveSpeed: 220,
});
