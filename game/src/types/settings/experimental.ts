export type SequenceMode = "random_balanced" | "predefined" | "random";

export type ExperimentalSettings = {
  numberOfTrials: number;
  blockSize: number;
  randomSeed: number | null;
  sequenceMode: SequenceMode;
};

export const defaultExperimentalSettings = (): ExperimentalSettings => ({
  /** Simulator slave: expected dataset MI trials (Shuqfa S001 = 14). */
  numberOfTrials: 14,
  blockSize: 4,
  randomSeed: 42,
  sequenceMode: "random_balanced",
});
