import type { SessionMode } from "./domain";

export type TargetSpawnMode = "random" | "alternate" | "balanced";

export type SequenceMode = "random_balanced" | "predefined" | "random";

export type SessionSettings = {
  general: {
    fullscreenPlaceholder: boolean;
    audioPlaceholder: boolean;
  };
  offline: {
    fixationDurationMs: number;
    cueDurationMs: number;
    controlDurationMs: number;
    feedbackDurationMs: number;
    itiDurationMs: number;
    autoMoveSpeed: number;
  };
  online: {
    /** Where Phaser subscribes for BCI stream samples in online mode. */
    inputAdapter: "mock" | "window" | "websocket";
    websocketUrl: string;
    bciDebugLogging: boolean;
    wsMaxReconnectAttempts: number;
    wsReconnectBaseMs: number;
    wsReconnectMaxMs: number;
    deadZone: number;
    gain: number;
    smoothing: number;
    maxHorizontalSpeed: number;
    mockSampleIntervalMs: number;
    targetSpawnMode: TargetSpawnMode;
    mockPattern:
      | "strong_left"
      | "strong_right"
      | "noisy"
      | "weak_ambiguous"
      | "alternating";
    mockNoiseLevel: number;
  };
  experimental: {
    numberOfTrials: number;
    blockSize: number;
    randomSeed: number | null;
    sequenceMode: SequenceMode;
  };
  pipeline: {
    baseUrl: string;
    onlineSource: "simulator" | "cyton";
    dataset: string;
    sessionDir: string;
    subject: string;
  };
  preModel: {
    notchHz: number;
    bandLowHz: number;
    bandHighHz: number;
    order: number;
    spatial: "none" | "car" | "hjorth" | "large_laplacian" | "spherical_spline";
    alignment: "none" | "euclidean";
    features: "none" | "ts" | "tdp" | "psd" | "csp" | "fbcsp";
    classifier:
      | "none"
      | "lda"
      | "slda"
      | "logistic_regression"
      | "linear_svm"
      | "knn"
      | "gaussian_nb"
      | "decision_tree"
      | "svm_rbf"
      | "random_forest"
      | "random_subspace_knn"
      | "boosted_trees"
      | "bagged_trees"
      | "rusboosted_trees";
    evaluation: "split" | "cross_validation" | "loso" | "nested" | "nested_loso";
    notchQ: number;
    hjorthDegrees: number;
    laplacianRadiusDegrees: number;
    laplacianSigmaDegrees: number;
    splineOrder: number;
    splineRidge: number;
    tsPoints: number;
    cspPairs: number;
    fbcspLowHz: number;
    fbcspHighHz: number;
    fbcspBandHz: number;
    fbcspOrder: number;
    fbcspKeep: number;
    thetaLowHz: number;
    thetaHighHz: number;
    muLowHz: number;
    muHighHz: number;
    betaLowHz: number;
    betaHighHz: number;
    regularizationC: number;
    neighbors: number;
    gamma: string;
    maxDepth: number | null;
    nEstimators: number | null;
    maxFeatures: number;
    shrinkage: number | null;
    folds: number;
    innerFolds: number;
    searchBudget: number;
    testFraction: number;
    validationFraction: number;
  };
};

export const defaultSessionSettings = (): SessionSettings => ({
  general: {
    fullscreenPlaceholder: false,
    audioPlaceholder: false,
  },
  offline: {
    fixationDurationMs: 800,
    cueDurationMs: 1000,
    controlDurationMs: 4000,
    feedbackDurationMs: 700,
    itiDurationMs: 500,
    autoMoveSpeed: 220,
  },
  online: {
    inputAdapter: "mock",
    websocketUrl: "ws://127.0.0.1:8090",
    bciDebugLogging: false,
    wsMaxReconnectAttempts: 8,
    wsReconnectBaseMs: 1000,
    wsReconnectMaxMs: 30_000,
    deadZone: 0.08,
    gain: 1.2,
    smoothing: 0.25,
    maxHorizontalSpeed: 280,
    mockSampleIntervalMs: 33,
    targetSpawnMode: "balanced",
    mockPattern: "alternating",
    mockNoiseLevel: 0.05,
  },
  experimental: {
    numberOfTrials: 8,
    blockSize: 4,
    randomSeed: 42,
    sequenceMode: "random_balanced",
  },
  pipeline: {
    baseUrl: "http://127.0.0.1:8765",
    onlineSource: "simulator",
    dataset: "shuqfa2024",
    sessionDir: "",
    subject: "",
  },
  preModel: {
    notchHz: 50,
    bandLowHz: 8,
    bandHighHz: 30,
    order: 4,
    spatial: "none",
    alignment: "none",
    features: "none",
    classifier: "none",
    evaluation: "cross_validation",
    notchQ: 30,
    hjorthDegrees: 36,
    laplacianRadiusDegrees: 70,
    laplacianSigmaDegrees: 30,
    splineOrder: 4,
    splineRidge: 0.00001,
    tsPoints: 32,
    cspPairs: 2,
    fbcspLowHz: 4,
    fbcspHighHz: 40,
    fbcspBandHz: 4,
    fbcspOrder: 4,
    fbcspKeep: 4,
    thetaLowHz: 4,
    thetaHighHz: 8,
    muLowHz: 8,
    muHighHz: 12,
    betaLowHz: 13,
    betaHighHz: 30,
    regularizationC: 1,
    neighbors: 5,
    gamma: "scale",
    maxDepth: null,
    nEstimators: 80,
    maxFeatures: 0.5,
    shrinkage: null,
    folds: 5,
    innerFolds: 3,
    searchBudget: 4,
    testFraction: 0.2,
    validationFraction: 0.2,
  },
});

export function settingsWithModeSnapshot(
  settings: SessionSettings,
  mode: SessionMode,
): Record<string, unknown> {
  return { ...settings, mode, snapshotAt: new Date().toISOString() };
}
