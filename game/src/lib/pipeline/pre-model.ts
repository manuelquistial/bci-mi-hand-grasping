import type { SessionSettings } from "@/types/settings";

import { postJson } from "./http";

export type FilterPreview = {
  dataset: string;
  subject: string;
  participants: number;
  sampleRateHz: number;
  seconds: number;
  channels: string[];
  notchHz: number;
  bandLowHz: number;
  bandHighHz: number;
  order: number;
  spatial: string;
  spatialChannels: string[];
  spatialSignal: number[][];
  alignment: string;
  alignmentTrials: number;
  alignmentChannels: string[];
  alignmentSignal: number[][];
  features: string;
  featureDetail: string;
  featureChannels: string[];
  featureSignal: number[][];
  classifier: string;
  evaluation: string;
  accuracy: number | null;
  classifierDetail: string;
  classifierFolds: { name: string; accuracy: number }[];
  metrics: Record<string, number | null>;
  confusionLabels: string[];
  confusionMatrix: number[][];
  raw: number[][];
  filtered: number[][];
};

function metricMap(value: unknown): Record<string, number | null> {
  if (!value || typeof value !== "object") return {};
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>).map(([key, item]) => [
      key,
      item == null ? null : Number(item),
    ]),
  );
}

export function pipelinePreModelFilter(
  settings: SessionSettings,
): Promise<FilterPreview> {
  const filter = settings.preModel;
  return postJson(settings.pipeline.baseUrl, "/v1/pre-model/filter", {
    dataset: settings.pipeline.dataset,
    session_dir: settings.pipeline.sessionDir,
    notch_hz: filter.notchHz,
    band_low_hz: filter.bandLowHz,
    band_high_hz: filter.bandHighHz,
    order: filter.order,
    spatial: filter.spatial,
    alignment: filter.alignment,
    features: filter.features,
    classifier: filter.classifier,
    evaluation: filter.evaluation,
    hyperparameters: {
      notch_q: filter.notchQ,
      hjorth_degrees: filter.hjorthDegrees,
      laplacian_radius_degrees: filter.laplacianRadiusDegrees,
      laplacian_sigma_degrees: filter.laplacianSigmaDegrees,
      spline_order: filter.splineOrder,
      spline_ridge: filter.splineRidge,
      ts_points: filter.tsPoints,
      csp_pairs: filter.cspPairs,
      fbcsp_low_hz: filter.fbcspLowHz,
      fbcsp_high_hz: filter.fbcspHighHz,
      fbcsp_band_hz: filter.fbcspBandHz,
      fbcsp_order: filter.fbcspOrder,
      fbcsp_keep: filter.fbcspKeep,
      theta_low_hz: filter.thetaLowHz,
      theta_high_hz: filter.thetaHighHz,
      mu_low_hz: filter.muLowHz,
      mu_high_hz: filter.muHighHz,
      beta_low_hz: filter.betaLowHz,
      beta_high_hz: filter.betaHighHz,
      regularization_c: filter.regularizationC,
      neighbors: filter.neighbors,
      gamma: filter.gamma,
      max_depth: filter.maxDepth,
      n_estimators: filter.nEstimators,
      max_features: filter.maxFeatures,
      shrinkage: filter.shrinkage,
      folds: filter.folds,
      inner_folds: filter.innerFolds,
      search_budget: filter.searchBudget,
      test_fraction: filter.testFraction,
      validation_fraction: filter.validationFraction,
    },
  }).then((payload) => ({
    dataset: String(payload.dataset ?? ""),
    subject: String(payload.subject ?? ""),
    participants: Number(payload.participants) || 0,
    sampleRateHz: Number(payload.sample_rate_hz) || 0,
    seconds: Number(payload.seconds) || 0,
    channels: Array.isArray(payload.channels) ? payload.channels.map(String) : [],
    notchHz: Number(payload.notch_hz) || 0,
    bandLowHz: Number(payload.band_low_hz) || 0,
    bandHighHz: Number(payload.band_high_hz) || 0,
    order: Number(payload.order) || 0,
    spatial: String(payload.spatial ?? "none"),
    spatialChannels: Array.isArray(payload.spatial_channels)
      ? payload.spatial_channels.map(String)
      : [],
    spatialSignal: Array.isArray(payload.spatial_signal)
      ? (payload.spatial_signal as number[][])
      : [],
    alignment: String(payload.alignment ?? "none"),
    alignmentTrials: Number(payload.alignment_trials) || 0,
    alignmentChannels: Array.isArray(payload.alignment_channels)
      ? payload.alignment_channels.map(String)
      : [],
    alignmentSignal: Array.isArray(payload.alignment_signal)
      ? (payload.alignment_signal as number[][])
      : [],
    features: String(payload.features ?? "none"),
    featureDetail: String(payload.feature_detail ?? ""),
    featureChannels: Array.isArray(payload.feature_channels)
      ? payload.feature_channels.map(String)
      : [],
    featureSignal: Array.isArray(payload.feature_signal)
      ? (payload.feature_signal as number[][])
      : [],
    classifier: String(payload.classifier ?? "none"),
    evaluation: String(payload.evaluation ?? ""),
    accuracy: payload.accuracy == null ? null : Number(payload.accuracy),
    classifierDetail: String(payload.classifier_detail ?? ""),
    classifierFolds: Array.isArray(payload.classifier_folds)
      ? payload.classifier_folds.map((item) => {
          const fold = item as { name?: unknown; accuracy?: unknown };
          return { name: String(fold.name ?? ""), accuracy: Number(fold.accuracy) || 0 };
        })
      : [],
    metrics: metricMap(payload.metrics),
    confusionLabels: Array.isArray(payload.confusion_labels)
      ? payload.confusion_labels.map(String)
      : [],
    confusionMatrix: Array.isArray(payload.confusion_matrix)
      ? payload.confusion_matrix.map((row) =>
          Array.isArray(row) ? row.map((value) => Number(value) || 0) : [],
        )
      : [],
    raw: Array.isArray(payload.raw) ? (payload.raw as number[][]) : [],
    filtered: Array.isArray(payload.filtered) ? (payload.filtered as number[][]) : [],
  }));
}
