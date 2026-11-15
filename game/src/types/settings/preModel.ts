export type PreModelSettings = {
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

export const defaultPreModelSettings = (): PreModelSettings => ({
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
});
