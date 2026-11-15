export function classifierDefaults(method: string): {
  nEstimators?: number | null;
  maxDepth?: number | null;
} {
  if (method === "random_forest") return { nEstimators: 80, maxDepth: null };
  if (method === "random_subspace_knn") return { nEstimators: 15, maxDepth: null };
  if (method === "boosted_trees") return { nEstimators: 40, maxDepth: 2 };
  if (method === "bagged_trees") return { nEstimators: 25, maxDepth: null };
  if (method === "rusboosted_trees") return { nEstimators: 30, maxDepth: 2 };
  if (method === "decision_tree") return { maxDepth: null };
  return {};
}

export function showsRegularization(method: string): boolean {
  return ["logistic_regression", "linear_svm", "svm_rbf"].includes(method);
}

export function showsNeighbors(method: string): boolean {
  return method === "knn" || method === "random_subspace_knn";
}

export function showsDepth(method: string): boolean {
  return [
    "decision_tree",
    "random_forest",
    "boosted_trees",
    "bagged_trees",
    "rusboosted_trees",
  ].includes(method);
}

export function showsEstimators(method: string): boolean {
  return [
    "random_forest",
    "random_subspace_knn",
    "boosted_trees",
    "bagged_trees",
    "rusboosted_trees",
  ].includes(method);
}
