export type PipelineSettings = {
  baseUrl: string;
  onlineSource: "simulator" | "cyton";
  dataset: string;
  sessionDir: string;
  subject: string;
  /** Registry id/name for the frozen online pack (LOSO SVM by default). */
  registry: string;
  controlWsPort: number;
  controlStrategy:
    | "baseline_proba"
    | "discrete"
    | "temporal_discrete"
    | "temporal_proportional";
  controlWindowK: number;
  controlProportionalScale: number;
};

export const defaultPipelineSettings = (): PipelineSettings => ({
  baseUrl: "http://127.0.0.1:8765",
  onlineSource: "simulator",
  dataset: "shuqfa2024",
  sessionDir: "",
  subject: "S001",
  registry: "20261110-165457-euclidean-csp-linear_svm",
  controlWsPort: 8090,
  controlStrategy: "baseline_proba",
  controlWindowK: 3,
  controlProportionalScale: 1,
});
