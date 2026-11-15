import { getJson } from "./http";

export type SimulatorDataset = {
  id: string;
  label: string;
  dataset: string;
  sessionDir: string;
};

export async function pipelineListDatasets(baseUrl: string): Promise<SimulatorDataset[]> {
  const payload = (await getJson(baseUrl, "/v1/datasets")) as {
    datasets?: Array<Partial<SimulatorDataset> & { id: string; label: string }>;
  };
  return (payload.datasets ?? []).map((item) => ({
    id: item.id,
    label: item.label,
    dataset: item.dataset || item.id,
    sessionDir: item.sessionDir ?? "",
  }));
}

export type SimulatorSubject = {
  id: string;
  label: string;
};

export async function pipelineListSubjects(
  baseUrl: string,
  dataset: string,
): Promise<SimulatorSubject[]> {
  const payload = (await getJson(
    baseUrl,
    `/v1/subjects?dataset=${encodeURIComponent(dataset)}`,
  )) as {
    subjects?: Array<Partial<SimulatorSubject> & { id: string }>;
  };
  return (payload.subjects ?? []).map((item) => ({
    id: item.id,
    label: item.label || item.id,
  }));
}

export type OnlineModelPack = {
  id: string;
  name: string;
  label: string;
  heldOut: boolean;
  onlineSubject: string;
  dataset: string;
  controlClassifier: string;
};

/** List frozen online packs. Game UI uses ``losoOnly`` to hide leak/pooled runs. */
export async function pipelineListOnlineModels(
  baseUrl: string,
  options: { losoOnly?: boolean } = {},
): Promise<OnlineModelPack[]> {
  const loso = options.losoOnly === false ? "0" : "1";
  const payload = (await getJson(baseUrl, `/v1/online/models?loso=${loso}`)) as {
    models?: Array<{
      id: string;
      name?: string;
      label?: string;
      held_out?: boolean;
      online_subject?: string;
      dataset?: string;
      control_classifier?: string;
    }>;
  };
  return (payload.models ?? []).map((item) => ({
    id: item.id,
    name: item.name || item.id,
    label: item.label || item.name || item.id,
    heldOut: Boolean(item.held_out),
    onlineSubject: item.online_subject || "",
    dataset: item.dataset || "",
    controlClassifier: item.control_classifier || "",
  }));
}
