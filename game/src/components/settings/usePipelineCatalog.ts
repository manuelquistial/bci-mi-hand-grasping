"use client";

import { useEffect, useState } from "react";

import {
  pipelineListDatasets,
  pipelineListOnlineModels,
  type OnlineModelPack,
  type SimulatorDataset,
} from "@/lib/pipeline/client";
import { useSettingsStore } from "@/state/settings-store";

/** Loads simulator datasets and keeps the selected dataset valid. */
export function useDatasets(): { datasets: SimulatorDataset[]; error: string | null } {
  const setNested = useSettingsStore((s) => s.setNested);
  const [datasets, setDatasets] = useState<SimulatorDataset[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const baseUrl = useSettingsStore.getState().settings.pipeline.baseUrl;
    pipelineListDatasets(baseUrl)
      .then((listed) => {
        if (cancelled) return;
        setDatasets(listed);
        setError(null);
        const current = useSettingsStore.getState().settings.pipeline.dataset;
        if (listed.length > 0 && !listed.some((item) => item.id === current)) {
          setNested("pipeline", { dataset: listed[0].id });
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Could not load datasets");
      });
    return () => {
      cancelled = true;
    };
  }, [setNested]);

  return { datasets, error };
}

/** Loads LOSO online packs and keeps the selected registry valid. */
export function useLosoModels(): { models: OnlineModelPack[]; error: string | null } {
  const setNested = useSettingsStore((s) => s.setNested);
  const baseUrlSetting = useSettingsStore((s) => s.settings.pipeline.baseUrl);
  const [models, setModels] = useState<OnlineModelPack[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const baseUrl = useSettingsStore.getState().settings.pipeline.baseUrl;
    pipelineListOnlineModels(baseUrl, { losoOnly: true })
      .then((listed) => {
        if (cancelled) return;
        setModels(listed);
        setError(null);
        const current = useSettingsStore.getState().settings.pipeline.registry;
        if (listed.length > 0 && !listed.some((item) => item.id === current)) {
          setNested("pipeline", { registry: listed[0].id });
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Could not load LOSO models");
      });
    return () => {
      cancelled = true;
    };
  }, [setNested, baseUrlSetting]);

  return { models, error };
}
