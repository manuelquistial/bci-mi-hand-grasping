"use client";

import { DatasetField, OnlineControlFields, ServiceUrlField } from "./PipelineFields";
import { SettingsFieldset, SettingsShell } from "./SettingsShell";
import { useDatasets, useLosoModels } from "./usePipelineCatalog";

export function SimulatorSettings() {
  const { datasets, error: datasetError } = useDatasets();
  const { models, error: registryError } = useLosoModels();
  return (
    <SettingsShell
      title="Simulator"
      help="Activity times come from the dataset. The frozen LOSO pack classifies each 2 s window and streams left/right probabilities to the game over WebSocket."
    >
      <SettingsFieldset legend="Pipeline">
        <ServiceUrlField />
        <DatasetField datasets={datasets} error={datasetError} />
        <OnlineControlFields models={models} error={registryError} />
      </SettingsFieldset>
    </SettingsShell>
  );
}
