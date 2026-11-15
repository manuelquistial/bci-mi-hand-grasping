"use client";

import { OnlineControlFields, ServiceUrlField } from "./PipelineFields";
import { SessionFieldset, TimingsFieldset } from "./SessionFieldsets";
import { SettingsFieldset, SettingsShell } from "./SettingsShell";
import { useLosoModels } from "./usePipelineCatalog";

export function RealSettings() {
  const { models, error } = useLosoModels();
  return (
    <SettingsShell
      title="Real"
      help="Each trial comes from the session. During activity the pipeline classifies 2 s windows and drives the avatar through the same WebSocket bridge."
    >
      <SettingsFieldset legend="Pipeline">
        <ServiceUrlField />
        <OnlineControlFields models={models} error={error} />
      </SettingsFieldset>
      <TimingsFieldset showFeedback />
      <SessionFieldset />
    </SettingsShell>
  );
}
