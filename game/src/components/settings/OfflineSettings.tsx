"use client";

import { ServiceUrlField } from "./PipelineFields";
import { SessionFieldset, TimingsFieldset } from "./SessionFieldsets";
import { SettingsFieldset, SettingsShell } from "./SettingsShell";

export function OfflineSettings() {
  return (
    <SettingsShell title="Offline">
      <SettingsFieldset legend="Pipeline">
        <ServiceUrlField />
      </SettingsFieldset>
      <TimingsFieldset showAutoMoveSpeed />
      <SessionFieldset />
    </SettingsShell>
  );
}
