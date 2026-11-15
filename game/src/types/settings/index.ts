import type { SessionMode } from "../domain";

import { defaultExperimentalSettings, type ExperimentalSettings } from "./experimental";
import { defaultGeneralSettings, type GeneralSettings } from "./general";
import { defaultOfflineSettings, type OfflineSettings } from "./offline";
import { defaultOnlineSettings, type OnlineSettings } from "./online";
import { defaultPipelineSettings, type PipelineSettings } from "./pipeline";
import { defaultPreModelSettings, type PreModelSettings } from "./preModel";

export type { ExperimentalSettings, SequenceMode } from "./experimental";
export type { GeneralSettings } from "./general";
export type { OfflineSettings } from "./offline";
export type { OnlineSettings, TargetSpawnMode } from "./online";
export type { PipelineSettings } from "./pipeline";
export type { PreModelSettings } from "./preModel";

export type SessionSettings = {
  general: GeneralSettings;
  offline: OfflineSettings;
  online: OnlineSettings;
  experimental: ExperimentalSettings;
  pipeline: PipelineSettings;
  preModel: PreModelSettings;
};

export const defaultSessionSettings = (): SessionSettings => ({
  general: defaultGeneralSettings(),
  offline: defaultOfflineSettings(),
  online: defaultOnlineSettings(),
  experimental: defaultExperimentalSettings(),
  pipeline: defaultPipelineSettings(),
  preModel: defaultPreModelSettings(),
});

export function settingsWithModeSnapshot(
  settings: SessionSettings,
  mode: SessionMode,
): Record<string, unknown> {
  return { ...settings, mode, snapshotAt: new Date().toISOString() };
}
