"use client";

import { useSettingsStore } from "@/state/settings-store";

import styles from "../settings-panel.module.css";
import { DatasetField, ServiceUrlField } from "./PipelineFields";
import { PreModelClassifierSection } from "./PreModelClassifierSection";
import { PreModelFeaturesSection } from "./PreModelFeaturesSection";
import { PreModelFilterSection } from "./PreModelFilterSection";
import { PreModelSpatialSection } from "./PreModelSpatialSection";
import { SettingsFieldset, SettingsShell } from "./SettingsShell";
import { useDatasets } from "./usePipelineCatalog";

function AlignmentSection() {
  const alignment = useSettingsStore((s) => s.settings.preModel.alignment);
  const setNested = useSettingsStore((s) => s.setNested);
  return (
    <SettingsFieldset legend="Alignment">
      <label className={styles.row}>
        Method
        <select
          value={alignment}
          onChange={(e) =>
            setNested("preModel", { alignment: e.target.value as typeof alignment })
          }
        >
          <option value="none">None</option>
          <option value="euclidean">Euclidean alignment</option>
        </select>
      </label>
    </SettingsFieldset>
  );
}

export function PreModelSettings() {
  const { datasets, error } = useDatasets();
  return (
    <SettingsShell
      title="Pre model"
      help="Offline, before epoching: notch, then bandpass. The default band is 8–30 Hz for left/right hand imagery. A spatial step can follow that filter."
    >
      <SettingsFieldset legend="Pipeline">
        <ServiceUrlField />
        <DatasetField datasets={datasets} error={error} />
      </SettingsFieldset>
      <PreModelFilterSection />
      <PreModelSpatialSection />
      <AlignmentSection />
      <PreModelFeaturesSection />
      <PreModelClassifierSection />
    </SettingsShell>
  );
}
