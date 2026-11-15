"use client";

import { useSettingsStore } from "@/state/settings-store";

import styles from "../settings-panel.module.css";
import { NumberField } from "./NumberField";
import { SettingsFieldset } from "./SettingsShell";

export function PreModelSpatialSection() {
  const preModel = useSettingsStore((s) => s.settings.preModel);
  const setNested = useSettingsStore((s) => s.setNested);
  return (
    <SettingsFieldset legend="Spatial">
      <label className={styles.row}>
        Method
        <select
          value={preModel.spatial}
          onChange={(e) =>
            setNested("preModel", { spatial: e.target.value as typeof preModel.spatial })
          }
        >
          <option value="none">None</option>
          <option value="car">CAR</option>
          <option value="hjorth">Hjorth</option>
          <option value="large_laplacian">Large Laplacian</option>
          <option value="spherical_spline">Spherical spline</option>
        </select>
      </label>
      {preModel.spatial === "hjorth" ? (
        <NumberField
          label="Neighborhood (°)"
          value={preModel.hjorthDegrees ?? 36}
          step={1}
          min={1}
          onChange={(value) => setNested("preModel", { hjorthDegrees: value })}
        />
      ) : null}
      {preModel.spatial === "large_laplacian" ? (
        <>
          <NumberField
            label="Radius (°)"
            value={preModel.laplacianRadiusDegrees ?? 70}
            step={1}
            min={1}
            onChange={(value) => setNested("preModel", { laplacianRadiusDegrees: value })}
          />
          <NumberField
            label="Sigma (°)"
            value={preModel.laplacianSigmaDegrees ?? 30}
            step={1}
            min={1}
            onChange={(value) => setNested("preModel", { laplacianSigmaDegrees: value })}
          />
        </>
      ) : null}
      {preModel.spatial === "spherical_spline" ? (
        <>
          <NumberField
            label="Legendre m"
            value={preModel.splineOrder ?? 4}
            step={1}
            min={2}
            onChange={(value) => setNested("preModel", { splineOrder: value })}
          />
          <NumberField
            label="Ridge"
            value={preModel.splineRidge ?? 0.00001}
            step={0.00001}
            min={0}
            onChange={(value) => setNested("preModel", { splineRidge: value })}
          />
        </>
      ) : null}
    </SettingsFieldset>
  );
}
