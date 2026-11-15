"use client";

import { useSettingsStore } from "@/state/settings-store";

import styles from "../settings-panel.module.css";
import { NumberField } from "./NumberField";
import { SettingsFieldset } from "./SettingsShell";

export function PreModelFeaturesSection() {
  const preModel = useSettingsStore((s) => s.settings.preModel);
  const setNested = useSettingsStore((s) => s.setNested);
  const hz = (
    label: string,
    value: number,
    key:
      | "thetaLowHz"
      | "thetaHighHz"
      | "muLowHz"
      | "muHighHz"
      | "betaLowHz"
      | "betaHighHz",
  ) => (
    <NumberField
      label={label}
      value={value}
      step={0.5}
      min={0}
      onChange={(next) => setNested("preModel", { [key]: next })}
    />
  );
  const fbcsp = (
    label: string,
    value: number,
    key: "fbcspLowHz" | "fbcspHighHz" | "fbcspBandHz" | "fbcspOrder" | "fbcspKeep",
    min: number,
  ) => (
    <NumberField
      label={label}
      value={value}
      step={1}
      min={min}
      onChange={(next) => setNested("preModel", { [key]: next })}
    />
  );
  return (
    <SettingsFieldset legend="Features">
      <label className={styles.row}>
        Method
        <select
          value={preModel.features}
          onChange={(e) =>
            setNested("preModel", { features: e.target.value as typeof preModel.features })
          }
        >
          <option value="none">None</option>
          <option value="ts">TS</option>
          <option value="tdp">TDP</option>
          <option value="psd">PSD</option>
          <option value="csp">CSP</option>
          <option value="fbcsp">FBCSP</option>
        </select>
      </label>
      {preModel.features === "ts" ? (
        <NumberField
          label="Points"
          value={preModel.tsPoints ?? 32}
          step={1}
          min={2}
          onChange={(value) => setNested("preModel", { tsPoints: value })}
        />
      ) : null}
      {preModel.features === "psd" ? (
        <>
          {hz("Theta low (Hz)", preModel.thetaLowHz ?? 4, "thetaLowHz")}
          {hz("Theta high (Hz)", preModel.thetaHighHz ?? 8, "thetaHighHz")}
          {hz("Mu low (Hz)", preModel.muLowHz ?? 8, "muLowHz")}
          {hz("Mu high (Hz)", preModel.muHighHz ?? 12, "muHighHz")}
          {hz("Beta low (Hz)", preModel.betaLowHz ?? 13, "betaLowHz")}
          {hz("Beta high (Hz)", preModel.betaHighHz ?? 30, "betaHighHz")}
        </>
      ) : null}
      {preModel.features === "csp" || preModel.features === "fbcsp" ? (
        <NumberField
          label="CSP pairs"
          value={preModel.cspPairs ?? 2}
          step={1}
          min={1}
          onChange={(value) => setNested("preModel", { cspPairs: value })}
        />
      ) : null}
      {preModel.features === "fbcsp" ? (
        <>
          {fbcsp("Band from (Hz)", preModel.fbcspLowHz ?? 4, "fbcspLowHz", 0)}
          {fbcsp("Band to (Hz)", preModel.fbcspHighHz ?? 40, "fbcspHighHz", 1)}
          {fbcsp("Band width (Hz)", preModel.fbcspBandHz ?? 4, "fbcspBandHz", 1)}
          {fbcsp("Band order", preModel.fbcspOrder ?? 4, "fbcspOrder", 1)}
          {fbcsp("Features kept", preModel.fbcspKeep ?? 4, "fbcspKeep", 1)}
        </>
      ) : null}
    </SettingsFieldset>
  );
}
