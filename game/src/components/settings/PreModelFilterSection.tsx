"use client";

import { useSettingsStore } from "@/state/settings-store";

import { NumberField } from "./NumberField";
import { SettingsFieldset } from "./SettingsShell";

/** The text inputs here historically coerced an empty field to 0, not NaN. */
function emptyAsZero(value: number): number {
  return Number.isNaN(value) ? 0 : value;
}

export function PreModelFilterSection() {
  const preModel = useSettingsStore((s) => s.settings.preModel);
  const setNested = useSettingsStore((s) => s.setNested);
  return (
    <SettingsFieldset legend="Filter">
      <NumberField
        label="Notch (Hz)"
        value={preModel.notchHz}
        min={1}
        step={1}
        onChange={(value) => setNested("preModel", { notchHz: emptyAsZero(value) })}
      />
      <NumberField
        label="Low (Hz)"
        value={preModel.bandLowHz}
        min={0.1}
        step={0.5}
        onChange={(value) => setNested("preModel", { bandLowHz: emptyAsZero(value) })}
      />
      <NumberField
        label="High (Hz)"
        value={preModel.bandHighHz}
        min={0.1}
        step={0.5}
        onChange={(value) => setNested("preModel", { bandHighHz: emptyAsZero(value) })}
      />
      <NumberField
        label="Notch Q"
        value={preModel.notchQ ?? 30}
        min={0.1}
        step={1}
        onChange={(value) => setNested("preModel", { notchQ: emptyAsZero(value) })}
      />
      <NumberField
        label="Order"
        value={preModel.order}
        min={1}
        max={8}
        step={1}
        onChange={(value) => setNested("preModel", { order: emptyAsZero(value) })}
      />
    </SettingsFieldset>
  );
}
