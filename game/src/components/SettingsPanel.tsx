"use client";

import { OfflineSettings } from "./settings/OfflineSettings";
import { PreModelSettings } from "./settings/PreModelSettings";
import { RealSettings } from "./settings/RealSettings";
import { SimulatorSettings } from "./settings/SimulatorSettings";

export type SetupScreen = "offline" | "pre-model" | "simulator" | "real";

export function SettingsPanel({ screen }: { screen: SetupScreen }) {
  switch (screen) {
    case "offline":
      return <OfflineSettings />;
    case "simulator":
      return <SimulatorSettings />;
    case "pre-model":
      return <PreModelSettings />;
    case "real":
      return <RealSettings />;
  }
}
