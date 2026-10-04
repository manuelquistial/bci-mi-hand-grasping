import type { TrialDefinition } from "@/types/domain";
import type { SessionSettings } from "@/types/settings";

/** Payload stored in Phaser registry when a session starts from BootScene. */
export type CollectorStartPayload = {
  sessionId: string;
  mode: "offline" | "online";
  settings: SessionSettings;
  trials: TrialDefinition[];
};
