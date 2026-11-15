import type { ExternalTrialClock, ExternalTrialHost } from "@/game/systems/external-trial-clock";
import type { OnlineMotionController } from "@/game/systems/online-motion-controller";
import type { BCIWireMessage } from "@/lib/bci/validate-sample";

export type WireRouterDeps = {
  motion: OnlineMotionController;
  clock: ExternalTrialClock;
  host: ExternalTrialHost;
  collected(): boolean;
  debug(): boolean;
};

/** Routes pipeline WebSocket messages: bci_control vs trial events vs v1. */
export class WireMessageRouter {
  constructor(private readonly deps: WireRouterDeps) {}

  handle(message: BCIWireMessage): void {
    const { motion, clock, host } = this.deps;
    switch (message.kind) {
      case "v2_control":
        motion.acceptCommand(message.command, this.deps.collected());
        return;
      case "v1":
        motion.acceptLegacy(message);
        return;
      case "v2_trial":
        if (message.event.type === "trial_start") {
          motion.onTrialStart(message.event);
          if (this.deps.debug()) console.debug("[BCI] trial_start", message.event);
          clock.onTrialStart(message.event, host);
          return;
        }
        motion.onTrialEnd(message.event);
        if (this.deps.debug()) console.debug("[BCI] trial_end", message.event);
        if (!host.enabled()) return;
        clock.onTrialEnd(host);
        motion.clearActiveTrial();
        return;
    }
  }
}
