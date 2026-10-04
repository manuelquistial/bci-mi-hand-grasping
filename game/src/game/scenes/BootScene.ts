import Phaser from "phaser";

import type { GameBridge } from "@/game/bridge/game-bridge";
import { ASSET_PATHS } from "@/game/utils/assetKeys";
import { logMissingAsset } from "@/game/utils/assetFallback";

function applySmoothTextureFilter(scene: Phaser.Scene): void {
  Object.keys(ASSET_PATHS).forEach((key) => {
    if (scene.textures.exists(key)) {
      scene.textures.get(key).setFilter(Phaser.Textures.FilterMode.LINEAR);
    }
  });
}

export class BootScene extends Phaser.Scene {
  private missingAssets = new Set<string>();

  constructor() {
    super("BootScene");
  }

  preload(): void {
    Object.entries(ASSET_PATHS).forEach(([key, path]) => {
      this.load.image(key, path);
    });
    this.load.on("loaderror", (file: Phaser.Loader.File) => {
      this.missingAssets.add(file.key);
      logMissingAsset(file.key, String(file.url));
    });
  }

  create(): void {
    applySmoothTextureFilter(this);
    this.registry.set("missingAssets", this.missingAssets);
    this.add
      .text(16, 16, "The Collector — idle", {
        fontSize: "18px",
        color: "#1e2a14",
        fontFamily: "system-ui, sans-serif",
      })
      .setOrigin(0, 0);
  }

  update(): void {
    const bridge = this.registry.get("bridge") as GameBridge | undefined;
    if (!bridge) return;
    const cmds = bridge.consumeCommands();
    for (const cmd of cmds) {
      if (cmd.type === "START_SESSION") {
        this.registry.set("activeSession", cmd.payload);
        if (cmd.payload.mode === "offline") {
          this.scene.start("OfflineScene");
        } else {
          this.scene.start("OnlineScene");
        }
      } else {
        bridge.dispatchCommand(cmd);
      }
    }
  }
}
