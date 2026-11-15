import Phaser from "phaser";

import type { GameBridge } from "@/game/bridge/game-bridge";
import { ASSET_KEYS, ASSET_PATHS } from "@/game/utils/assetKeys";
import { ensureTexture } from "@/game/utils/assetFallback";
import { logMissingAsset } from "@/game/utils/assetFallback";

function applySmoothTextureFilter(scene: Phaser.Scene): void {
  Object.keys(ASSET_PATHS).forEach((key) => {
    if (scene.textures.exists(key)) {
      scene.textures.get(key).setFilter(Phaser.Textures.FilterMode.LINEAR);
    }
  });
}

/**
 * Preloads assets and waits for START_SESSION.
 * Shows the same grass field as the game — no debug "idle" placeholder.
 */
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
    const { width, height } = this.scale;
    const grassKey = ensureTexture(
      this,
      ASSET_KEYS.background.grass,
      "grass",
      width,
      height,
    );
    this.add
      .image(width / 2, height / 2, grassKey)
      .setDisplaySize(width, height)
      .setDepth(0);
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
