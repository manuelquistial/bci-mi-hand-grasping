import type { GameBridge } from "@/game/bridge/game-bridge";
import { BootScene } from "@/game/scenes/BootScene";
import { OfflineScene } from "@/game/scenes/OfflineScene";
import { OnlineScene } from "@/game/scenes/OnlineScene";

export async function createCollectorGame(
  parent: HTMLElement,
  bridge: GameBridge,
  width: number,
  height: number,
): Promise<import("phaser").Game> {
  const Phaser = (await import("phaser")).default;
  const config: import("phaser").Types.Core.GameConfig = {
    type: Phaser.AUTO,
    parent,
    width,
    height,
    backgroundColor: "#88c943",
    scene: [BootScene, OfflineScene, OnlineScene],
    render: {
      antialias: true,
      antialiasGL: true,
      pixelArt: false,
      roundPixels: false,
    },
    physics: {
      default: "arcade",
      arcade: {
        gravity: { x: 0, y: 0 },
        debug: false,
      },
    },
    scale: {
      mode: Phaser.Scale.FIT,
      autoCenter: Phaser.Scale.CENTER_BOTH,
    },
  };
  const game = new Phaser.Game(config);
  game.registry.set("bridge", bridge);
  return game;
}

export function destroyGame(game: import("phaser").Game | null): void {
  game?.destroy(true);
}
