import Phaser from "phaser";

export type FallbackType =
  | "grass"
  | "player"
  | "coin"
  | "banner"
  | "arrowLeft"
  | "arrowRight"
  | "generic";

const FALLBACK_COLORS: Record<FallbackType, number> = {
  grass: 0x88c943,
  player: 0x4a90d9,
  coin: 0xffd700,
  banner: 0x8b5a2b,
  arrowLeft: 0xffffff,
  arrowRight: 0xffffff,
  generic: 0xcccccc,
};

export function createFallbackTexture(
  scene: Phaser.Scene,
  key: string,
  type: FallbackType,
  width = 64,
  height = 64,
): void {
  if (scene.textures.exists(key)) return;

  const graphics = scene.make.graphics({ x: 0, y: 0 }, false);
  const color = FALLBACK_COLORS[type];
  graphics.fillStyle(color, 1);

  switch (type) {
    case "grass":
      graphics.fillRect(0, 0, width, height);
      break;
    case "player":
      graphics.fillCircle(width / 2, height / 2, Math.min(width, height) / 2 - 2);
      break;
    case "coin":
      graphics.fillCircle(width / 2, height / 2, Math.min(width, height) / 2 - 4);
      graphics.lineStyle(2, 0xe6b800, 1);
      graphics.strokeCircle(width / 2, height / 2, Math.min(width, height) / 2 - 4);
      break;
    case "banner":
      graphics.fillRoundedRect(0, 0, width, height, 8);
      graphics.lineStyle(2, 0x6b4423, 1);
      graphics.strokeRoundedRect(0, 0, width, height, 8);
      break;
    case "arrowLeft":
      graphics.fillTriangle(
        width * 0.7,
        height * 0.2,
        width * 0.7,
        height * 0.8,
        width * 0.2,
        height * 0.5,
      );
      break;
    case "arrowRight":
      graphics.fillTriangle(
        width * 0.3,
        height * 0.2,
        width * 0.3,
        height * 0.8,
        width * 0.8,
        height * 0.5,
      );
      break;
    default:
      graphics.fillRect(0, 0, width, height);
  }

  graphics.generateTexture(key, width, height);
  graphics.destroy();
  scene.textures.get(key).setFilter(Phaser.Textures.FilterMode.LINEAR);
  console.warn(`[collector] fallback texture for ${key} (${type})`);
}

export function ensureTexture(
  scene: Phaser.Scene,
  key: string,
  fallbackType: FallbackType,
  width = 64,
  height = 64,
): string {
  if (!scene.textures.exists(key)) {
    createFallbackTexture(scene, key, fallbackType, width, height);
  }
  return key;
}

export function logMissingAsset(key: string, path: string): void {
  console.warn(`[collector] missing asset ${key} (${path})`);
}
