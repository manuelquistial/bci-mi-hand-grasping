import type Phaser from "phaser";

import type { NormalizedCrop } from "@/game/config/assetCrops";

function applyCrop(
  sprite: Phaser.GameObjects.Image,
  crop: NormalizedCrop,
): { width: number; height: number } {
  const frameWidth = sprite.frame.width;
  const frameHeight = sprite.frame.height;

  const cropX = Math.round(frameWidth * crop.x);
  const cropY = Math.round(frameHeight * crop.y);
  const cropWidth = Math.round(frameWidth * crop.width);
  const cropHeight = Math.round(frameHeight * crop.height);

  sprite.setCrop(cropX, cropY, cropWidth, cropHeight);
  sprite.setOrigin(
    (cropX + cropWidth / 2) / frameWidth,
    (cropY + cropHeight / 2) / frameHeight,
  );

  return { width: cropWidth, height: cropHeight };
}

export function getSpriteFitScale(
  sprite: Phaser.GameObjects.Image,
  maxWidth: number,
  maxHeight: number,
  crop?: NormalizedCrop,
): number {
  const { width, height } = crop
    ? {
        width: Math.round(sprite.frame.width * crop.width),
        height: Math.round(sprite.frame.height * crop.height),
      }
    : { width: sprite.frame.width, height: sprite.frame.height };

  if (width <= 0 || height <= 0) {
    return 1;
  }

  return Math.min(maxWidth / width, maxHeight / height);
}

export function fitSprite(
  sprite: Phaser.GameObjects.Image,
  maxWidth: number,
  maxHeight: number,
  crop?: NormalizedCrop,
): Phaser.GameObjects.Image {
  if (crop) {
    applyCrop(sprite, crop);
  } else {
    sprite.setCrop(0, 0, sprite.frame.width, sprite.frame.height);
    sprite.setOrigin(0.5, 0.5);
  }

  const scale = getSpriteFitScale(sprite, maxWidth, maxHeight, crop);
  return sprite.setScale(scale);
}

/** Anchor by horizontal center and the top of the visible content. */
export function setSpriteOriginTop(
  sprite: Phaser.GameObjects.Image,
  crop?: NormalizedCrop,
): void {
  const frameWidth = sprite.frame.width;
  const frameHeight = sprite.frame.height;

  if (crop) {
    const cropX = Math.round(frameWidth * crop.x);
    const cropWidth = Math.round(frameWidth * crop.width);
    sprite.setOrigin((cropX + cropWidth / 2) / frameWidth, crop.y);
    return;
  }

  sprite.setOrigin(0.5, 0);
}
