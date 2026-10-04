import { ASSET_KEYS } from "@/game/utils/assetKeys";

/** Normalized crop (0–1) of the visible content inside each PNG. */
export type NormalizedCrop = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export const CHARACTER_CROP: NormalizedCrop = {
  x: 0.27,
  y: 0.06,
  width: 0.46,
  height: 0.78,
};

export const STAGE_BANNER_CROP: NormalizedCrop = {
  x: 0.01,
  y: 0.27,
  width: 0.98,
  height: 0.39,
};

export const ASSET_CROPS: Partial<Record<string, NormalizedCrop>> = {
  [ASSET_KEYS.character.idle]: CHARACTER_CROP,
  [ASSET_KEYS.character.walkRight]: CHARACTER_CROP,
  [ASSET_KEYS.character.walkLeft]: CHARACTER_CROP,
  [ASSET_KEYS.character.success]: CHARACTER_CROP,
  [ASSET_KEYS.ui.stageBanner]: STAGE_BANNER_CROP,
};

export function getAssetCrop(textureKey: string): NormalizedCrop | undefined {
  return ASSET_CROPS[textureKey];
}
