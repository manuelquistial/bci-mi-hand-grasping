import Phaser from "phaser";

import { DISPLAY_SIZES } from "@/game/config/displaySizes";
import { ASSET_KEYS } from "@/game/utils/assetKeys";
import { ensureTexture } from "@/game/utils/assetFallback";
import { fitSprite, getSpriteFitScale } from "@/game/utils/fitSprite";

export class Target {
  private glow: Phaser.GameObjects.Image;
  private coin: Phaser.GameObjects.Image;
  private pulseTweens: Phaser.Tweens.Tween[] = [];
  private visible = false;

  constructor(
    private readonly scene: Phaser.Scene,
    private readonly uiScale: number,
  ) {
    const glowKey = ensureTexture(scene, ASSET_KEYS.effects.glowRing, "coin", 80, 80);
    const coinKey = ensureTexture(scene, ASSET_KEYS.items.coin, "coin", 48, 48);

    this.glow = scene.add.image(0, 0, glowKey);
    fitSprite(
      this.glow,
      DISPLAY_SIZES.glowRing.width * uiScale,
      DISPLAY_SIZES.glowRing.height * uiScale,
    );
    this.glow.setAlpha(0.6);
    this.glow.setDepth(5);
    this.glow.setVisible(false);

    this.coin = scene.add.image(0, 0, coinKey);
    fitSprite(
      this.coin,
      DISPLAY_SIZES.coin.width * uiScale,
      DISPLAY_SIZES.coin.height * uiScale,
    );
    this.coin.setDepth(6);
    this.coin.setVisible(false);
  }

  showAt(x: number, y: number): void {
    this.coin.setPosition(x, y);
    this.glow.setPosition(x, y);
    this.coin.setVisible(true);
    this.glow.setVisible(true);
    this.coin.setAlpha(1);
    this.visible = true;
  }

  hide(): void {
    this.coin.setVisible(false);
    this.glow.setVisible(false);
    this.stopPulse();
    this.visible = false;
  }

  setHighlighted(highlighted: boolean): void {
    if (highlighted && this.visible) {
      this.startPulse();
    } else {
      this.stopPulse();
    }
  }

  setDimmed(dimmed: boolean): void {
    this.coin.setAlpha(dimmed ? 0.45 : 1);
    this.glow.setAlpha(dimmed ? 0.25 : 0.6);
  }

  destroy(): void {
    this.stopPulse();
    this.glow.destroy();
    this.coin.destroy();
  }

  private startPulse(): void {
    if (this.pulseTweens.length > 0) return;
    const coinBase = getSpriteFitScale(
      this.coin,
      DISPLAY_SIZES.coin.width * this.uiScale,
      DISPLAY_SIZES.coin.height * this.uiScale,
    );
    this.pulseTweens.push(
      this.scene.tweens.add({
        targets: this.coin,
        scale: { from: coinBase, to: coinBase * 1.12 },
        duration: 400,
        yoyo: true,
        repeat: -1,
      }),
    );
    const glowBase = getSpriteFitScale(
      this.glow,
      DISPLAY_SIZES.glowRing.width * this.uiScale,
      DISPLAY_SIZES.glowRing.height * this.uiScale,
    );
    this.pulseTweens.push(
      this.scene.tweens.add({
        targets: this.glow,
        scale: { from: glowBase, to: glowBase * 1.12 },
        duration: 400,
        yoyo: true,
        repeat: -1,
      }),
    );
  }

  private stopPulse(): void {
    this.pulseTweens.forEach((tween) => tween.stop());
    this.pulseTweens = [];
    fitSprite(
      this.coin,
      DISPLAY_SIZES.coin.width * this.uiScale,
      DISPLAY_SIZES.coin.height * this.uiScale,
    );
    fitSprite(
      this.glow,
      DISPLAY_SIZES.glowRing.width * this.uiScale,
      DISPLAY_SIZES.glowRing.height * this.uiScale,
    );
  }
}
