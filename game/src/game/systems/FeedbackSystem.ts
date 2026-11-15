import Phaser from "phaser";

import { getAssetCrop } from "@/game/config/assetCrops";
import { DISPLAY_SIZES } from "@/game/config/displaySizes";
import { ASSET_KEYS } from "@/game/utils/assetKeys";
import { ensureTexture } from "@/game/utils/assetFallback";
import { fitSprite, getSpriteFitScale } from "@/game/utils/fitSprite";

export class FeedbackSystem {
  private successIcon: Phaser.GameObjects.Image | null = null;
  private sparkles: Phaser.GameObjects.Image[] = [];
  private feedbackTween: Phaser.Tweens.Tween | null = null;

  constructor(
    private readonly scene: Phaser.Scene,
    private readonly uiScale: number,
  ) {}

  showSuccess(x: number, y: number): void {
    this.showIcon(x, y, ASSET_KEYS.ui.successIcon, DISPLAY_SIZES.successIcon, true);
  }

  showFail(x: number, y: number): void {
    this.showIcon(x, y, ASSET_KEYS.ui.failIcon, DISPLAY_SIZES.failIcon, false);
  }

  clear(): void {
    this.feedbackTween?.stop();
    this.feedbackTween = null;

    // Kill tweens first; updates on a destroyed Image hit null frame.sourceSize.
    if (this.successIcon) {
      this.scene.tweens.killTweensOf(this.successIcon);
      this.successIcon.destroy();
      this.successIcon = null;
    }

    for (const sparkle of this.sparkles) {
      this.scene.tweens.killTweensOf(sparkle);
      sparkle.destroy();
    }
    this.sparkles = [];
  }

  destroy(): void {
    this.clear();
  }

  private showIcon(
    x: number,
    y: number,
    textureKey: string,
    size: { width: number; height: number },
    withSparkles: boolean,
  ): void {
    this.clear();
    const iconKey = ensureTexture(this.scene, textureKey, "generic", 64, 64);
    const crop = getAssetCrop(textureKey);
    const width = size.width * this.uiScale;
    const height = size.height * this.uiScale;

    this.successIcon = this.scene.add.image(
      x,
      y - DISPLAY_SIZES.feedbackIconOffsetY * this.uiScale,
      iconKey,
    );
    const targetScale = getSpriteFitScale(this.successIcon, width, height, crop);
    fitSprite(this.successIcon, width, height, crop);
    this.successIcon.setScale(0);
    this.successIcon.setDepth(20);
    this.successIcon.setAlpha(0);

    this.feedbackTween = this.scene.tweens.add({
      targets: this.successIcon,
      scale: targetScale,
      alpha: 1,
      duration: 300,
      ease: "Back.easeOut",
    });

    if (!withSparkles) return;

    const sparkleKey = ensureTexture(
      this.scene,
      ASSET_KEYS.effects.sparkle,
      "coin",
      24,
      24,
    );
    for (let i = 0; i < 4; i += 1) {
      const sparkle = this.scene.add.image(
        x + (Math.random() - 0.5) * 80 * this.uiScale,
        y + (Math.random() - 0.5) * 60 * this.uiScale,
        sparkleKey,
      );
      fitSprite(
        sparkle,
        DISPLAY_SIZES.sparkle.width * this.uiScale,
        DISPLAY_SIZES.sparkle.height * this.uiScale,
      );
      sparkle.setDepth(19);
      sparkle.setAlpha(0);
      this.sparkles.push(sparkle);
      this.scene.tweens.add({
        targets: sparkle,
        alpha: { from: 0, to: 1 },
        y: sparkle.y - 30 * this.uiScale,
        duration: 600,
        delay: i * 80,
        yoyo: true,
      });
    }
  }
}
