import Phaser from "phaser";

import { CHARACTER_CROP } from "@/game/config/assetCrops";
import { DISPLAY_SIZES } from "@/game/config/displaySizes";
import type { TargetSide } from "@/types/domain";
import { ASSET_KEYS } from "@/game/utils/assetKeys";
import { ensureTexture } from "@/game/utils/assetFallback";
import { fitSprite, getSpriteFitScale } from "@/game/utils/fitSprite";

export type PlayerPose = "idle" | "left" | "right" | "success";

export class Player {
  private sprite: Phaser.GameObjects.Image;
  private currentTween: Phaser.Tweens.Tween | null = null;
  private walkTween: Phaser.Tweens.Tween | null = null;
  private pose: PlayerPose = "idle";
  private startX: number;
  private startY: number;

  constructor(
    private readonly scene: Phaser.Scene,
    startX: number,
    startY: number,
    private readonly uiScale: number,
  ) {
    this.startX = startX;
    this.startY = startY;
    const textureKey = ensureTexture(scene, ASSET_KEYS.character.idle, "player", 80, 100);
    this.sprite = scene.add.image(startX, startY, textureKey);
    this.sprite.setDepth(10);
    this.applyPlayerSize();
  }

  getX(): number {
    return this.sprite.x;
  }

  getY(): number {
    return this.sprite.y;
  }

  setX(x: number): void {
    this.sprite.x = x;
  }

  setPose(pose: PlayerPose): void {
    if (this.pose === pose) return;
    if (!this.sprite?.active || !this.sprite.frame) return;
    this.pose = pose;
    // Stop bob first without resizing — frame may be mid-tween / about to change.
    this.walkTween?.stop();
    this.walkTween = null;
    const key =
      pose === "left"
        ? ASSET_KEYS.character.walkLeft
        : pose === "right"
          ? ASSET_KEYS.character.walkRight
          : pose === "success"
            ? ASSET_KEYS.character.success
            : ASSET_KEYS.character.idle;
    this.sprite.setTexture(ensureTexture(this.scene, key, "player", 80, 100));
    this.applyPlayerSize();
    if (pose === "left" || pose === "right") {
      this.startWalkBob();
    }
  }

  moveTo(x: number, y: number, durationMs: number, onComplete?: () => void): void {
    this.currentTween?.stop();
    this.currentTween = this.scene.tweens.add({
      targets: this.sprite,
      x,
      y,
      duration: durationMs,
      ease: "Linear",
      onComplete: () => {
        this.currentTween = null;
        onComplete?.();
      },
    });
  }

  stopMove(): void {
    this.currentTween?.stop();
    this.currentTween = null;
  }

  resetPosition(): void {
    if (!this.sprite?.active) return;
    this.stopMove();
    this.walkTween?.stop();
    this.walkTween = null;
    this.sprite.setVisible(true);
    this.sprite.setAlpha(1);
    this.sprite.setPosition(this.startX, this.startY);
    // Force idle texture even if pose was already "idle".
    this.pose = "success";
    this.setPose("idle");
  }

  prepareWalk(direction: TargetSide): void {
    this.setPose(direction);
  }

  destroy(): void {
    this.stopMove();
    this.stopWalkAnimation();
    this.sprite.destroy();
  }

  private size(): { width: number; height: number } {
    return {
      width: DISPLAY_SIZES.player.width * this.uiScale,
      height: DISPLAY_SIZES.player.height * this.uiScale,
    };
  }

  private applyPlayerSize(): void {
    if (!this.sprite?.active || !this.sprite.frame) return;
    const { width, height } = this.size();
    fitSprite(this.sprite, width, height, CHARACTER_CROP);
  }

  private startWalkBob(): void {
    if (this.walkTween) return;
    if (!this.sprite?.active || !this.sprite.frame) return;
    const { width, height } = this.size();
    const base = getSpriteFitScale(this.sprite, width, height, CHARACTER_CROP);
    this.walkTween = this.scene.tweens.add({
      targets: this.sprite,
      scaleX: { from: base, to: base * 1.05 },
      scaleY: { from: base, to: base * 0.95 },
      duration: 200,
      yoyo: true,
      repeat: -1,
    });
  }

  private stopWalkAnimation(): void {
    this.walkTween?.stop();
    this.walkTween = null;
    this.applyPlayerSize();
  }
}
