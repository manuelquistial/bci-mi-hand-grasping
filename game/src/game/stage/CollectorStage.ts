import Phaser from "phaser";

import { getAssetCrop } from "@/game/config/assetCrops";
import { DISPLAY_SIZES } from "@/game/config/displaySizes";
import {
  COLLECT_RADIUS,
  movementDurationMs,
  playerStart,
  STAGE_LAYOUT,
  stageScale,
  stageX,
  stageY,
  targetPoint,
} from "@/game/config/stage-layout";
import { Player } from "@/game/entities/Player";
import { Target } from "@/game/entities/Target";
import { FeedbackSystem } from "@/game/systems/FeedbackSystem";
import type { TargetSide, TrialOutcome, TrialPhase } from "@/types/domain";
import { ASSET_KEYS } from "@/game/utils/assetKeys";
import { ensureTexture } from "@/game/utils/assetFallback";
import { fitSprite, setSpriteOriginTop } from "@/game/utils/fitSprite";

const RESET_PHASES: ReadonlySet<TrialPhase> = new Set<TrialPhase>(
  ["idle", "preparing", "fixation", "inter_trial_interval"],
);

const CUE_LABEL: Record<TargetSide, string> = {
  left: "Left",
  right: "Right",
};

/**
 * Shared visual stage: grass field, character, coin, cue and feedback.
 * Both offline and online sessions render through this.
 */
export class CollectorStage {
  readonly player: Player;
  readonly target: Target;
  readonly feedback: FeedbackSystem;

  private banner: Phaser.GameObjects.Image;
  private stageText: Phaser.GameObjects.Text;
  private cueText: Phaser.GameObjects.Text;
  private fixationCross: Phaser.GameObjects.Text;
  private dashedLine: Phaser.GameObjects.Image;
  private arrowCue: Phaser.GameObjects.Image | null = null;
  private hudCueY = 0;
  private readonly uiScale: number;

  constructor(
    private readonly scene: Phaser.Scene,
    private readonly width: number,
    private readonly height: number,
    bannerLabel: string,
  ) {
    this.uiScale = stageScale(width, height);
    this.createBackground();

    const start = playerStart(width, height);
    this.player = new Player(scene, start.x, start.y, this.uiScale);
    this.target = new Target(scene, this.uiScale);
    this.feedback = new FeedbackSystem(scene, this.uiScale);

    const bannerKey = ensureTexture(scene, ASSET_KEYS.ui.stageBanner, "banner", 400, 80);
    this.banner = scene.add
      .image(stageX(STAGE_LAYOUT.bannerX, width), stageY(STAGE_LAYOUT.bannerTopY, height), bannerKey)
      .setDepth(8);
    this.stageText = scene.add.text(0, 0, "", this.bannerTextStyle()).setOrigin(0.5).setDepth(9);

    this.fixationCross = scene.add
      .text(width / 2, height / 2, "+", this.hudTextStyle(DISPLAY_SIZES.fixationCross.fontSize))
      .setOrigin(0.5)
      .setDepth(15)
      .setVisible(false);

    this.cueText = scene.add
      .text(stageX(STAGE_LAYOUT.cueX, width), 0, "", this.hudTextStyle(DISPLAY_SIZES.cueFontSize))
      .setOrigin(0.5)
      .setDepth(15)
      .setVisible(false);

    const dashedKey = ensureTexture(scene, ASSET_KEYS.effects.dashedLine, "generic", 200, 8);
    this.dashedLine = scene.add.image(width / 2, start.y, dashedKey).setDepth(4).setVisible(false).setAlpha(0.4);

    this.setBanner(bannerLabel);
  }

  setBanner(label: string): void {
    const textStyle = this.bannerTextStyle();
    const measureText = this.scene.add.text(0, 0, label, textStyle).setVisible(false);
    const { minWidth, paddingX, widthScale, textOffsetY } = DISPLAY_SIZES.stageBanner;
    const bannerWidth =
      Math.max(minWidth, measureText.width + paddingX * 2) * widthScale * this.uiScale;
    measureText.destroy();

    const bannerCrop = getAssetCrop(ASSET_KEYS.ui.stageBanner);
    const frame = this.scene.textures.get(this.banner.texture.key).get();
    const contentWidth = bannerCrop ? frame.width * bannerCrop.width : frame.width;
    const contentHeight = bannerCrop ? frame.height * bannerCrop.height : frame.height;
    const bannerHeight = bannerWidth * (contentHeight / contentWidth);

    fitSprite(this.banner, bannerWidth, bannerHeight, bannerCrop);
    setSpriteOriginTop(this.banner, bannerCrop);
    this.banner.setPosition(
      stageX(STAGE_LAYOUT.bannerX, this.width),
      stageY(STAGE_LAYOUT.bannerTopY, this.height),
    );

    const bannerBounds = this.banner.getBounds();
    this.stageText
      .setText(label)
      .setOrigin(0.5)
      .setPosition(bannerBounds.centerX, bannerBounds.centerY + textOffsetY * this.uiScale);

    this.hudCueY =
      bannerBounds.bottom +
      (STAGE_LAYOUT.cueGap + STAGE_LAYOUT.cueOffsetY) * this.uiScale;
    this.syncCueHudPositions();
  }

  showForPhase(
    phase: TrialPhase,
    side: TargetSide | null,
    outcome: TrialOutcome | null,
  ): void {
    const showCue = phase === "cue" || phase === "control";
    const showTarget = showCue || phase === "feedback";

    this.fixationCross.setVisible(phase === "fixation");

    if (RESET_PHASES.has(phase)) {
      this.player.resetPosition();
      this.feedback.clear();
    } else if (phase === "control") {
      this.feedback.clear();
    }

    if (showTarget && side) {
      const point = targetPoint(side, this.width, this.height);
      this.target.showAt(point.x, point.y);
      this.target.setHighlighted(showCue);
      this.target.setDimmed(phase === "feedback");
    } else {
      this.target.hide();
    }

    if (showCue && side) {
      this.showCue(side);
      this.dashedLine.setVisible(phase === "control");
    } else {
      this.clearCue();
    }

    if (phase === "feedback" && side) {
      const point = targetPoint(side, this.width, this.height);
      if (outcome === "success") {
        this.player.setPose("success");
        this.feedback.showSuccess(point.x, point.y);
      } else {
        this.player.setPose("idle");
        this.feedback.showFail(point.x, point.y);
      }
    }
  }

  startAutoMove(side: TargetSide, speed: number, onComplete: () => void): void {
    const dest = targetPoint(side, this.width, this.height);
    const duration = movementDurationMs(dest.x - this.player.getX(), speed);
    this.player.prepareWalk(side);
    this.player.moveTo(dest.x, dest.y, duration, onComplete);
  }

  stopAutoMove(): void {
    this.player.stopMove();
  }

  applyHorizontalVelocity(velocityX: number, dt: number): number {
    const next = this.clampedPlayerX(velocityX * dt);
    if (Math.abs(velocityX) < 12) {
      this.player.setPose("idle");
    } else {
      this.player.setPose(velocityX < 0 ? "left" : "right");
    }
    return next;
  }

  /** One discrete step (px) toward left (−) or right (+). */
  applyHorizontalStep(deltaX: number): number {
    const next = this.clampedPlayerX(deltaX);
    if (Math.abs(deltaX) < 1) {
      this.player.setPose("idle");
    } else {
      this.player.setPose(deltaX < 0 ? "left" : "right");
    }
    return next;
  }

  /** Snap onto the coin so continuous control cannot overshoot. */
  snapToTarget(side: TargetSide): number {
    const point = targetPoint(side, this.width, this.height);
    this.player.setX(point.x);
    this.player.setPose("success");
    return point.x;
  }

  distanceTo(side: TargetSide): number {
    const point = targetPoint(side, this.width, this.height);
    return Math.hypot(this.player.getX() - point.x, this.player.getY() - point.y);
  }

  isAtTarget(side: TargetSide): boolean {
    return this.distanceTo(side) <= COLLECT_RADIUS * this.uiScale;
  }

  destroy(): void {
    this.player.destroy();
    this.target.destroy();
    this.feedback.destroy();
  }

  private showCue(side: TargetSide): void {
    this.cueText.setText(CUE_LABEL[side]).setVisible(true);
    const arrowKey = ensureTexture(
      this.scene,
      side === "left" ? ASSET_KEYS.ui.arrowLeft : ASSET_KEYS.ui.arrowRight,
      side === "left" ? "arrowLeft" : "arrowRight",
      64,
      64,
    );

    if (!this.arrowCue) {
      this.arrowCue = this.scene.add.image(0, 0, arrowKey).setDepth(15);
    } else {
      this.arrowCue.setTexture(arrowKey);
    }
    fitSprite(
      this.arrowCue,
      DISPLAY_SIZES.arrow.width * this.uiScale,
      DISPLAY_SIZES.arrow.height * this.uiScale,
    );
    this.arrowCue.setVisible(true);
    this.syncCueHudPositions();

    const dest = targetPoint(side, this.width, this.height);
    const start = playerStart(this.width, this.height);
    this.dashedLine.setPosition((start.x + dest.x) / 2, start.y);
    this.dashedLine.setDisplaySize(
      Math.abs(dest.x - start.x),
      DISPLAY_SIZES.dashedLine.height * this.uiScale,
    );
  }

  private clearCue(): void {
    this.cueText.setVisible(false);
    this.arrowCue?.setVisible(false);
    this.dashedLine.setVisible(false);
    this.target.setHighlighted(false);
  }

  private syncCueHudPositions(): void {
    const cueX = stageX(STAGE_LAYOUT.cueX, this.width);
    this.cueText.setPosition(cueX, this.hudCueY);
    this.arrowCue?.setPosition(cueX, this.hudCueY + STAGE_LAYOUT.cueArrowOffsetY * this.uiScale);
  }

  /** Move the player by `deltaX`, clamped to the stage margins; returns the new x. */
  private clampedPlayerX(deltaX: number): number {
    const margin = 48 * this.uiScale;
    const next = Phaser.Math.Clamp(this.player.getX() + deltaX, margin, this.width - margin);
    this.player.setX(next);
    return next;
  }

  private hudTextStyle(baseFontSize: number): Phaser.Types.GameObjects.Text.TextStyle {
    return {
      fontSize: `${baseFontSize * this.uiScale}px`,
      color: "#ffffff",
      fontStyle: "bold",
      stroke: "#2a3d1f",
      strokeThickness: 4,
    };
  }

  private bannerTextStyle(): Phaser.Types.GameObjects.Text.TextStyle {
    return {
      fontSize: `${DISPLAY_SIZES.bannerFontSize * this.uiScale}px`,
      color: "#ffffff",
      fontStyle: "bold",
      stroke: "#4a3018",
      strokeThickness: 3,
      align: "center",
    };
  }

  private createBackground(): void {
    const { width, height } = this;
    const grassKey = ensureTexture(this.scene, ASSET_KEYS.background.grass, "grass", width, height);
    this.scene.add.image(width / 2, height / 2, grassKey).setDisplaySize(width, height).setDepth(0);
  }
}
