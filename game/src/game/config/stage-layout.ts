import type { TargetSide } from "@/types/domain";

import { GAME_HEIGHT, GAME_WIDTH } from "./constants";

/** Positions authored for a 1280×720 field. Scaled to the live canvas. */
export const STAGE_LAYOUT = {
  playerStartX: 640,
  playerStartY: 420,
  leftTargetX: 250,
  rightTargetX: 1030,
  targetY: 420,
  bannerX: 640,
  bannerTopY: 10,
  cueGap: 24,
  cueOffsetY: -12,
  cueX: 640,
  cueArrowOffsetY: 80,
} as const;

/** How close the character must get to the coin to collect it. */
export const COLLECT_RADIUS = 78;

export function stageScale(width: number, height: number): number {
  return Math.min(width / GAME_WIDTH, height / GAME_HEIGHT);
}

export function stageX(authoredX: number, width: number): number {
  return (authoredX / GAME_WIDTH) * width;
}

export function stageY(authoredY: number, height: number): number {
  return (authoredY / GAME_HEIGHT) * height;
}

export function targetPoint(
  side: TargetSide,
  width: number,
  height: number,
): { x: number; y: number } {
  return {
    x: stageX(
      side === "left" ? STAGE_LAYOUT.leftTargetX : STAGE_LAYOUT.rightTargetX,
      width,
    ),
    y: stageY(STAGE_LAYOUT.targetY, height),
  };
}

export function playerStart(
  width: number,
  height: number,
): { x: number; y: number } {
  return {
    x: stageX(STAGE_LAYOUT.playerStartX, width),
    y: stageY(STAGE_LAYOUT.playerStartY, height),
  };
}

export function movementDurationMs(distancePx: number, speed: number): number {
  const safeSpeed = Math.max(1, speed);
  return Math.round((Math.abs(distancePx) / safeSpeed) * 1000);
}
