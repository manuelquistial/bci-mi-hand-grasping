export const ASSET_KEYS = {
  background: {
    grass: "background.grass",
  },
  character: {
    idle: "character.idle",
    walkRight: "character.walkRight",
    walkLeft: "character.walkLeft",
    success: "character.success",
  },
  items: {
    coin: "items.coin",
  },
  ui: {
    stageBanner: "ui.stageBanner",
    panelLarge: "ui.panelLarge",
    panelSmall: "ui.panelSmall",
    arrowLeft: "ui.arrowLeft",
    arrowRight: "ui.arrowRight",
    successIcon: "ui.successIcon",
    failIcon: "ui.failIcon",
  },
  effects: {
    dashedLine: "effects.dashedLine",
    sparkle: "effects.sparkle",
    glowRing: "effects.glowRing",
  },
  hands: {
    hands: "hands.hands",
  },
} as const;

export const ASSET_PATHS: Record<string, string> = {
  [ASSET_KEYS.background.grass]: "/assets/background/grass_field.png",
  [ASSET_KEYS.character.idle]: "/assets/character/idle.png",
  [ASSET_KEYS.character.walkRight]: "/assets/character/walk_right.png",
  [ASSET_KEYS.character.walkLeft]: "/assets/character/walk_left.png",
  [ASSET_KEYS.character.success]: "/assets/character/success.png",
  [ASSET_KEYS.items.coin]: "/assets/items/coin.png",
  [ASSET_KEYS.ui.stageBanner]: "/assets/ui/stage_banner.png",
  [ASSET_KEYS.ui.panelLarge]: "/assets/ui/panel_large.png",
  [ASSET_KEYS.ui.panelSmall]: "/assets/ui/panel_small.png",
  [ASSET_KEYS.ui.arrowLeft]: "/assets/ui/arrow_left.png",
  [ASSET_KEYS.ui.arrowRight]: "/assets/ui/arrow_right.png",
  [ASSET_KEYS.ui.successIcon]: "/assets/ui/success_icon.png",
  [ASSET_KEYS.ui.failIcon]: "/assets/ui/fail_icon.png",
  [ASSET_KEYS.effects.dashedLine]: "/assets/effects/dashed_line.png",
  [ASSET_KEYS.effects.sparkle]: "/assets/effects/sparkle.png",
  [ASSET_KEYS.effects.glowRing]: "/assets/effects/glow_ring.png",
  [ASSET_KEYS.hands.hands]: "/assets/hands/hands.png",
};
