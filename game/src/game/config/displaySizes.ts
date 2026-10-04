/** Visible size on the 1280×720 stage, after content crop. */
export const DISPLAY_SIZES = {
  player: { width: 110, height: 145 },
  coin: { width: 142, height: 142 },
  glowRing: { width: 165, height: 165 },
  fixationCross: { fontSize: 86 },
  stageBanner: { minWidth: 260, paddingX: 52, widthScale: 1.12, textOffsetY: -5 },
  arrow: { width: 180, height: 240 },
  successIcon: { width: 118, height: 158 },
  failIcon: { width: 118, height: 158 },
  feedbackIconOffsetY: 95,
  sparkle: { width: 28, height: 28 },
  dashedLine: { height: 6 },
  cueFontSize: 48,
  bannerFontSize: 28,
} as const;
