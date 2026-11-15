import { describe, expect, it } from "vitest";

import { defaultOnlineSettings } from "@/types/settings/online";

import { OnlineMotionController, type AvatarMotion } from "./online-motion-controller";

function fakeAvatar() {
  const state = { x: 0 };
  const avatar: AvatarMotion = {
    getX: () => state.x,
    applyHorizontalStep: (dx) => (state.x += dx),
    applyHorizontalVelocity: (vx, dt) => (state.x += vx * dt),
  };
  return { state, avatar };
}

const command = (sequence: number, control: number, strategy?: string) => ({
  version: 2 as const,
  type: "bci_control" as const,
  trialId: "trial_0",
  sequence,
  sourceWindowEnd: null,
  issuedAt: 0,
  control,
  valid: true,
  strategy,
});

describe("OnlineMotionController", () => {
  it("steps once per new discrete command", () => {
    const motion = new OnlineMotionController();
    const { state, avatar } = fakeAvatar();
    const online = defaultOnlineSettings();
    motion.acceptCommand(command(1, 1, "discrete"), false);
    const now = performance.now();
    motion.step(now, 0.016, avatar, online);
    motion.step(now, 0.016, avatar, online);
    expect(state.x).toBe(96);
    motion.acceptCommand(command(2, -1, "discrete"), false);
    motion.step(now, 0.016, avatar, online);
    expect(state.x).toBe(0);
  });

  it("moves proportionally and ignores duplicate sequences", () => {
    const motion = new OnlineMotionController();
    const { state, avatar } = fakeAvatar();
    const online = { ...defaultOnlineSettings(), smoothing: 1 };
    motion.acceptCommand(command(1, 1), false);
    motion.acceptCommand(command(1, -1), false);
    motion.step(performance.now(), 0.05, avatar, online);
    expect(state.x).toBeGreaterThan(0);
  });

  it("holds when no sample or command is present", () => {
    const motion = new OnlineMotionController();
    const { state, avatar } = fakeAvatar();
    const frame = motion.step(1, 0.05, avatar, defaultOnlineSettings());
    expect(state.x).toBe(0);
    expect(frame.velocityX).toBe(0);
  });

  it("neutralises motion after collection", () => {
    const motion = new OnlineMotionController();
    const { state, avatar } = fakeAvatar();
    motion.acceptCommand(command(1, 1), false);
    motion.onCollected();
    motion.step(performance.now(), 0.05, avatar, defaultOnlineSettings());
    expect(state.x).toBe(0);
  });
});
