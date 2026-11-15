"use client";

import { useEffect, useRef } from "react";

import { createCollectorGame, destroyGame } from "@/game/core/game-bootstrap";
import type { GameBridge } from "@/game/bridge/game-bridge";
import type { CollectorStartPayload } from "@/game/scenes/collector-session-types";

type Props = {
  bridge: GameBridge;
  /** Applied after Phaser boots so React Strict Mode remounts cannot drop START_SESSION. */
  startPayload: CollectorStartPayload;
};

export function PhaserGameContainer({ bridge, startPayload }: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const gameRef = useRef<import("phaser").Game | null>(null);

  useEffect(() => {
    const el = hostRef.current;
    if (!el) return;
    let cancelled = false;
    void (async () => {
      const game = await createCollectorGame(el, bridge);
      if (cancelled) {
        destroyGame(game);
        return;
      }
      gameRef.current = game;
      // Dispatch only once the BootScene loop can consume it.
      bridge.dispatchCommand({
        type: "START_SESSION",
        payload: startPayload,
      });
    })();
    return () => {
      cancelled = true;
      destroyGame(gameRef.current);
      gameRef.current = null;
    };
  }, [bridge, startPayload]);

  return (
    <div
      ref={hostRef}
      className="phaser-host"
      style={{ width: "100%", height: "100%", background: "#2f5a18" }}
    />
  );
}
