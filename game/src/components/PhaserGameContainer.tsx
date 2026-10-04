"use client";

import { useEffect, useRef } from "react";

import { createCollectorGame, destroyGame } from "@/game/core/game-bootstrap";
import type { GameBridge } from "@/game/bridge/game-bridge";

type Props = {
  bridge: GameBridge;
  width: number;
  height: number;
};

export function PhaserGameContainer({ bridge, width, height }: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const gameRef = useRef<import("phaser").Game | null>(null);

  useEffect(() => {
    const el = hostRef.current;
    if (!el) return;
    let cancelled = false;
    void (async () => {
      const game = await createCollectorGame(el, bridge, width, height);
      if (cancelled) {
        destroyGame(game);
        return;
      }
      gameRef.current = game;
    })();
    return () => {
      cancelled = true;
      destroyGame(gameRef.current);
      gameRef.current = null;
    };
  }, [bridge, width, height]);

  return (
    <div
      ref={hostRef}
      className="phaser-host"
      style={{
        width: "100%",
        maxWidth: width,
        aspectRatio: `${width} / ${height}`,
        margin: "0 auto",
      }}
    />
  );
}
