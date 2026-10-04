"use client";

import dynamic from "next/dynamic";
import { useRef, useMemo, useEffect } from "react";

import { GameBridge } from "@/game/bridge/game-bridge";
import { generateTrialSequence } from "@/game/utils/sequence";
import { useGameBridgeEvents } from "@/hooks/useGameBridgeEvents";
import { apiCreateSession } from "@/lib/api/client";
import { usePersistenceStore } from "@/state/persistence-store";
import { useHudStore } from "@/state/hud-store";
import { useSettingsStore } from "@/state/settings-store";

import { GameControls } from "./GameControls";
import { OnlineDebugPanel } from "./OnlineDebugPanel";
import { SessionHUD } from "./SessionHUD";
import { SettingsPanel } from "./SettingsPanel";
import styles from "./game-page.module.css";

const PhaserGameContainer = dynamic(
  () =>
    import("@/components/PhaserGameContainer").then(
      (m) => m.PhaserGameContainer,
    ),
  { ssr: false },
);

export function GamePageClient() {
  const bridge = useMemo(() => new GameBridge(), []);
  const sessionIdRef = useRef<string | null>(null);
  useGameBridgeEvents(bridge, sessionIdRef);

  const settings = useSettingsStore((s) => s.settings);
  const mode = useSettingsStore((s) => s.mode);
  const setMode = useSettingsStore((s) => s.setMode);
  const lastError = usePersistenceStore((s) => s.lastError);
  const sessionStatus = useHudStore((s) => s.sessionStatus);

  useEffect(() => {
    if (sessionStatus !== "running" && sessionStatus !== "paused") {
      return;
    }
    bridge.dispatchCommand({
      type: "UPDATE_SETTINGS",
      payload: { settings },
    });
  }, [bridge, sessionStatus, settings]);

  const start = async () => {
    const st = useSettingsStore.getState().settings;
    const md = useSettingsStore.getState().mode;
    const session = await apiCreateSession(md, st);
    sessionIdRef.current = session.id;
    const trials = generateTrialSequence(st, md);
    bridge.dispatchCommand({
      type: "START_SESSION",
      payload: {
        sessionId: session.id,
        mode: md,
        settings: st,
        trials,
      },
    });
  };

  const stop = () => {
    bridge.dispatchCommand({ type: "STOP_SESSION" });
  };

  const pause = () => bridge.dispatchCommand({ type: "PAUSE_SESSION" });
  const resume = () => bridge.dispatchCommand({ type: "RESUME_SESSION" });

  return (
    <div className={styles.layout}>
      <aside className={styles.sidebar}>
        <SettingsPanel />
        <GameControls
          onStart={start}
          onStop={stop}
          onPause={pause}
          onResume={resume}
          mode={mode}
          onModeChange={setMode}
        />
        {lastError ? (
          <p className={styles.error} role="alert">
            {lastError}
          </p>
        ) : null}
      </aside>
      <main className={styles.main}>
        <SessionHUD />
        <div className={styles.canvasWrap}>
          <PhaserGameContainer
            bridge={bridge}
            width={settings.general.width}
            height={settings.general.height}
          />
        </div>
        {mode === "online" ? <OnlineDebugPanel bridge={bridge} /> : null}
      </main>
    </div>
  );
}
