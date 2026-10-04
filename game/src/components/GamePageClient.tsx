"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";

import { GameBridge } from "@/game/bridge/game-bridge";
import { generateTrialSequence } from "@/game/utils/sequence";
import { useGameBridgeEvents } from "@/hooks/useGameBridgeEvents";
import { apiCreateSession } from "@/lib/api/client";
import {
  pipelineOnlineStart,
  pipelineOnlineStop,
  pipelineRecordStart,
  pipelineRecordStop,
} from "@/lib/pipeline/client";
import { useHudStore } from "@/state/hud-store";
import { usePersistenceStore } from "@/state/persistence-store";
import { useSettingsStore } from "@/state/settings-store";

import { GameControls } from "./GameControls";
import { SessionHUD } from "./SessionHUD";
import { SettingsPanel, type SetupScreen } from "./SettingsPanel";
import styles from "./game-page.module.css";

const PhaserGameContainer = dynamic(
  () =>
    import("@/components/PhaserGameContainer").then(
      (m) => m.PhaserGameContainer,
    ),
  { ssr: false },
);

type View = "setup" | "session";

async function enterFullscreen(): Promise<void> {
  if (document.fullscreenElement) return;
  try {
    await document.documentElement.requestFullscreen();
  } catch {
    // The session view still covers the window if the browser refuses.
  }
}

async function exitFullscreen(): Promise<void> {
  if (!document.fullscreenElement) return;
  try {
    await document.exitFullscreen();
  } catch {
    // Already left fullscreen.
  }
}

export function GamePageClient({ initialStep }: { initialStep: SetupScreen }) {
  const bridge = useMemo(() => new GameBridge(), []);
  const sessionIdRef = useRef<string | null>(null);
  const viewRef = useRef<View>("setup");
  useGameBridgeEvents(bridge, sessionIdRef);

  const settings = useSettingsStore((s) => s.settings);
  const setMode = useSettingsStore((s) => s.setMode);
  const setNested = useSettingsStore((s) => s.setNested);
  const lastError = usePersistenceStore((s) => s.lastError);
  const setLastError = usePersistenceStore((s) => s.setLastError);
  const sessionStatus = useHudStore((s) => s.sessionStatus);
  const [view, setView] = useState<View>("setup");
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    if (initialStep === "offline") setMode("offline");
    if (initialStep === "simulator" || initialStep === "real") {
      setMode("online");
      setNested("pipeline", {
        onlineSource: initialStep === "simulator" ? "simulator" : "cyton",
      });
    }
  }, [initialStep, setMode, setNested]);

  const showSession = (next: View) => {
    viewRef.current = next;
    setView(next);
  };

  useEffect(() => {
    if (sessionStatus !== "running") return;
    bridge.dispatchCommand({
      type: "UPDATE_SETTINGS",
      payload: { settings },
    });
  }, [bridge, sessionStatus, settings]);

  const stop = () => {
    if (viewRef.current !== "session") return;
    showSession("setup");
    bridge.dispatchCommand({ type: "STOP_SESSION" });
    const st = useSettingsStore.getState().settings;
    const md = useSettingsStore.getState().mode;
    const finish =
      md === "offline" ? pipelineRecordStop(st.pipeline) : pipelineOnlineStop(st.pipeline);
    void finish.catch((error: unknown) => {
      setLastError(error instanceof Error ? error.message : "Could not stop the pipeline");
    });
    void exitFullscreen();
  };
  const stopRef = useRef(stop);
  stopRef.current = stop;

  useEffect(() => {
    if (view !== "session") return;
    const onKey = (event: KeyboardEvent) => {
      if (event.code !== "Space") return;
      if (event.repeat) return;
      const target = event.target;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement
      ) {
        return;
      }
      event.preventDefault();
      stopRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [view]);

  const start = async () => {
    const st = useSettingsStore.getState().settings;
    const md = useSettingsStore.getState().mode;
    setLastError(null);
    setStarting(true);
    await enterFullscreen();
    try {
      const session = await apiCreateSession(md, st);
      sessionIdRef.current = session.id;
      if (md === "offline") {
        await pipelineRecordStart(st.pipeline, session.id);
      } else {
        await pipelineOnlineStart(st.pipeline);
      }
      const trials = generateTrialSequence(st, md);
      showSession("session");
      bridge.dispatchCommand({
        type: "START_SESSION",
        payload: {
          sessionId: session.id,
          mode: md,
          settings: st,
          trials,
        },
      });
    } catch (error: unknown) {
      showSession("setup");
      setLastError(error instanceof Error ? error.message : "Could not start session");
      void exitFullscreen();
    } finally {
      setStarting(false);
    }
  };

  if (view === "session") {
    return (
      <div className={styles.session}>
        <SessionHUD />
        <div className={styles.stage}>
          <PhaserGameContainer bridge={bridge} />
        </div>
        <p className={styles.stopHint}>Press space to stop</p>
      </div>
    );
  }

  return (
    <div className={styles.setup}>
      <SettingsPanel screen={initialStep} />
      <GameControls onStart={start} starting={starting} />
      {lastError ? (
        <p className={styles.error} role="alert">
          {lastError}
        </p>
      ) : null}
    </div>
  );
}
