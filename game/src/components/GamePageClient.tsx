"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";

import { GameBridge } from "@/game/bridge/game-bridge";
import type { CollectorStartPayload } from "@/game/scenes/collector-session-types";
import { generateTrialSequence } from "@/game/utils/sequence";
import { useGameBridgeEvents } from "@/hooks/useGameBridgeEvents";
import { apiCreateSession } from "@/lib/api/client";
import {
  pipelineOnlineStart,
  pipelineOnlineStop,
  pipelinePreModelFilter,
  pipelineRecordStart,
  pipelineRecordStop,
  type FilterPreview,
} from "@/lib/pipeline/client";
import { useHudStore } from "@/state/hud-store";
import { usePersistenceStore } from "@/state/persistence-store";
import { useSettingsStore } from "@/state/settings-store";

import { FilterCharts } from "./FilterCharts";
import { GameControls } from "./GameControls";
import { SettingsPanel, type SetupScreen } from "./SettingsPanel";
import styles from "./game-page.module.css";

const PhaserGameContainer = dynamic(
  () =>
    import("@/components/PhaserGameContainer").then(
      (m) => m.PhaserGameContainer,
    ),
  { ssr: false },
);

type View = "setup" | "session" | "charts";

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
  const [preview, setPreview] = useState<FilterPreview | null>(null);
  const [startPayload, setStartPayload] = useState<CollectorStartPayload | null>(
    null,
  );

  useEffect(() => {
    if (view !== "charts") return;
    window.scrollTo(0, 0);
  }, [view]);

  useEffect(() => {
    if (initialStep === "offline") setMode("offline");
    if (initialStep === "simulator" || initialStep === "real") {
      setMode("online");
      setNested("pipeline", {
        onlineSource: initialStep === "real" ? "cyton" : "simulator",
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

  const leaveSession = (opts: { stopPipeline: boolean; stopPhaser: boolean }) => {
    if (viewRef.current !== "session") return;
    setStartPayload(null);
    showSession("setup");
    if (opts.stopPhaser) {
      bridge.dispatchCommand({ type: "STOP_SESSION" });
    }
    if (opts.stopPipeline) {
      const st = useSettingsStore.getState().settings;
      const md = useSettingsStore.getState().mode;
      const finish =
        md === "offline"
          ? pipelineRecordStop(st.pipeline)
          : pipelineOnlineStop(st.pipeline);
      void finish.catch((error: unknown) => {
        setLastError(
          error instanceof Error ? error.message : "Could not stop the pipeline",
        );
      });
    }
    void exitFullscreen();
    useHudStore.setState({ sessionStatus: "idle", bciStatusLine: null });
  };

  // Simulator/online finished all trials → leave Phaser and return to setup.
  useEffect(() => {
    if (sessionStatus !== "completed") return;
    if (viewRef.current !== "session") return;
    leaveSession({ stopPipeline: true, stopPhaser: false });
    // leaveSession is stable enough via viewRef; avoid re-binding loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionStatus]);

  const stop = () => {
    leaveSession({ stopPipeline: true, stopPhaser: true });
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
    try {
      if (initialStep === "pre-model") {
        const next = await pipelinePreModelFilter(st);
        setPreview(next);
        viewRef.current = "charts";
        setView("charts");
        return;
      }
      await enterFullscreen();
      const session = await apiCreateSession(md, st);
      sessionIdRef.current = session.id;
      if (md === "offline") {
        await pipelineRecordStart(st.pipeline, session.id);
      } else {
        const online = await pipelineOnlineStart(st.pipeline);
        const wsUrl = online.control_ws_url;
        if (typeof wsUrl === "string" && wsUrl) {
          useSettingsStore.getState().setNested("online", {
            inputAdapter: "websocket",
            websocketUrl: wsUrl,
          });
        }
      }
      // Fresh settings after WS URL update (do not reuse the stale `st` snapshot).
      const live = useSettingsStore.getState().settings;
      const trials =
        md === "online" && live.pipeline.onlineSource === "simulator"
          ? []
          : generateTrialSequence(live, md);
      const payload: CollectorStartPayload = {
        sessionId: session.id,
        mode: md,
        settings: live,
        trials,
      };
      setStartPayload(payload);
      showSession("session");
      // START_SESSION is dispatched by PhaserGameContainer after BootScene is ready.
    } catch (error: unknown) {
      setStartPayload(null);
      showSession("setup");
      setLastError(error instanceof Error ? error.message : "Could not start session");
      void exitFullscreen();
    } finally {
      setStarting(false);
    }
  };

  if (view === "charts" && preview) {
    return (
      <div className={styles.chartsView}>
        <button
          type="button"
          className={styles.back}
          onClick={() => {
            viewRef.current = "setup";
            setView("setup");
          }}
        >
          Back
        </button>
        <FilterCharts preview={preview} />
      </div>
    );
  }

  if (view === "session" && startPayload) {
    return (
      <div className={styles.session}>
        <div className={styles.stage}>
          <PhaserGameContainer bridge={bridge} startPayload={startPayload} />
        </div>
        <p className={styles.stopHint}>Press space to stop</p>
      </div>
    );
  }

  return (
    <div className={styles.setup}>
      <SettingsPanel screen={initialStep} />
      <GameControls
        onStart={start}
        starting={starting}
        label={initialStep === "pre-model" ? "Start pre model" : "Start session"}
      />
      {lastError ? (
        <p className={styles.error} role="alert">
          {lastError}
        </p>
      ) : null}
    </div>
  );
}
